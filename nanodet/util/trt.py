"""
参考以下代码实现：
- https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/inference.py
- https://stackoverflow.com/a/67492525
- https://github.com/hpc203/nanodet-plus-opencv/blob/d3cba74e539bcd1e8bcc77695289e1c3aeeee6c0/onnxruntime/main.py
- https://forums.developer.nvidia.com/t/how-to-use-tensorrt-by-the-multi-threading-package-of-python/123085/8
"""
import numpy as np
import cv2
import os
import tensorrt as trt
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda
import math
import threading
import atexit

TRT_LOGGER = trt.Logger()


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModel:
    def __init__(self,
                 engine_file_path,
                 label_path,
                 prob_threshold=0.4,
                 iou_threshold=0.3) -> None:
        assert os.path.exists(engine_file_path)
        assert os.path.exists(label_path)
        self.cfx = cuda.Device(0).make_context()
        with open(engine_file_path,
                  "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(
            self.trt_engine)

        # Load labels
        self.class_names = list(
            map(lambda x: x.strip(),
                open(label_path, 'r').readlines()))
        self.num_classes = len(self.class_names)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold

        self.mean = np.array([103.53, 116.28, 123.675],
                             dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395],
                            dtype=np.float32).reshape(1, 1, 3)
        self.input_shape = (self.trt_engine.get_binding_shape('new_input')[0],
                            self.trt_engine.get_binding_shape('new_input')[1])

        self.reg_max = int((self.trt_engine.get_binding_shape('output')[-1] -
                            self.num_classes) / 4) - 1
        self.project = np.arange(self.reg_max + 1)
        self.strides = (8, 16, 32, 64)
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid(
                (math.ceil(self.input_shape[0] / self.strides[i]),
                 math.ceil(self.input_shape[1] / self.strides[i])),
                self.strides[i])
            self.mlvl_anchors.append(anchors)

        self.registered = False

    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        return np.stack((xv, yv), axis=-1)

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / (self.std)
        return img

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] /
                                                      hw_scale)
                img = cv2.resize(srcimg, (neww, newh),
                                 interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img,
                                         0,
                                         0,
                                         left,
                                         self.input_shape[1] - neww - left,
                                         cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] *
                                 hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh),
                                 interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img,
                                         top,
                                         self.input_shape[0] - newh - top,
                                         0,
                                         0,
                                         cv2.BORDER_CONSTANT,
                                         value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape)
        return img, newh, neww, top, left

    def preprocess(self, img):
        return cv2.resize(img, self.input_shape)

    def postprocess(self, preds):
        preds = preds.reshape(-1, self.num_classes + (self.reg_max + 1) * 4)
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(self.strides, self.mlvl_anchors):
            cls_score, bbox_pred = preds[ind:(
                ind + anchors.shape[0]), :self.num_classes], preds[ind:(
                    ind + anchors.shape[0]), self.num_classes:]
            ind += anchors.shape[0]
            bbox_pred = self.softmax(bbox_pred.reshape(-1, self.reg_max + 1),
                                     axis=1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride

            bboxes = self.distance2bbox(anchors,
                                        bbox_pred,
                                        max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(),
                                   self.prob_threshold, self.iou_threshold)
        if len(indices) > 0:
            mlvl_bboxes = mlvl_bboxes[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            return mlvl_bboxes, confidences, classIds
        else:
            return np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1]) / max_shape[1]
            y1 = np.clip(y1, 0, max_shape[0]) / max_shape[0]
            x2 = np.clip(x2, 0, max_shape[1]) / max_shape[1]
            y2 = np.clip(y2, 0, max_shape[0]) / max_shape[0]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def allocate_buffers(engine):
        """Allocates host and device buffer for TRT engine inference.
        This function is similair to the one in common.py, but
        converts network outputs (which are np.float32) appropriately
        before writing them to Python buffer. This is needed, since
        TensorRT plugins doesn't support output type description, and
        in our particular case, we use NMS plugin as network output.
        Args:
            engine (trt.ICudaEngine): TensorRT engine
        Returns:
            inputs [HostDeviceMem]: engine input memory
            outputs [HostDeviceMem]: engine output memory
            bindings [int]: buffer to device bindings
            stream (cuda.Stream): cuda stream for engine inference synchronization
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(
                engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def _infer(self, img: np.ndarray):
        threading.Thread.__init__(self)
        self.cfx.push()
        if not self.registered:
            self.registered = True
            atexit.register(self.destroy)

        # Copy image data to host buffer
        np.copyto(self.inputs[0].host, img.ravel())
        # Transfer input data to the GPU.
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            for inp in self.inputs
        ]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            for out in self.outputs
        ]
        # Synchronize the stream
        self.stream.synchronize()

        self.cfx.pop()

        # Return only the host outputs.
        return [out.host for out in self.outputs][0]

    def destroy(self):
        self.cfx.pop()

    def __call__(self, img: np.ndarray):
        return self._infer(img)

    def infer(self, img: np.ndarray, overlay=True):
        blob = self.preprocess(img)
        outputs = self._infer(blob)
        bboxes, confidences, classIds = self.postprocess(outputs)
        if overlay:
            self.visualize(img, bboxes, confidences, classIds)
        return bboxes, confidences, classIds

    def visualize(self, img, bboxes, confidences, classIds):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, bbox in enumerate(bboxes):
            label = int(classIds[i])
            color = (_COLORS[label] * 255).astype(np.uint8).tolist()
            text = f"{self.class_names[label]}:{confidences[i]*100:.1f}%"
            txt_color = (0, 0,
                         0) if np.mean(_COLORS[label]) > 0.5 else (255, 255,
                                                                   255)
            txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            x1, y1, x2, y2 = bbox
            x1 *= img.shape[1]
            y1 *= img.shape[0]
            x2 *= img.shape[1]
            y2 *= img.shape[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color,
                          2)
            # 自动调整文本位置，使其在图像内
            if y1 - txt_size[1] - 2 < 0:
                y1 = txt_size[1] + 2
            if x1 + txt_size[0] + 2 > img.shape[1]:
                x1 = img.shape[1] - txt_size[0] - 2
            cv2.rectangle(img, (int(x1), int(y1 - txt_size[1] - 2)),
                          (int(x1 + txt_size[0]), int(y1)), color, -1)
            cv2.putText(img,
                        text, (int(x1), int(y1 - 2)),
                        font,
                        0.5,
                        txt_color,
                        thickness=1,
                        lineType=cv2.LINE_AA)
        return img


_COLORS = (np.array([
    0.000,
    0.447,
    0.741,
    0.850,
    0.325,
    0.098,
    0.929,
    0.694,
    0.125,
    0.494,
    0.184,
    0.556,
    0.466,
    0.674,
    0.188,
    0.301,
    0.745,
    0.933,
    0.635,
    0.078,
    0.184,
    0.300,
    0.300,
    0.300,
    0.600,
    0.600,
    0.600,
    1.000,
    0.000,
    0.000,
    1.000,
    0.500,
    0.000,
    0.749,
    0.749,
    0.000,
    0.000,
    1.000,
    0.000,
    0.000,
    0.000,
    1.000,
    0.667,
    0.000,
    1.000,
    0.333,
    0.333,
    0.000,
    0.333,
    0.667,
    0.000,
    0.333,
    1.000,
    0.000,
    0.667,
    0.333,
    0.000,
    0.667,
    0.667,
    0.000,
    0.667,
    1.000,
    0.000,
    1.000,
    0.333,
    0.000,
    1.000,
    0.667,
    0.000,
    1.000,
    1.000,
    0.000,
    0.000,
    0.333,
    0.500,
    0.000,
    0.667,
    0.500,
    0.000,
    1.000,
    0.500,
    0.333,
    0.000,
    0.500,
    0.333,
    0.333,
    0.500,
    0.333,
    0.667,
    0.500,
    0.333,
    1.000,
    0.500,
    0.667,
    0.000,
    0.500,
    0.667,
    0.333,
    0.500,
    0.667,
    0.667,
    0.500,
    0.667,
    1.000,
    0.500,
    1.000,
    0.000,
    0.500,
    1.000,
    0.333,
    0.500,
    1.000,
    0.667,
    0.500,
    1.000,
    1.000,
    0.500,
    0.000,
    0.333,
    1.000,
    0.000,
    0.667,
    1.000,
    0.000,
    1.000,
    1.000,
    0.333,
    0.000,
    1.000,
    0.333,
    0.333,
    1.000,
    0.333,
    0.667,
    1.000,
    0.333,
    1.000,
    1.000,
    0.667,
    0.000,
    1.000,
    0.667,
    0.333,
    1.000,
    0.667,
    0.667,
    1.000,
    0.667,
    1.000,
    1.000,
    1.000,
    0.000,
    1.000,
    1.000,
    0.333,
    1.000,
    1.000,
    0.667,
    1.000,
    0.333,
    0.000,
    0.000,
    0.500,
    0.000,
    0.000,
    0.667,
    0.000,
    0.000,
    0.833,
    0.000,
    0.000,
    1.000,
    0.000,
    0.000,
    0.000,
    0.167,
    0.000,
    0.000,
    0.333,
    0.000,
    0.000,
    0.500,
    0.000,
    0.000,
    0.667,
    0.000,
    0.000,
    0.833,
    0.000,
    0.000,
    1.000,
    0.000,
    0.000,
    0.000,
    0.167,
    0.000,
    0.000,
    0.333,
    0.000,
    0.000,
    0.500,
    0.000,
    0.000,
    0.667,
    0.000,
    0.000,
    0.833,
    0.000,
    0.000,
    1.000,
    0.000,
    0.000,
    0.000,
    0.143,
    0.143,
    0.143,
    0.286,
    0.286,
    0.286,
    0.429,
    0.429,
    0.429,
    0.571,
    0.571,
    0.571,
    0.714,
    0.714,
    0.714,
    0.857,
    0.857,
    0.857,
    0.000,
    0.447,
    0.741,
    0.314,
    0.717,
    0.741,
    0.50,
    0.5,
    0,
]).astype(np.float32).reshape(-1, 3))
