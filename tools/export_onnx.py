# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import onnx
import onnxsim
import torch
from onnx import helper, TensorProto
import onnx.numpy_helper
import numpy as np

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


def generate_ouput_names(head_cfg):
    cls_names, dis_names = [], []
    for stride in head_cfg.strides:
        cls_names.append("cls_pred_stride_{}".format(stride))
        dis_names.append("dis_pred_stride_{}".format(stride))
    return cls_names + dis_names


def main(config, model_path, output_path, input_shape=(320, 320)):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    if config.model.arch.backbone.name == "RepVGG":
        deploy_config = config.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[0], input_shape[1])
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )
    logger.log("finished exporting onnx ")

    logger.log("start simplifying onnx ")
    input_data = {"data": dummy_input.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(output_path, input_data=input_data)
    if flag:
        onnx.save(model_sim, output_path)
        logger.log("simplify onnx successfully")
    else:
        logger.log("simplify onnx failed")

    modify_input(output_path, output_path, input_shape)
    logger.log("finished modifying input")

def modify_input(model_path, output_path, input_shape=(320, 320)):
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    # 创建一个新的输入
    new_input_shape = [input_shape[0], input_shape[1], 3]
    new_input = helper.make_tensor_value_info("new_input", TensorProto.FLOAT, new_input_shape)

    # 添加一个维度扩展操作
    expand_dims_node = helper.make_node(
        'Unsqueeze',
        inputs=['new_input'],
        outputs=['expanded_input'],
        axes=[0],
        name='expand_dims_node'
    )

    # 添加一个转置操作
    transpose_node = helper.make_node(
        'Transpose',
        inputs=['expanded_input'],
        outputs=['transposed_input'],
        perm=[0, 3, 1, 2],
        name='transpose_node'
    )

    # 添加 normalize 操作
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape((3, 1, 1))
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape((3, 1, 1))

    mean_initializer = onnx.numpy_helper.from_array(mean, name='mean')
    std_initializer = onnx.numpy_helper.from_array(std, name='std')

    model.graph.initializer.extend([mean_initializer, std_initializer])

    normalize_node = helper.make_node(
        'Sub',
        inputs=['transposed_input', 'mean'],
        outputs=['normalized_input'],
        name='normalize_sub_node'
    )

    normalize_div_node = helper.make_node(
        'Div',
        inputs=['normalized_input', 'std'],
        outputs=[input_name],
        name='normalize_div_node'
    )

    # 将新的输入和扩展维度、转置操作以及 normalize 操作添加到图中
    model.graph.input.insert(0, new_input)
    model.graph.node.insert(0, expand_dims_node)
    model.graph.node.insert(1, transpose_node)
    model.graph.node.insert(2, normalize_node)
    model.graph.node.insert(3, normalize_div_node)

    # 找到悬空的输入节点 "data"
    input_to_remove = None
    for input_tensor in model.graph.input:
        if input_tensor.name == 'data':
            input_to_remove = input_tensor
            break

    # 从 graph.input 中移除悬空的输入节点
    if input_to_remove is not None:
        model.graph.input.remove(input_to_remove)

    onnx.save(model, output_path)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to onnx.",
    )
    parser.add_argument("--cfg_path", type=str, help="Path to .yml config file.")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to .ckpt model."
    )
    parser.add_argument(
        "--out_path", type=str, default="nanodet.onnx", help="Onnx model output path."
    )
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    out_path = args.out_path
    input_shape = args.input_shape
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best/model_best.ckpt")
    main(cfg, model_path, out_path, input_shape)
    print("Model saved to:", out_path)
