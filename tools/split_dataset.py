import os
import random
import argparse


def check_dataset(src_dir):
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f'原始数据集文件夹 "{src_dir}" 不存在')

    jpg_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    xml_files = [f for f in os.listdir(src_dir) if f.endswith('.xml')]

    if not jpg_files or not xml_files:
        raise FileNotFoundError(f'原始数据集文件夹 "{src_dir}" 中缺少 jpg 或 xml 文件')

    return jpg_files


def split_dataset(src_dir, output_dir, split_ratio):
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    # 检查原始数据集并获取所有 jpg 文件列表
    jpg_files = check_dataset(src_dir)

    # 创建训练集和验证集目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 打乱文件列表并按比例划分训练集和验证集
    random.shuffle(jpg_files)
    train_files = jpg_files[:int(len(jpg_files) * split_ratio)]
    val_files = jpg_files[int(len(jpg_files) * split_ratio):]

    # 创建硬链接到训练集和验证集目录
    for f in train_files:
        src_jpg = os.path.join(src_dir, f)
        dst_jpg = os.path.join(train_dir, f)
        src_xml = os.path.join(src_dir, f.replace('.jpg', '.xml'))
        dst_xml = os.path.join(train_dir, f.replace('.jpg', '.xml'))
        os.link(src_jpg, dst_jpg)
        os.link(src_xml, dst_xml)

    for f in val_files:
        src_jpg = os.path.join(src_dir, f)
        dst_jpg = os.path.join(val_dir, f)
        src_xml = os.path.join(src_dir, f.replace('.jpg', '.xml'))
        dst_xml = os.path.join(val_dir, f.replace('.jpg', '.xml'))
        os.link(src_jpg, dst_jpg)
        os.link(src_xml, dst_xml)

    print("数据集划分完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将数据集分为训练集和验证集')
    parser.add_argument('src_dir', help='原始数据集文件夹路径')
    parser.add_argument('output_dir', help='train 和 val 文件夹的父目录')
    parser.add_argument('--split_ratio',
                        type=float,
                        default=0.8,
                        help='训练集所占比例，默认为 0.8')

    args = parser.parse_args()

    split_dataset(args.src_dir, args.output_dir, args.split_ratio)
