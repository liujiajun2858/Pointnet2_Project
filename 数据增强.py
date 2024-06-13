
####！！！
##要是运行不了就改laspy版本，例如==1.7.0，但是源代码这版本又太低，得下载高的,比方卸载再重下，就会下2.x的
import os
import numpy as np
import laspy.file as lp

def random_translate_flip_crop(las_file, output_file, translate_range=(0.1, 0.1, 0.1),
                               flip_prob=0.5, crop_ratio=0.5):
    # 读取原始LAS文件
    with lp.File(las_file, mode='r') as f:
        num_points = f.header.point_records_count
        bbox = f.header.min, f.header.max

        # 创建一个新的LAS文件用于写入增强后的数据
        with lp.File(output_file, mode='w', header=f.header) as out_f:
            # 读取X, Y, Z, RGB, Classification
            x = f.x.copy()
            y = f.y.copy()
            z = f.z.copy()
            rgb = f.red.copy(), f.green.copy(), f.blue.copy()
            classification = f.classification.copy()

            # 应用随机平移
            tx, ty, tz = np.random.uniform(-translate_range[0], translate_range[0], 1), \
                         np.random.uniform(-translate_range[1], translate_range[1], 1), \
                         np.random.uniform(-translate_range[2], translate_range[2], 1)
            x += tx
            y += ty
            z += tz

            # 确保平移后的点仍在原始边界框内
            x = np.clip(x, bbox[0][0], bbox[1][0])
            y = np.clip(y, bbox[0][1], bbox[1][1])
            z = np.clip(z, bbox[0][2], bbox[1][2])

            # 应用随机翻转（这里以X轴为例）
            if np.random.rand() < flip_prob:  # 以一定概率决定是否翻转
                x = bbox[1][0] - (x - bbox[0][0])

            # 应用随机裁剪（仅保留部分点）
            if crop_ratio < 1:
                num_to_keep = int(num_points * crop_ratio)
                indices_to_keep = np.random.choice(num_points, num_to_keep, replace=False)
                x = x[indices_to_keep]
                y = y[indices_to_keep]
                z = z[indices_to_keep]
                rgb = [rgb_channel[indices_to_keep] for rgb_channel in rgb]
                classification = classification[indices_to_keep]

            # 写入增强后的数据到新的LAS文件
            out_f.x = x
            out_f.y = y
            out_f.z = z
            out_f.red = rgb[0]
            out_f.green = rgb[1]
            out_f.blue = rgb[2]
            out_f.classification = classification

# 设置输入和输出目录
input_dir = 'D:\\test0'
output_dir = 'D:\\test113'

# 遍历输入目录下的所有LAS文件
for filename in os.listdir(input_dir):
    if filename.endswith('.las'):
        las_file = os.path.join(input_dir, filename)
        # 为每个 LAS 文件生成两个增强后的版本
        for i in range(4):
            output_file = os.path.join(output_dir, f'{filename.split(".")[0]}_augmented_{i}.las')
            random_translate_flip_crop(las_file, output_file)














