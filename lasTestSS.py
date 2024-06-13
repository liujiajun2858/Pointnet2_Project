# 参考
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import argparse
import sys
import os
import numpy as np
import logging
from pathlib import Path
import importlib
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

from dataset.lasDataLoader import testDatasetToPred

# PointNet
from PointNet2.pointnet_sem_seg import get_model as PNss
# PointNet++
from PointNet2.pointnet2_sem_seg import get_model as PN2SS


# True为PointNet++
PN2bool = True
# PN2bool = False


# region 函数：投票；日志输出；保存结果为las。
# 投票决定结果
def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


# 日志
def log_string(str):
    logger.info(str)
    print(str)


# save to LAS
import laspy
def SaveResultLAS(newLasPath, las_offsets, las_scales,point_np, rgb_np, label1, label2):
    # data
    newx = point_np[:, 0]+las_offsets[0]
    newy = point_np[:, 1]+las_offsets[1]
    newz = point_np[:, 2]+las_offsets[2]
    newred = rgb_np[:, 0]
    newgreen = rgb_np[:, 1]
    newblue = rgb_np[:, 2]
    newclassification = label1
    newuserdata = label2
    minx = min(newx)
    miny = min(newy)
    minz = min(newz)

    # create a new header
    newheader = laspy.LasHeader(point_format=3, version="1.2")
    newheader.scales = np.array([0.0001, 0.0001, 0.0001])
    newheader.offsets = np.array([minx, miny, minz])
    newheader.add_extra_dim(laspy.ExtraBytesParams(name="Classification", type=np.uint8))
    newheader.add_extra_dim(laspy.ExtraBytesParams(name="UserData", type=np.uint8))
    # create a Las
    newlas = laspy.LasData(newheader)
    newlas.x = newx
    newlas.y = newy
    newlas.z = newz
    newlas.red = newred
    newlas.green = newgreen
    newlas.blue = newblue
    newlas.Classification = newclassification
    newlas.UserData = newuserdata
    # write
    newlas.write(newLasPath)

# 超参数
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--pnModel', type=bool, default=True, help='True = PointNet++；False = PointNet')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--block_size', type=float, default='1.0', help='点云分块的尺寸')
    parser.add_argument('--GPU', type=str, default='0', help='specify GPU device')
    parser.add_argument('--num_point', type=int, default=1024, help='point number [default: 4096]')
    parser.add_argument('--num_votes', type=int, default=1,
                        help='aggregate segmentation scores with voting [default: 1]')
    return parser.parse_args()

#endregion


# 当前文件的路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型的路径
pathTrainModel = os.path.join(ROOT_DIR, 'trainModel/pointnet_model')
if PN2bool:
    pathTrainModel = os.path.join(ROOT_DIR, 'trainModel/PointNet2_model')

# 预测结果路径
visual_dir = ROOT_DIR + '/testResultPN/'
if PN2bool:
    visual_dir = ROOT_DIR + '/testResultPN2/'
visual_dir = Path(visual_dir)
visual_dir.mkdir(exist_ok=True)

# 日志的路径
pathLog = os.path.join(ROOT_DIR, 'LOG_test_eval.txt')

# 测试数据的路径
pathDataset = os.path.join(ROOT_DIR, 'dataset/lasDatasetClassification2/')

# 点云语义分割的类别名称：这里只分了2类
classNumber = 2
classes = ['非果荚', '果荚']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


if __name__ == '__main__':
    #region LOG info
    logger = logging.getLogger("test_eval")
    logger.setLevel(logging.INFO) #日志级别：DEBUG, INFO, WARNING, ERROR, 和 CRITICAL
    file_handler = logging.FileHandler(pathLog)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    #endregion

    #region 超参数
    args = parse_args()
    args.pnModel = PN2bool
    log_string('--- hyper-parameter ---')
    log_string(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    batchSize = args.batch_size
    blockSize = args.block_size
    pointNumber = args.num_point
    voteNumber = args.num_votes
    #endregion


    #region ---------- 加载语义分割的模型 ----------
    log_string("---------- Loading sematic segmentation model ----------")
    ssModel = ''
    if PN2bool:
        ssModel = PN2SS(classNumber).cuda()
    else:
        ssModel = PNss(classNumber).cuda()
    path_model = os.path.join(pathTrainModel, 'best_model.pth')
    checkpoint = torch.load(path_model)
    ssModel.load_state_dict(checkpoint['model_state_dict'])
    ssModel = ssModel.eval()
    #endregion


    # 模型推断（inference）或评估（evaluation）阶段，不需要计算梯度，而且关闭梯度计算可以显著减少内存占用，加速计算。
    log_string('--- Evaluation whole scene')
    with torch.no_grad():
        # IOU 结果
        total_seen_class = [0 for _ in range(classNumber)]
        total_correct_class = [0 for _ in range(classNumber)]
        total_iou_deno_class = [0 for _ in range(classNumber)]

        # 测试区域的所有文件
        testDataset = testDatasetToPred(split='test', data_root=pathDataset,block_points=pointNumber,block_size=blockSize)
        las_name = testDataset.file_list
        las_name = [x[:-4] for x in las_name] # 名称（无扩展名）
        testCount = len(las_name)
        # 遍历需要预测的物体
        for batch_idx in range(testCount):
            log_string("Inference [%d/%d] %s ..." % (batch_idx + 1, testCount, las_name[batch_idx]))
            # 数据
            las_points = testDataset.scene_points_list[batch_idx]
            # 真值
            las_label = testDataset.semantic_labels_list[batch_idx]
            las_labelR = np.reshape(las_label, (las_label.size, 1))
            # 预测标签
            vote_label_pool = np.zeros((las_label.shape[0], classNumber))

            # 同一物体多次预测
            for _ in tqdm(range(voteNumber), total=voteNumber):
                las_offsets, las_scales,scene_data, scene_label, scene_smpw, scene_point_index = testDataset[batch_idx] # 很慢
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + batchSize - 1) // batchSize
                batch_data = np.zeros((batchSize, pointNumber, 6))

                batch_label = np.zeros((batchSize, pointNumber))
                batch_point_index = np.zeros((batchSize, pointNumber))
                batch_smpw = np.zeros((batchSize, pointNumber))

                for sbatch in range(s_batch_num): # 慢
                    print('running...[%d/%d]'% (sbatch, s_batch_num))
                    start_idx = sbatch * batchSize
                    end_idx = min((sbatch + 1) * batchSize, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = ssModel(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    # 投票产生预测标签
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            # region  保存预测的结果
            # 预测标签
            pred_label = np.argmax(vote_label_pool, 1)
            pred_labelR = np.reshape(pred_label, (pred_label.size, 1))

            # 点云-真值-预测标签
            pcrgb_ll = np.hstack((las_points, las_labelR, pred_labelR))

            # # ---------- 保存成 txt ----------
            # pathTXT = os.path.join(visual_dir, las_name[batch_idx] + '.txt')
            # np.savetxt(pathTXT, pcrgb_ll, fmt='%f', delimiter='\t')
            # log_string('save:' + pathTXT)
            # ---------- 保存成 las ----------
            pathLAS = os.path.join(visual_dir, las_name[batch_idx] + '.las')
            SaveResultLAS(pathLAS, las_offsets[0], las_scales,pcrgb_ll[:,0:3], pcrgb_ll[:,3:6], pcrgb_ll[:,6], pcrgb_ll[:,7])
            log_string('save:' + pathLAS)
            # endregion


            # IOU 临时结果
            total_seen_class_tmp = [0 for _ in range(classNumber)]
            total_correct_class_tmp = [0 for _ in range(classNumber)]
            total_iou_deno_class_tmp = [0 for _ in range(classNumber)]
            
            for l in range(classNumber):
                total_seen_class_tmp[l] += np.sum((las_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (las_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (las_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float64) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (las_name[batch_idx], tmp_iou))


        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
        iou_per_class_str = '----- IoU -----\n'
        for l in range(classNumber):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

    log_string('--------------------------------------\n\n')