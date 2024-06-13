# 参考
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch
# 先在Terminal运行：python -m visdom.server
# 再运行本文件

import argparse
import os
import datetime
import logging
import importlib
import shutil
import random
from tqdm import tqdm
import numpy as np
import time
import visdom
import torch
import warnings
warnings.filterwarnings('ignore')

from dataset.lasDataLoader import lasDataset
from PointNet2 import dataProcess


# PointNet



from PointNet2.pointnet_sem_seg import get_model as PNss
# from PointNet2.pointnet2_part_seg_msg import get_model as PNss
from PointNet2.pointnet_sem_seg import get_loss as PNloss
# from  PointNet2.pointnet2_sem_seg_msg import get_loss as PNloss

# PointNet++
from PointNet2.pointnet2_sem_seg import get_model as PN2SS
from PointNet2.pointnet2_sem_seg import get_loss as PN2loss
# from PointNet2.pointnet2_sem_seg_msg import get_model as PN2SS
# from PointNet2.pointnet2_sem_seg_msg import get_loss as PN2loss


# True为PointNet++
PN2bool = True
# PN2bool = False


# 当前文件的路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出 PointNet训练模型的路径: PointNet
dirModel1 = ROOT_DIR + '/trainModel/pointnet_model'
if not os.path.exists(dirModel1):
        os.makedirs(dirModel1)
# 输出 PointNet++训练模型的路径
dirModel2 = ROOT_DIR + '/trainModel/PointNet2_model'
if not os.path.exists(dirModel2):
        os.makedirs(dirModel2)

# 日志的路径
pathLog = os.path.join(ROOT_DIR, 'LOG_train.txt')

# 训练数据集的路径
pathDataset = os.path.join(ROOT_DIR, 'dataset/lasDatasetClassification/')

# 点云语义分割的类别名称：这里只分了2类
classNumber = 2
classes = ['非果荚', '果荚']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

# 日志和输出
def log_string(str):
    logger.info(str)
    print(str)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--pnModel', type=bool, default=True, help='True = PointNet++；False = PointNet')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 8]')
    parser.add_argument('--block_size', type=float, default='1.0', help='点云分块的尺寸')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run [default: 30]')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate [default: 0.001]') #原始是0.001
    parser.add_argument('--GPU', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or AdamW [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 1024]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')#可以试试小的，例如0.05
    parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of train dataset [default: 0.6]')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of validation dataset [default: 0.2]')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of test dataset [default: 0.2]')
    return parser.parse_args()


if __name__ == '__main__':
    # # python -m visdom.server
    visdomTL = visdom.Visdom()
    visdomTLwindow = visdomTL.line([0], [0], opts=dict(title='train_loss'))
    visdomVL = visdom.Visdom()
    visdomVLwindow = visdomVL.line([0], [0], opts=dict(title='validate_loss'))
    visdomTVL = visdom.Visdom(env='PointNet++')

    # region 随机数
    # 返回1～10000间的一个整数，作为随机种子 opt的类型为：<class 'argparse.Namespace'>
    manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    # 保证在有种子的情况下生成的随机数都是一样的
    random.seed(manualSeed)
    # 设置一个用于生成随机数的种子，返回的是一个torch.Generator对象
    torch.manual_seed(manualSeed)
    # endregion

    # region 创建日志文件
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(pathLog)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    #endregion

    #region 超参数
    args = parse_args()
    args.pnModel = PN2bool
    log_string('------------ hyper-parameter ------------')
    log_string(args)
    # 指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    pointNumber = args.npoint
    batchSize = args.batch_size
    blockSize = args.block_size
    trainRatio = args.train_ratio
    valRatio = args.val_ratio
    testRatio = args.test_ratio
    #endregion

    # region dataset
    # train data
    trainData = lasDataset(split='train', data_root=pathDataset, train_ratio=trainRatio, val_ratio=valRatio,test_ratio=testRatio,
                             num_point=pointNumber, block_size=blockSize, sample_rate=1.0, transform=None)
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    # Validation data
    testData = lasDataset(split='test',
                            data_root=pathDataset, num_point=pointNumber,train_ratio=trainRatio, val_ratio=valRatio,test_ratio=testRatio,
                            block_size=blockSize, sample_rate=1.0, transform=None)
    testDataLoader = torch.utils.data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    log_string("The number of training data is: %d" % len(trainData))
    log_string("The number of validation data is: %d" % len(testData))

    weights = torch.Tensor(trainData.labelweights).cuda()
    #endregion


    # region loading model：使用预训练模型或新训练
    modelSS = ''
    criterion = ''
    if PN2bool:
        modelSS = PN2SS(classNumber).cuda()
        criterion = PN2loss().cuda()
        modelSS.apply(inplace_relu)
    else:
        modelSS = PNss(classNumber).cuda()
        criterion = PNloss().cuda()
        modelSS.apply(inplace_relu)



    # 权重初始化
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        # else:
        #     torch.nn.init.xavier_normal_(m.weight.data)
        #     torch.nn.init.constant_(m.bias.data, 0.0)  #这和上面两行新加的

    # 判断是否存在模型，并选择是否采用预训练模型
    try:
        path_premodel = ''
        if PN2bool:
            path_premodel = os.path.join(dirModel2, 'best_model.pth')
        else:
            path_premodel = os.path.join(dirModel1, 'best_model.pth')
        checkpoint = torch.load(path_premodel)
        start_epoch = checkpoint['epoch']
        # print('pretrain epoch = '+str(start_epoch))
        modelSS.load_state_dict(checkpoint['model_state_dict'])
        log_string('!!!!!!!!!! Use pretrain model')
    except:
        log_string('...... starting new training ......')
        start_epoch = 0
        modelSS = modelSS.apply(weights_init)
    #endregion

    # 重新训练模型
    start_epoch = 0
    modelSS = modelSS.apply(weights_init)


    #region 训练的参数和选项

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            modelSS.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer=='AdamW':
        optimizer =torch.optim.AdamW(
            modelSS.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay_rate
        )
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0
    #endregion


    for epoch in range(start_epoch, args.epoch):
        # region Train on chopped scenes
        log_string('****** Epoch %d (%d/%s) ******' % (global_epoch + 1, epoch + 1, args.epoch))

        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum updated to: %f' % momentum)

        modelSS = modelSS.apply(lambda x: bn_momentum_adjust(x, momentum))
        modelSS = modelSS.train()
        #endregion

        # region 训练
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):# 0.9代表进度条的平滑度
            # 梯度归零
            optimizer.zero_grad()

            # xyzL
            points = points.data.numpy() # ndarray = bs,1024,6
            points[:, :, :3] = dataProcess.rotate_point_cloud_z(points[:, :, :3]) ### 数据增广
            points = torch.Tensor(points) # tensor = bs,1024,6
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1) # tensor = bs,6,1024

            # 预测结果
            seg_pred, trans_feat = modelSS(points) # tensor = bs,1024,2  # tensor = bs,512,16
            seg_pred = seg_pred.contiguous().view(-1, classNumber) # tensor = (bs*1024=)点数量,2

            # 真实标签
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy() # ndarray = (bs*1024=)点数量
            target = target.view(-1, 1)[:, 0] # tensor = (bs*1024=)点数量

            # loss
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()

            # 优化器来更新模型的参数
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy() # ndarray = (bs*1024=)点数量
            correct = np.sum(pred_choice == batch_label) # 预测正确的点数量

            total_correct += correct
            total_seen += (batchSize * pointNumber)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        # draw
        trainLoss = (loss_sum.item()) / num_batches
        visdomTL.line([trainLoss], [epoch+1], win=visdomTLwindow, update='append')
        #endregion

        # region 保存模型
        if epoch % 1 == 0:
            modelpath=''
            if PN2bool:
                modelpath = os.path.join(dirModel2, 'model' + str(epoch + 1) + '.pth')
            else:
                modelpath = os.path.join(dirModel1, 'model' + str(epoch + 1) + '.pth')
            state = {
                'epoch': epoch,
                'model_state_dict': modelSS.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, modelpath)
            logger.info('Save model...'+modelpath)
        #endregion

        # region Evaluate on chopped scenes
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(classNumber)
            total_seen_class = [0 for _ in range(classNumber)]
            total_correct_class = [0 for _ in range(classNumber)]
            total_iou_deno_class = [0 for _ in range(classNumber)]
            modelSS = modelSS.eval()

            log_string('****** Epoch Evaluation %d (%d/%s) ******' % (global_epoch + 1, epoch + 1, args.epoch))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy() # ndarray = bs,1024,6
                points = torch.Tensor(points) # tensor = bs,1024,6
                points, target = points.float().cuda(), target.long().cuda() # tensor = bs,1024,6 # tensor = bs,1024,1
                points = points.transpose(2, 1) # tensor = bs,6,1024

                seg_pred, trans_feat = modelSS(points) # tensor = bs,1024,2 # tensor = bs,512,16
                pred_val = seg_pred.contiguous().cpu().data.numpy() # ndarray = bs,1024,2
                seg_pred = seg_pred.contiguous().view(-1, classNumber) # tensor = bs*1024,2

                batch_label = target.cpu().data.numpy() # ndarray = bs,1024,1
                # 将数组重塑为 (bs, 1024)
                dim_size = batch_label.shape[1]
                new_shape = (-1, dim_size)
                batch_label = batch_label.reshape(new_shape) # ndarray = bs,1024

                target = target.view(-1, 1)[:, 0] # tensor = bs*1024
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2) # ndarray = bs,1024
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (batchSize * pointNumber)
                tmp, _ = np.histogram(batch_label, range(classNumber + 1))
                labelweights += tmp

                for l in range(classNumber):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(classNumber):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            # draw
            valLoss = (loss_sum.item()) / num_batches
            visdomVL.line([valLoss], [epoch+1], win=visdomVLwindow, update='append')

            # region 根据 mIoU确定最佳模型
            if mIoU >= best_iou:
                best_iou = mIoU
                bestmodelpath = ''
                if PN2bool:
                    bestmodelpath = os.path.join(dirModel2, 'best_model.pth')
                else:
                    bestmodelpath = os.path.join(dirModel1, 'best_model.pth')
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': modelSS.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, bestmodelpath)
                logger.info('Save best model......'+bestmodelpath)
            log_string('Best mIoU: %f' % best_iou)
            #endregion

        #endregion

        global_epoch += 1

        # draw
        visdomTVL.line(X=[epoch+1], Y=[trainLoss],name="train loss", win='line', update='append',
                       opts=dict(showlegend=True, markers=False,
                                 title='PointNet++ train validate loss',
                                 xlabel='epoch', ylabel='loss'))
        visdomTVL.line(X=[epoch+1], Y=[valLoss], name="train loss", win='line', update='append')

    log_string('-------------------------------------------------\n\n')