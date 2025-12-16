import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import argparse
from datetime import datetime


from lib.pvt import CAFE
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import time
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import warnings
warnings.filterwarnings("ignore")

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask,reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    IOU = 0.0

    for i in range(num1):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)


        cbam1_out, cbam2_out, cbam3_out, out4, res1, res2, res3 = model(image)


        res = F.interpolate(cbam1_out + cbam2_out + cbam3_out + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)

        mean_dice = mean_dice_np(target,input)
        DSC = DSC + mean_dice

        mean_iou = mean_iou_np(target, input)
        IOU = IOU + mean_iou

    return DSC / num1, IOU / num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack

            images = Variable(images).to(device)
            gts = Variable(gts).to(device)

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            cbam1_out, cbam2_out, cbam3_out, out4, P2, P3, P4 = model(images)


            loss_cbam1 = structure_loss(cbam1_out, gts)
            loss_cbam2 = structure_loss(cbam2_out, gts)
            loss_cbam3 = structure_loss(cbam3_out, gts)
            loss_out4 = structure_loss(out4, gts)

            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)

            loss = loss_cbam1 + loss_cbam2 + loss_cbam3 + loss_out4 + loss_P2 + loss_P3 + loss_P4

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}],'
                  ' loss-all:[{:0.4f}], lr:[{:0.7f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(), optimizer.param_groups[0]['lr']))
            logging.info('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}],'
                         ' loss-all:[{:0.4f}], lr:[{:0.7f}]'.
                         format(datetime.now(), epoch, opt.epoch, i, total_step,
                                loss_record.show(), optimizer.param_groups[0]['lr']))

    global dict_plot


    if (epoch + 1) % 1 == 0:
        meandice, meaniou = test(model, test_path, 'test')
        print('{} Epoch[{:03d}/{:03d}], meandice:{}, meaniou:{}'. format(datetime.now(), epoch, opt.epoch, meandice, meaniou))
        logging.info('{} Epoch[{:03d}/{:03d}], meandice:{}, meaniou:{}'. format(datetime.now(), epoch, opt.epoch, meandice, meaniou))
        if epoch>=1:
            dict_plot['test'].append(meandice)
            if meandice > best:
                print('#'*80)
                best = meandice
                torch.save(model.state_dict(), save_path + 'epoch' + str(epoch) + '-best_meandice_{:.3f}, meaniou_{:.3f}'.format(best, meaniou)+ '.pth')
                print(f'best meandice:{best}, meaniou:{meaniou}')
                logging.info('best meandice:{}, meaniou:{}'.format(best, meaniou))
                print('#'*80)

        # === 每个 epoch 保存 avg loss 到 txt 文件 ===
        avg_epoch_loss = loss_record.avg.item()
        with open(loss_txt_path, 'a') as f:
            f.write(f"{epoch},{avg_epoch_loss:.6f}\n")



if __name__ == '__main__':

    dict_plot = {'test':[]}

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=12, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.75, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default=r'../jinshan-dataset-926-risk/train-926-risk',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default=r'../jinshan-dataset-926-risk/val-926-risk',   # 此处是验证集，根据验证集保存的最高模型，在测试集上运行test.py进行测试，然后运行EVAL.py计算不同阈值下的指标
                        help='path to testing dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/MCD-Net')
    parser.add_argument('--log_path', type=str,
                        default='./log/MCD-Net/')

    opt = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

    logging.basicConfig(filename=f'{opt.log_path}train_log_{current_time}.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CAFE().to(device)

    # # Wrap model for multi-GPU training
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    print(optimizer)
    logging.info(optimizer)
    logging.info(opt)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    # 创建保存模型的目录
    save_path = (opt.train_save + '-' + current_time + '/')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建保存每轮损失值的文件
    loss_txt_path = (opt.log_path + 'Loss' + '_' + current_time + '.txt')
    os.makedirs(os.path.dirname(loss_txt_path), exist_ok=True)

    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, opt.test_path)
        scheduler.step()  # 在每个 epoch 结束后更新学习率

