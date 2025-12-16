# 推理，输出推理结果的掩膜（前景白色，背景黑色）,推出的结果可用于EVAL.py计算不同阈值下的指标
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from utils.dataloader import test_dataset
import cv2
from tqdm import tqdm

from lib.pvt import CAFE


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
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default=r"./checkpoint/best.pth")

    opt = parser.parse_args()
    model = CAFE()
    model.load_state_dict(torch.load(opt.pth_path))

    # 显式指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f'powerpoint:{opt.pth_path}')
    for _data_name in ['test']:
        dice_bank = []
        iou_bank = []
        data_path = os.path.join('../jinshan-dataset-926-risk/test-926-risk', _data_name)
        save_path = os.path.join('../jinshan-dataset-926-risk/test-926-risk/test-result/MCD-Net', _data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)

        for i in tqdm(range(num1), desc=f"{_data_name}预测中"):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            save_gt = gt
            save_img = image
            image = image.to(device)

            # 前向传播
            cbam1_out, cbam2_out, cbam3_out, out4, res1, res2, res3 = model(image)

            # 融合结果
            # eval Dice
            res = F.interpolate(cbam1_out + cbam2_out + cbam3_out + out4 + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            #######################################

            save_image_name = name

            cv2.imwrite(save_path + '/' + save_image_name, res * 255)

            dice = mean_dice_np(gt, res)
            iou = mean_iou_np(gt, res)

            dice_bank.append(dice)
            iou_bank.append(iou)

        print('{}--Dice: {:.4f}, IoU: {:.4f}'.format(_data_name, np.mean(dice_bank), np.mean(iou_bank)))
