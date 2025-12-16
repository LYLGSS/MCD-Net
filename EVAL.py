# 计算不同阈值下的指标
import argparse
from utils.dataloader import test_dataset
from tabulate import tabulate
from PIL import Image
from tqdm import tqdm
################################################################
from my_utils.eval_function import *
import pandas as pd
################################################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,default=r'../jinshan-dataset-926-risk/test-926-risk')
    ######## 修改这个地方test.py代码推理的掩膜路径)
    parser.add_argument('--Tuili_path', type=str,default=r'../jinshan-dataset-926-risk/test-926-risk/test-result/MCD-Net')
    ########
    parser.add_argument('--output_file', type=str,default=r'../jinshan-dataset-926-risk/test-926-risk/eval-result/MCD-Net/eval-result.xlsx')

    parser.add_argument('--ROC', type=str,default=r'../jinshan-dataset-926-risk/test-926-risk/eval-result/MCD-Net/ROC',help='ROC')

    opt = parser.parse_args()
    print(opt.Tuili_path)
    Thresholds = np.linspace(1, 0, 256)
    headers = ['test']
    results = []


    for _data_name in ['test']:
        # 数据地址
        data_path = os.path.join(opt.data_path, _data_name)
        # 新添加的一个地址（作为推理的结果地址）
        Tuili_path = os.path.join(opt.Tuili_path, _data_name)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)

        num1 = len(os.listdir(gt_root)) # 查看数量
        test_loader = test_dataset(image_root, gt_root, 352)



        # 存储指标（n组指标，一组一个确定的指标）
        threshold_Fmeasure = np.zeros((num1, len(Thresholds)))
        threshold_Emeasure = np.zeros((num1, len(Thresholds)))
        threshold_IoU = np.zeros((num1, len(Thresholds)))
        threshold_Sensitivity = np.zeros((num1, len(Thresholds)))
        threshold_Specificity = np.zeros((num1, len(Thresholds)))
        threshold_Dice = np.zeros((num1, len(Thresholds)))

        Smeasure = np.zeros(num1)
        wFmeasure = np.zeros(num1)
        MAE = np.zeros(num1)

        for i in tqdm(range(num1), desc=f"Processing {_data_name}", ncols=100):

            image, gt, name = test_loader.load_data()

            pred_mask = np.array(Image.open(os.path.join(Tuili_path, name)))
            gt_mask = np.array(gt)

            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            assert pred_mask.shape == gt_mask.shape
            gt_mask = gt_mask.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)
            pred_mask = pred_mask.astype(np.float64) / 255


            # 4.计算指标
            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in tqdm(enumerate(Thresholds), total=len(Thresholds), desc="Thresholds", leave=False, ncols=100):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[
                    j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)

            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou


        result = []

        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)

        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)

        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)

        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)

        result.extend([meanDic, meanIoU, wFm, Sm, meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe])
        results.append([_data_name, *result])


        #######保存的地址#######
        curve_save_dir = os.path.join(opt.ROC, _data_name)
        os.makedirs(curve_save_dir, exist_ok=True)
        avg_dice_per_threshold = np.mean(threshold_Dice, axis=0)
        np.save(os.path.join(curve_save_dir, 'dice_{}.npy'.format(os.path.basename(opt.Tuili_path))),
                avg_dice_per_threshold)






    tab = tabulate(results, headers=['dataset', *headers], floatfmt=".3f")
    print(tab)

    # 将结果存储到 Excel 文件中
    df = pd.DataFrame(results, columns=['Dataset', 'Mean Dice', 'Mean IoU', 'wFmeasure', 'Smeasure', 'Mean Em', 'MAE',
                                        'Max Em', 'Max Dice', 'Max IoU', 'Mean Sensitivity', 'Max Sensitivity',
                                        'Mean Specificity', 'Max Specificity'])

    # 如果文件已存在，加载现有数据并追加新数据
    excel_path = opt.output_file
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)  # 将新数据追加到现有数据中
    else:
        combined_df = df

    # 保存到 Excel 文件
    combined_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")





