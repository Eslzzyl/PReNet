import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_model(gt_path, model_path, model_name, nimgs=100, nrain=1):
    psnrs = np.zeros((nimgs, 1))
    ssims = np.zeros((nimgs, 1))

    for img_num in range(1, nimgs + 1):
        for _ in range(1, nrain + 1):
            gt_img_path = os.path.join(gt_path, f'norain-{img_num:03d}.png')
            gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE) / 255.0

            model_img_path = os.path.join(model_path, f'rain-{img_num:03d}.png')
            model_img = cv2.imread(model_img_path, cv2.IMREAD_GRAYSCALE) / 255.0

            psnrs[img_num - 1] += peak_signal_noise_ratio(gt_img, model_img)
            ssims[img_num - 1] += structural_similarity(gt_img * 255, model_img * 255, data_range=255)

    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    print(f'{model_name}: psnr={avg_psnr:.4f}, ssim={avg_ssim:.4f}')

def evaluate_joder():
    jorder_psnrs = []
    jorder_ssims = []

    for img_num in range(1, 101):
        gt_img_path = os.path.join(gt_path, f'norain-{img_num:03d}.png')
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE) / 255.0

        jorder_img_path = os.path.join(JORDER_path, f'Derained-Rain100L-rain-{img_num:03d}.png')
        jorder_img = cv2.imread(jorder_img_path, cv2.IMREAD_GRAYSCALE) / 255.0

        jorder_psnr = peak_signal_noise_ratio(gt_img, jorder_img)
        jorder_ssim = structural_similarity(gt_img * 255, jorder_img * 255, data_range=255)

        jorder_psnrs.append(jorder_psnr)
        jorder_ssims.append(jorder_ssim)

    avg_jorder_psnr = np.mean(jorder_psnrs)
    avg_jorder_ssim = np.mean(jorder_ssims)
    print(f'JORDER: psnr={avg_jorder_psnr:.4f}, ssim={avg_jorder_ssim:.4f}')
    
if __name__ == "__main__":
    gt_path = '../datasets/test/Rain100L/'
    JORDER_path = '../results/Rain100L/Rain100L_JORDER/'

    PReNet = '../results/Rain100L/PReNet/'
    PReNet_r = '../results/Rain100L/PReNet_r/'
    PRN = '../results/Rain100L/PRN6/'
    PRN_r = '../results/Rain100L/PRN_r/'

    # model_paths = [PReNet, PReNet_r, PRN, PRN_r]
    model_paths = [PReNet]

    # model_names = ['PReNet', 'PReNet_r', 'PRN', 'PRN_r']
    model_names = ['PReNet']

    for model_path, model_name in zip(model_paths, model_names):
        evaluate_model(gt_path, model_path, model_name)
