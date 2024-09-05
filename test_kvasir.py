import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from config import get_config
from lib.CODEFormerUnet import CODEFormer
from utils.dataloader import get_loader,test_dataset
from PIL import Image

pred_path = 'output/clinic-DB/pred/'
gt_path = 'output/clinic-DB/gt/'
img_path = 'output/clinic-DB/origin_imgs/'

def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_class == 1 else torch.long
    n_val = len(loader)
    pred_idx=0
    gt_idx=0
    img_idx = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            for img in pred:
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(pred_path+'/'+str(pred_idx)+'.jpg')
                pred_idx += 1
            for img in true_masks:
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(gt_path+'/'+str(gt_idx)+'.jpg')
                gt_idx += 1
            for img in imgs:
                mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
                original_img = imgs * std + mean    # 归一化是减均值除方差
                original_img = torch.clamp(original_img, 0, 1)

                # 将图像转换为 numpy 数组用于可视化
                original_img_np = original_img.cpu().numpy().squeeze().transpose(1, 2, 0)  # (height, width, channels)

                # 将 NumPy 数组转换为 PIL 图像
                img = Image.fromarray((original_img_np * 255).astype(np.uint8))
                img.save(img_path + '/' + str(img_idx) + '.jpg')
                img_idx += 1


            pbar.update()


def test_net(net,
              device,
              batch_size=1,
              n_class=1,
              img_size=512):


    val_img_dir = 'datasets/CVC-ClinicDB/val/images/'
    val_mask_dir = 'datasets/CVC-ClinicDB/val/masks/'

    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = False)
    net.eval()

    eval_net(net, val_loader, device)


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=3000, help='maximum epoch number to train')

    parser.add_argument('--batch_size', type=int,
                        default=3, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='segmentation network learning rate')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    return parser.parse_args()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    config = get_config(args)

    net = CODEFormer(out_chans=1).to(device)
    #net.load_from(config)
    net.to(device=device)

    net.load_state_dict(
        torch.load("./checkpoints/best_clinic-db_91.882.pth", map_location=device), True
    )

    try:
        test_net(net=net,
                  batch_size=1,
                  device=device,
                  img_size=224)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

