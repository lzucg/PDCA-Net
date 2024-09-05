import argparse
import logging
import os
import sys
from config import get_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lib.SegformerUnet import SegFormerUnet
from lib.CODEFormerUnet import CODEFormer
from utils.eval import eval_net
from utils.dataloader import get_loader,test_dataset
import warnings

warnings.filterwarnings("ignore")

torch.cuda.set_device(0)
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

train_img_dir = 'datasets/Kvasir_SEG_Training_880/train/images/'
train_mask_dir = 'datasets/Kvasir_SEG_Training_880/train/masks/'
val_img_dir = 'datasets/Kvasir_SEG_Training_880/val/images/'
val_mask_dir = 'datasets/Kvasir_SEG_Training_880/val/masks/'
dir_checkpoint = './checkpoints/'


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def cal(loader):
    tot = 0
    for batch in loader:
        imgs, _ = batch
        tot += imgs.shape[0]
    return tot

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def train_net(net,
              device,
              epochs=500,
              batch_size=8,
              lr=0.001,
              save_cp=True,
              n_class=1,
              img_size=512):

    train_loader = get_loader(train_img_dir, train_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = True)
    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=1, trainsize=img_size, augmentation = True)

    n_train = cal(train_loader)
    n_val = cal(val_loader)
    logger = get_logger('kvasir.log')


    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Vailding size:   {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:  {img_size}
    ''')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//5, lr/10)
    if n_class > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_dice = 0

    #net.load_state_dict(torch.load("checkpoints/clon_dice_89.721.pth"))
    #size_rates = [256, 384, 512, 640]
    size_rates = [224]
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        b_cp = False
        Batch = len(train_loader)
        with tqdm(total=n_train*len(size_rates), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
               # for rate in size_rates:
                imgs, true_masks = batch
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if n_class == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = structure_loss(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

        scheduler.step()
        val_dice, epoch_acc = eval_net(net, val_loader, device)
        if val_dice > best_dice:
            best_dice = val_dice
            b_cp = True
        epoch_loss = epoch_loss / Batch

        logger.info('epoch: {} train_loss: {:.3f} epoch_dice: {:.3f}'.format(epoch + 1, epoch_loss, val_dice* 100))
        logger.info('epoch: {} train_loss: {:.3f} best_dice: {:.3f}, epoch_acc: {:.3f}'.format(epoch + 1, epoch_loss, best_dice* 100, epoch_acc * 100))
        if save_cp and b_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + 'epoch:{}_dice:{:.3f}.pth'.format(epoch + 1, best_dice*100))
            logging.info(f'Checkpoint {epoch + 1} saved !')



def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Kvasir', help='experiment_name')
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
                        default=8, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='segmentation network learning rate')
    parser.add_argument('-l', '--learning -rate', metavar='LR', type=float, nargs='?', default=0.001,
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
    parser.add_argument('--resume', default="", help='resume from checkpoint')
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
    if args.dataset == "Kvasir":
        args.root_path = os.path.join(args.root_path, "train_npz")
    config = get_config(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #net = SegFormerUnet()
    net = CODEFormer(out_chans=1)
    #net.load_from(config)
    #net = nn.DataParallel(net, device_ids=[2])
    net = net.to(device)

    try:
        train_net(net=net,
                  epochs=args.max_epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  img_size=args.img_size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
