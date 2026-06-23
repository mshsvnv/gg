import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
from metric import calculate_psnr, calculate_ssim, calculate_lpips, calculate_dists
from utils import AverageMeter, rgb_to_yuv420, yuv420_to_rgb
import logging


def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, model):
    # Убрана проверка local_rank, логгер создается сразу
    os.makedirs(args.log_path, exist_ok=True)
    # log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
    log_path = args.log_path
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    logger.info(args)
    
    dataset_train = Vimeo90K_Train_Dataset(augment=True)
    # Убран DistributedSampler, добавлен shuffle=True
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch
    
    dataset_val = Vimeo90K_Test_Dataset()
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_geo = AverageMeter()
    avg_dis = AverageMeter()
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        # Убран sampler.set_epoch(epoch) - он нужен только для DDP
        num_items = 0
        data_time_interval = 0.0
        train_time_interval = 0.0
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, imgt, img1, flow, embt = data
            img0 = yuv420_to_rgb(*rgb_to_yuv420(img0))
            imgt = yuv420_to_rgb(*rgb_to_yuv420(imgt))
            img1 = yuv420_to_rgb(*rgb_to_yuv420(img1))

            data_time_interval += time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad()

            # model вместо ddp_model
            _, loss_rec, loss_geo, loss_dis = model(img0, img1, embt, imgt, flow)

            loss = loss_rec + loss_geo + loss_dis
            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_geo.update(loss_geo.cpu().data)
            avg_dis.update(loss_dis.cpu().data)
            train_time_interval += time.time() - time_stamp
            num_items += args.batch_size

            # Убрано условие local_rank == 0
            if i % 100 == 0 and i != 0:
                logger.info(
                    'epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} s/K_items lr:{:.5e} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e}'.format(
                        epoch+1, args.epochs,
                        i * args.batch_size, args.iters_per_epoch * args.batch_size,
                        data_time_interval * 1000 / num_items, train_time_interval * 1000 / num_items,
                        lr, avg_rec.avg, avg_geo.avg, avg_dis.avg
                    )
                )
                avg_rec.reset()
                avg_geo.reset()
                avg_dis.reset()
                num_items = 0
                data_time_interval = 0.0
                train_time_interval = 0.0

            iters += 1
            time_stamp = time.time()

        if (epoch+1) % args.eval_interval == 0:
            psnr = evaluate(args, model, dataloader_val, epoch, logger)
            if psnr > best_psnr:
                best_psnr = psnr
                # Сохраняем state_dict напрямую из модели
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'best'))
            torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest'))


def evaluate(args, model, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_geo_list = []
    loss_dis_list = []
    psnr_list = []
    ssim_list = []
    lpips_list = []
    dists_list = []
    time_stamp = time.time()
    for _, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, flow, embt = data
        img0 = yuv420_to_rgb(*rgb_to_yuv420(img0))
        imgt = yuv420_to_rgb(*rgb_to_yuv420(imgt))
        img1 = yuv420_to_rgb(*rgb_to_yuv420(img1))

        with torch.no_grad():
            # model вместо ddp_model
            imgt_pred, loss_rec, loss_geo, loss_dis = model(img0, img1, embt, imgt, flow)

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_geo_list.append(loss_geo.cpu().numpy())
        loss_dis_list.append(loss_dis.cpu().numpy())

        for j in range(img0.shape[0]):
            psnr_val = calculate_psnr(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0))
            psnr_list.append(psnr_val)
            ssim_val = calculate_ssim(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0))
            ssim_list.append(ssim_val)
            lpips_val = calculate_lpips(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0))
            lpips_list.append(lpips_val)
            dists_val = calculate_dists(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0))
            dists_list.append(dists_val)

    eval_time_interval = time.time() - time_stamp
    
    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e} psnr:{:.3f} ssim:{:.3f} lpips:{:.4f} dists:{:.4f}'.format(
        epoch+1, args.epochs, eval_time_interval, 
        np.array(loss_rec_list).mean(), 
        np.array(loss_geo_list).mean(), 
        np.array(loss_dis_list).mean(), 
        np.array(psnr_list).mean(), 
        np.array(ssim_list).mean(),
        np.array(lpips_list).mean(),
        np.array(dists_list).mean()
    ))

    return np.array(psnr_list).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_RGB2')
    # Удалены аргументы --local_rank и --world_size
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    # Убрана инициализация процесса dist.init_process_group
    # Убрано torch.cuda.set_device
    args.device = torch.device(args.device)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.model_name == 'IFRNet':
        from models.IFRNet import Model
    elif args.model_name == 'IFRNet_L':
        from models.IFRNet_L import Model
    elif args.model_name == 'IFRNet_S':
        from models.IFRNet_S import Model
    elif args.model_name == 'IFRNet_RGB2':
        from models.IFRNET_RGB2 import Model
    elif args.model_name == 'IFRNet_RGB2P':
        from models.IFRNET_RGB2P import Model
    elif args.model_name == 'IFRNet_YUV420':
        from models.IFRNet_YUV420 import Model
    elif args.model_name == 'IFRNet_YUV420P':
        from models.IFRNet_YUV420P import Model
    elif args.model_name == 'IFRNet_YUV420F':
        from models.IFRNet_YUV420 import Model
    elif args.model_name == 'IFRNet_YUV420FP':
        from models.IFRNet_YUV420P import Model
    else:
        print(f"Unknown model: {args.model_name}")
        quit(-1)

    args.log_path = args.log_path + '/' + args.model_name
    args.num_workers = args.batch_size

    # Модель сразу переносится на устройство без DDP обертки
    model = Model().to(args.device)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
    
    train(args, model)