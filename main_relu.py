from __future__ import print_function, division
import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching')
parser.add_argument('--model', default='gwcnet-gc-relu', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default="sceneflow", help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/scenflow_finalpass/",
                    help='data path')
parser.add_argument('--trainlist', default="./filenames/sceneflow_train.txt", help='training list')
parser.add_argument('--testlist', default="./filenames/sceneflow_test.txt", help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
# 这个要改
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="10,12,14,16:2", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default="./checkpoints", help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt',
                    help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
print('# pwcnet parameters:', sum(param.numel() for param in model.parameters()))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def find_bottom_4(lst):
    # 使用enumerate函数得到每个元素的下标
    lst_with_index = list(enumerate(lst))

    # 使用lambda函数作为sorted函数的key参数，以元素的值作为排序依据
    sorted_lst = sorted(lst_with_index, key=lambda x: x[1], reverse=False)

    # 返回前6个元素的值和下标
    bottom_4_values = [x[1] for x in sorted_lst[:20]]
    bottom_4_indices = [x[0] for x in sorted_lst[:20]]

    return bottom_4_values, bottom_4_indices


def train():
    # 记录开始训练的时间
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time_1 = start_time
    loss_list = []
    D1_list = []
    EPE_list = []
    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()
        if epoch_idx > -1:
            # testing
            avg_test_scalars = AverageMeterDict()
            # bestepoch = 0
            # error = 100
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    # save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                         batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                         time.time() - start_time))
            avg_test_scalars = avg_test_scalars.mean()
            nowerror = avg_test_scalars["D1"][0]
            if nowerror < error:
                bestepoch = epoch_idx
                error = avg_test_scalars["D1"][0]

            loss_list.append(avg_test_scalars["loss"])
            D1_list.append(avg_test_scalars["D1"][0])
            EPE_list.append(avg_test_scalars["EPE"][0])

            # 单次epoch的test的loss, D1和EPE写到文件里
            train_log_yk = open("train_log_scene.txt", mode="a", encoding="utf-8")
            this_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("时间为:", this_time, file=train_log_yk)
            print("第" + str(epoch_idx) + "个epoch的loss的值为：", avg_test_scalars["loss"], end="\t", file=train_log_yk)
            print("第" + str(epoch_idx) + "个epoch的D1的值为：", avg_test_scalars["D1"][0], end="\t", file=train_log_yk)
            print("第" + str(epoch_idx) + "个epoch的EPE的值为：", avg_test_scalars["EPE"][0], file=train_log_yk)
            train_log_yk.close()  # 关闭文件

            save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            print("avg_test_scalars", avg_test_scalars)
            print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
            gc.collect()
    bottom_loss_values, bottom_loss_indices = find_bottom_4(loss_list)  # 看上去是4,其实是6个
    bottom_D1_values, bottom_D1_indices = find_bottom_4(D1_list)
    bottom_EPE_values, bottom_EPE_indices = find_bottom_4(EPE_list)
    train_log_yk = open("train_log_scene.txt", mode="a", encoding="utf-8")
    print("loss前4个最小元素的值为：", bottom_loss_values, end="\t", file=train_log_yk)
    print("loss最小的下标为：", bottom_loss_indices, file=train_log_yk)
    print("D1前4个最小元素的值为：", bottom_D1_values, end="\t", file=train_log_yk)
    print("D1最小的下标为：", bottom_D1_indices, file=train_log_yk)
    print("EPE前4个最小元素的值为：", bottom_EPE_values, end="\t", file=train_log_yk)
    print("EPE最小的下标为：", bottom_EPE_indices, file=train_log_yk)
    # 记录结束训练的时间
    final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("开始训练的时间为:", start_time_1, file=train_log_yk)
    print("训练完成的时间为:", final_time, file=train_log_yk)

    print('MAX epoch %d total test error = %.5f' % (bestepoch, error), file=train_log_yk)
    # 切记关闭文件
    train_log_yk.close()  # 关闭文件


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func().forward(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    # disp_ests, pred3,combine, pred0 = model(imgL, imgR)
    disp_ests, pred3 = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1_pred3"] = [D1_metric(pred, disp_gt, mask) for pred in pred3]
    # scalar_outputs["D1_combine"] = [D1_metric(pred, disp_gt, mask) for pred in combine]
    # scalar_outputs["D1_pred0"] = [D1_metric(pred, disp_gt, mask) for pred in pred0]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func().forward(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
