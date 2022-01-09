# -*-coding:utf-8-*-
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
# import onnx
# from onnx import shape_inference

# 提升计算速度(但计算中可能有随机性导致网络前馈结果略有差异)
cudnn.benchmark = True

#region parameter parser
parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')    # 读取相机参数时，将深度间隔乘以这个因子

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/d192/', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=100, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

# torch.cuda.set_per_process_memory_fraction(0.8, 0)
# torch.cuda.empty_cache()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
#endregion

#region create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)     # tensorboard可视化

print("argv:", sys.argv[1:])
print_args(args)
#endregion

# @mark dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=True)    # dtop_last: 丢掉多出来的一点不用
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)


# @mark model, optimizer
model = MVSNet(refine=False)
if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)  # 使用多GPU加速
model.cuda()    # 强制放到cuda上计算
model_loss = mvsnet_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

#region load parameters 从上次末尾继续训练
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))    # nelement()统计tensor的个数
#endregion


# @mark main function
def train():
    # 动态调整学习率(后期逐渐减小)
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]       # 几个需要下降学习率的epoch index
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            # 训练一个batch的数据
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)

                # @doubleZ
                img_ = torch.nn.functional.interpolate(image_outputs["ref_img"], scale_factor=0.25, mode='bilinear', align_corners=True)[0]
                stack = torch.stack([image_outputs["depth_est"], image_outputs["depth_gt"], image_outputs["errormap"], img_])
                logger.add_images('train/est-gt-errormap-img', stack, global_step)

            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time))

        # save checkpoint 默认一个epoch保存一次模型
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
            # torch.save(model, "mvsnet.pth")


        # testing
        avg_test_scalars = DictAverageMeter()   # 把loss这些信息记到dict中，方便输出
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)

            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    """训练DataLoader中取出的一次数据

    Args:
        sample ([imgs, proj_matrices, depth, depth_values, mask]): 1ref图+2src图，3个投影矩阵，深度图真值，深度假设列表，mask

    Returns:
        [loss, scalar_outputs, image_outputs]: 
            scalar_outputs: loss, abs_depth_error, thresXmm_error
            image_outputs: depth_est, depth_gt, ref_img, mask, errormap
    """
    model.train()           # 切换到train模式
    optimizer.zero_grad()   # 优化器梯度清零开始新一次的训练

    sample_cuda = tocuda(sample)    # 将所有Tensor类型的变量放到cuda计算
    depth_gt = sample_cuda["depth"] # 深度图ground truth数据
    mask = sample_cuda["mask"]      # mask用于将没有深度的地方筛除掉不计算loss

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])     # 将数据放到model中进行训练
    depth_est = outputs["depth"]    # MVSNet得到的深度估计图

    loss = model_loss(depth_est, depth_gt, mask)    # 计算estimation和ground truth的loss，mask用于选取有深度值的位置，只用这些位置的深度真值计算loss
    loss.backward()     # loss函数梯度反传
    optimizer.step()    # 优化器中所有参数沿梯度下降一步

    scalar_outputs = {"loss": loss}         # 这轮训练得到的loss
    image_outputs = {
        "depth_est": visualize_depth(depth_est[0] * mask[0]),      # 深度图估计(滤除掉本来就没有深度的位置)
        "depth_gt": visualize_depth(sample["depth"][0]),        # 深度图真值
        "ref_img": sample["imgs"][:, 0],    # 要估计深度的ref图
        "mask": sample["mask"]              # mask图(0-1二值图，为1代表这里有深度值)
    }

    # 导出onnx模型一直报错，大概是说pytorch的某些操作对于onnx导出不支持
    # torch.onnx.export(model, (sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"]), "mvsnet.onnx")
    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load("mvsnet.onnx")), model)

    if detailed_summary:
        image_outputs["errormap"] = visualize_depth((depth_est[0] - depth_gt[0]).abs() * mask[0])                             # 预测图和真值图的区别部分
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)  # 绝对深度估计误差(整个场景深度估计的偏差平均值) mean[abs(est - gt)]
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)        # 整个场景深度估计误差大于2mm的偏差偏差值(认为2mm之内都是估计的可以接受的) mean[abs(est - gt) > threshold]
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss = model_loss(depth_est, depth_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()
