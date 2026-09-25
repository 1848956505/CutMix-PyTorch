# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import model.resnet as RN
import model.pyramidnet as PYRM
import model.preact_resnet as PARN
import model.wide_resnet as WRN
import model.dense_net as DN

import utils
import numpy as np

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix/Mixup PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
# 新增：Mixup配置
parser.add_argument('--mixup_alpha', default=1.0, type=float,
                    help='Mixup Beta distribution alpha')
parser.add_argument('--aug', default='baseline', type=str,
                    choices=['baseline', 'cutmix', 'mixup'],
                    help='augmentation method: baseline, cutmix, or mixup')
parser.add_argument(
        '--recipe',
        default='cutmix',
        type=str,
        choices=['cutmix', 'mixup'],
        help='training recipe: cutmix or mixup'
    )
parser.add_argument(
        '--seed',
        default=20170922,
        type=int,
        help='random seed'
    )

parser.add_argument(
    '--amp',
    action = "store_true",
    help = 'use automatic mixed precision'
)

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Experiment Configuration")
    print(f"Recipe        : {args.recipe}")
    print(f"Dataset       : {args.dataset}")
    print(f"Network       : {args.net_type}")
    print(f"Augmentation  : {args.aug}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Weight decay  : {args.weight_decay}")
    

    if args.aug == 'cutmix':
        print(f"CutMix beta   : {args.beta}")
        print(f"CutMix prob   : {args.cutmix_prob}")
    elif args.aug == 'mixup':
        print(f"Mixup alpha   : {args.mixup_alpha}")

    print("=" * 60)

    if args.dataset.startswith('cifar'):
            # Mixup实验CIFAR-10配置
        if args.recipe == 'mixup':
            normalize = transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
    
        # 保留CutMix/PyramidNet原有配置
        else:
            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('/root/autodl-tmp/CIFAR_100', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('/root/autodl-tmp/CIFAR_100', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('/root/autodl-tmp/CIFAR_10', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('/root/autodl-tmp/CIFAR_10', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('../..//home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    # 这里新增模型
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    elif args.net_type == 'preact_resnet18':
        model = PARN.preact_resnet18(
            num_classes=numberofclass
        )
    elif args.net_type == 'wide_resnet28_10':
        model = WRN.wide_resnet28_10(
            num_classes = numberofclass
        )
    elif args.net_type == 'densenet_bc190':
        model = DN.dense_net_bc190(
            num_classes = numberofclass
        )
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=(args.recipe == 'cutmix'))
    scaler = torch.amp.GradScaler(
    'cuda',
    enabled=args.amp
)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):
    
        adjust_learning_rate(optimizer, epoch)
    
        # train for one epoch
        train_loss, train_err1, train_err5 = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scaler
        )
    
        # evaluate on validation set
        val_loss, err1, err5 = validate(
            val_loader,
            model,
            criterion,
            epoch
        )
    
        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
    
        if is_best:
            best_err5 = err5
    
        current_lr = optimizer.param_groups[0]['lr']
    
        
        print(
            f"Epoch [{epoch + 1:3d}/{args.epochs}] | "
            f"LR {current_lr:.5f} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc@1 {100.0 - train_err1:.2f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc@1 {100.0 - err1:.2f}% | "
            f"Val Acc@5 {100.0 - err5:.2f}% | "
            f"Best Acc@1 {100.0 - best_err1:.2f}%"
        )
    
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'augmentation': args.aug,
            'args': vars(args),
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict()
        }, is_best)

    print(
        f"Best Val Acc@1: {100.0 - best_err1:.2f}% | "
        f"Val Acc@5: {100.0 - best_err5:.2f}%"
    )


def train(train_loader, model, criterion, optimizer, epoch, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # 使用tqdm进度条
    pbar = tqdm(
        train_loader,
        desc=f"Train {epoch + 1}/{args.epochs}",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5
    )

    for i, (input, target) in enumerate(pbar):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # ———————————————————— CutMix ————————————————————
        if args.aug == 'cutmix':
            r = np.random.rand()
        
            if args.beta > 0 and r < args.cutmix_prob:
                # 1. 从 Beta 分布中采样混合系数
                lam = np.random.beta(args.beta, args.beta)
        
                # 2. 打乱 batch
                rand_index = torch.randperm(
                    input.size(0),
                    device=input.device
                )
        
                target_a = target
                target_b = target[rand_index]
        
                # 3. 随机生成 CutMix 区域
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        
                # 4. 用另一张图像的区域进行替换
                input[:, :, bbx1:bbx2, bby1:bby2] = \
                    input[rand_index, :, bbx1:bbx2, bby1:bby2]
        
                # 5. 根据实际裁剪面积重新计算 lambda
                lam = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1)
                    / (input.size()[-1] * input.size()[-2])
                )

                with torch.autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=args.amp
                ):
                    # forward
                    output = model(input)
            
                    # CutMix 加权损失
                    loss = (
                        criterion(output, target_a) * lam
                        + criterion(output, target_b) * (1.0 - lam)
                    )
            
                    # CutMix 加权训练准确率
                    err1, err5 = mixed_accuracy(
                        output.data,
                        target_a,
                        target_b,
                        lam,
                        topk=(1, 5)
                    )
        
            else:
                # 本 batch 不执行 CutMix
                output = model(input)
                loss = criterion(output, target)
        
                err1, err5 = accuracy(
                    output.data,
                    target,
                    topk=(1, 5)
                )

        # ———————————————————— Mixup ————————————————————
        elif args.aug == 'mixup':
            # 随机采样混合系数
            lam = np.random.beta(
                args.mixup_alpha,
                args.mixup_alpha
            )
            # 打乱batch的顺序
            rand_index = torch.randperm(input.size(0), device=input.device)
            # 取顺序标签和乱序的标签
            target_a = target
            target_b = target[rand_index]
            # 构造融合样本
            mixed_input = lam * input + (1.0 - lam) * input[rand_index]

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=args.amp
            ):
                output = model(mixed_input)
                # 这里也是，通过加权计算损失，来间接达到标签融合的效果
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    
                err1, err5 = mixed_accuracy(
                    output.data,
                    target_a,
                    target_b,
                    lam,
                    topk=(1, 5)
                )
            
        # ———————————————————— Baseline ————————————————————
        else:
            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=args.amp
            ):
                
                output = model(input)
                loss = criterion(output, target)
    
                err1, err5 = accuracy(
                    output.data,
                    target,
                    topk=(1, 5)
                )

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(
            loss=f"{losses.avg:.4f}",
            acc1=f"{100.0 - top1.avg:.2f}%"
        )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg, top5.avg

# 根据混合系数，随机生成矩形框
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # 公式： lam = 1 - （box area / W * H）比例
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 这里是随机选择矩形框的中心点。
    cx = np.random.randint(W)   # 从[0-W)中随机取整数
    cy = np.random.randint(H)

    # 这里clip将bbx1限制在了0到W之间。主对角线上两个顶点的坐标
    bbx1 = np.clip(cx - cut_w // 2, 0, W)   # 左上角的 x 坐标
    bby1 = np.clip(cy - cut_h // 2, 0, H)   # 左上角的 y 坐标
    bbx2 = np.clip(cx + cut_w // 2, 0, W)   # 右下角的 x 坐标
    bby2 = np.clip(cy + cut_h // 2, 0, H)   # 右下角的 y 坐标

    return bbx1, bby1, bbx2, bby2

@torch.no_grad()
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pbar = tqdm(
        val_loader,
        desc=f"Val   {epoch + 1}/{args.epochs}",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5
    )
    
    for i, (input, target) in enumerate(pbar):
        input, target = input.cuda(), target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        pbar.set_postfix(
            loss=f"{losses.avg:.4f}",
            acc1=f"{100.0 - top1.avg:.2f}%"
        )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

# 新增：Mixup实验准确率计算
def mixed_accuracy(output, target_a, target_b, lam, topk=(1, 5)):
    err_a = accuracy(output, target_a, topk)
    err_b = accuracy(output, target_b, topk)

    return [
        lam * a + (1.0 - lam) * b
        for a, b in zip(err_a, err_b)
    ]
    

if __name__ == '__main__':
    main()
