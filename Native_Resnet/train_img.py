'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
import time
import sys
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from dataset import FacesDataset

from resnet import *
import math
from visdom import Visdom
import numpy as np

from .flops_benchmark import add_flops_counting_methods


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lrdecay', default=10, type=int,
                    help='epochs to decay lr')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lrfact', default=1, type=float,
                    help='learning rate factor')
parser.add_argument('--lossfact', default=1, type=float,
                    help='loss factor')
parser.add_argument('--target', default=0.4, type=float, help='target rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='folder path to save checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--print-freq', '-p', default=25, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--expname', default='give_me_a_name', type=str, metavar='n',
                    help='name of experiment (default: test')
parser.set_defaults(test=False)
parser.set_defaults(visdom=False)

best_prec1 = 0

def main():
    print(time.ctime())
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)

    if args.visdom:
        global plotter
        plotter = VisdomLinePlotter(env_name=args.expname)
        
    # set the target rates for each layer
    # the default is to use the same target rate for each layer
    target_rates_list = [args.target] * 33
    #for i in range(7,30):
    #    target_rates_list[i]=0.4
    target_rates = {i:target_rates_list[i] for i in range(len(target_rates_list))}

    model = ResNet101_ImageNet()

    model = add_flops_counting_methods(model)

    # optionally initialize from pretrained
    if args.pretrained:
        latest_checkpoint = args.pretrained
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(latest_checkpoint))
            # TODO: clean this part up
            checkpoint = torch.load(latest_checkpoint)
            state = model.state_dict()
            loaded_state_dict = checkpoint
            for k in loaded_state_dict:
                if k in state:
                    state[k] = loaded_state_dict[k]
                else:
                    if 'fc' in k:
                        state[k.replace('fc', 'linear')] = loaded_state_dict[k]
                    if 'downsample' in k:
                        state[k.replace('downsample', 'shortcut')] = loaded_state_dict[k]
            model.load_state_dict(state) 
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(latest_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(latest_checkpoint))
    
    model = torch.nn.DataParallel(model).cuda()

    # ImageNet Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = FacesDataset("../images/train","../images/train_labels.csv", 
        transforms.Compose([
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15, resample=False, expand=False, center=None),
                #transforms.Scale(224),
                transforms.ToTensor(),
                #normalize,
            ]))

    val_loader = FacesDataset("../images/val","../images/val_labels.csv", transforms.Compose([transforms.ToTensor()]))
    
    train_loader =torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # optionally resume from a checkpoint
    if args.resume:
        latest_checkpoint = os.path.join(args.resume, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if 'fc' in name],
                            'lr': args.lrfact * args.lr, 'weight_decay': args.weight_decay},
                            {'params': [param for name, param in model.named_parameters() if 'fc' not in name],
                            'lr': args.lr, 'weight_decay': args.weight_decay}
                            ], momentum=args.momentum)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.test:
        test_acc = validate(val_loader, model, criterion, 60, target_rates)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, target_rates)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, target_rates)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch, target_rates):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    activations = AverageMeter()

    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to train mode
    model.train()
    model.start_flops_count()

    end = time.time()

    ttt = torch.FloatTensor(33).fill_(0)
    ttt = ttt.cuda()
    ttt = torch.autograd.Variable(ttt, requires_grad=False)

    for i, (input, target) in enumerate(train_loader):
        # input = input.view((input.size(0), input.size(2), input.size(3), input.size(1)))
        print(model.compute_average_flops_cost()/ 1e9 / 2)
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # classification loss
        loss_classify = criterion(output, target_var)

        loss = loss_classify

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}))\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    # log values to visdom
    if args.visdom:
        plotter.plot('act', 'train', epoch, activations.avg)
        plotter.plot('top1', 'train', epoch, top1.avg)
        plotter.plot('top5', 'train', epoch, top5.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)


def validate(val_loader, model, criterion, epoch, target_rates):
    """Perform validation on the validation set"""
    print(time.ctime())
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            #print(target.data.cpu().numpy()) #.data[0].cpu().numpy()) #for i in range(args.batch_size))
            
            # classification loss
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i < 0:    #% args.print_freq==0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    if args.visdom:
        plotter.plot('act', 'test', epoch, activations.avg)
        plotter.plot('top1', 'test', epoch, top1.avg)
        plotter.plot('top5', 'test', epoch, top5.avg)
        plotter.plot('loss', 'test', epoch, losses.avg)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(server='172.220.4.32',port='6006')
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name)
    def plot_heatmap(self, map, epoch):
        self.viz.heatmap(X=map,
                         env=self.env,
                         opts=dict(title='activations {}'.format(epoch),
                                   xlabel='modules',
                                   ylabel='classes'
                                   ))

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
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lrdecay))
    factor = args.lrfact
    if args.visdom:
        plotter.plot('learning_rate', 'train', epoch, lr)
    optimizer.param_groups[0]['lr'] = factor * lr
    optimizer.param_groups[1]['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
