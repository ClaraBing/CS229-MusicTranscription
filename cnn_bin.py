'''
Future work: treat as a detection problem since there are too many silence notes
(Currently training the two-way has note / no note classifier.)
'''

# utils
from __future__ import print_function
import argparse
from time import time
import os # for mkdir
import sys # for flushing stdout
# torch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from collections import Counter
import math
# our code
from PitchEstimationDataSet import *
from model import *
from model_bin import *
from util_cnn import *
from config import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mode', type=str,
                    help='Running mode: train / test / features')
# NOTE: please specify pretrained model in config.py
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-interval', type=int, default=25000, metavar='LR',
                    help='decrease lr if avg err of a lr-interval plateaus (default: 5000)')
parser.add_argument('--update_momentum', type=bool, default=True, metavar='M',
                    help='whether to update SGD momentum')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=5000, metavar='N',
                    help='how many batches to wait before saving the trained model')
parser.add_argument('--config', type = str, default='', help='which config to use, see config.py')

# NOTE: save_dir and save_prefix are moved to config.py

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# Configuration

# NOTE: use argument --config to specify the configuration to be used
cfg = config_mel_bin()
# print ("Using configuration %s" % args.config)
# Fields in cfg:
#   annot_folder: where the annotations are
#   image_folder: where the spectrograms are
#   sr_ratio
#   audio_type: 'RAW' or 'MIX'
#   multiple: whether to use multiple pitches at a time; if True, then the input/target will be vectors.
#   save_dir: where the trained models are saved
#   save_prefix: prefix of the file name of the saved model ('.pt' files)
#   use_pretrained: whether or not to use a pretrained model
#   pretrained_path: path to the pretrained model

if True:
    # train
    training_set = PitchEstimationDataSet(cfg['annot_folder']+'train/', cfg['image_folder']+'train/', sr_ratio=cfg['sr_ratio'], audio_type=cfg['audio_type'], multiple=cfg['multiple'], fusion_mode=cfg['fusion_mode'])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

if True:
    # val
    val_set = PitchEstimationDataSet(cfg['annot_folder']+'val/', cfg['image_folder']+'val/', sr_ratio=cfg['sr_ratio'], audio_type=cfg['audio_type'], multiple=cfg['multiple'], fusion_mode=cfg['fusion_mode'])
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)

if False:
    # test
    test_set = PitchEstimationDataSet(cfg['annot_folder']+'test/', cfg['image_folder']+'test/', sr_ratio=cfg['sr_ratio'], audio_type=cfg['audio_type'], multiple=cfg['multiple'], fusion_mode=cfg['fusion_mode'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)


def train(model, model_bin, train_loader, criterion, epoch):
    model.train()
    model_bin.eval() # the binary classifier is trained priot to this

    batch_start = time()
    avg_loss, prev_avg_loss = 0, 1000
    for batch_idx, dictionary in enumerate(train_loader):
        data = dictionary['image']
        target = dictionary['frequency']
        data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.LongTensor) # NOTE: may need to change target back to LongTensor for single notes
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        bin_output = model_bin(data)
        # select potentially positive samples
        proposal_mask = bin_output[:, 0] < bin_output[:, 1]
        if not proposal_mask.data.any():
            # no positive sampled in this batch
            continue
        proposal_data = Variable(torch.index_select(data.data, 0, proposal_mask.data.nonzero()[:, 0])).type(torch.FloatTensor).cuda()
        proposal_target = Variable(target.data[proposal_mask.data]).type(torch.LongTensor).cuda()
        output = model(proposal_data)
        loss = criterion(output, proposal_target) # F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # training log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: epoch {} iter {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime per batch: {:.6f}s'.format(
                epoch, batch_idx, batch_idx * len(proposal_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],
                (time()-batch_start)/args.log_interval))
            batch_start = time()
            sys.stdout.flush()
        # update lr
        if cfg['update_lr_in_epoch'] and batch_idx % args.lr_interval == 0:
            args.lr /= 10
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
                param_group['lr'] = args.lr
            print('Update lr to ' + str(args.lr))
        # save trained model
        if batch_idx % args.save_interval == 0:
            save_name = cfg['save_prefix'] + str(batch_idx) + '.pt'
            print('Saving model: ' + save_name)
            torch.save(model, cfg['save_dir']+save_name)

def validate(data_loader, model, model_bin, criterion, outfile=None, breakEarly=False):
    out_mtrx = np.empty((len(data_loader), 109, 2))

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    tp, tn = AverageMeter(), AverageMeter()

    model.eval()
    model_bin.eval()

    for batch_idx, dictionary in enumerate(data_loader):
        batch_start = time()

        data, bin_target, target = Variable(dictionary['image'], volatile=True).type(torch.FloatTensor),Variable(dictionary['hasNote']).type(torch.LongTensor), Variable(dictionary['frequency']).type(torch.LongTensor)
        if args.cuda:
            data, bin_target, target = data.cuda(), bin_target.cuda(), target.cuda()

        # compute output
        bin_output = model_bin(data)
        output_neg_mask = bin_output[:, 0]*0.7 > bin_output[:, 1]
        output = model(data)
        output[:, 0] = output_neg_mask.type(torch.IntTensor)
        # performance measure: loss & top 1/5 accuracy
        loss = criterion(output, target)
        losses.update(loss.data[0], data.size(0))
        prec1, prec5 = accuracy(output.data, target.data, topk=(1,5))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))
        # Save probabilities & corresponding pitch bins
        # probs, pitch_bins = torch.sort(output.data, 1, True) # params: data, axis, descending
        # out_mtrx[batch_idx, :, 0] = np.exp(probs.view(-1).cpu().numpy())
        # out_mtrx[batch_idx, :, 1] = pitch_bins.view(-1).cpu().numpy()
        true_pos, true_neg = recall(bin_output.data, bin_target.data)
        if bin_target.data[0] > 0:
          tp.update(true_pos[0], data.size(0))
        else:
          tn.update(true_neg[0], data.size(0))
                
        batch_time.update(time() - batch_start)
        
        if batch_idx % (200*args.log_interval) == 0:
            print('Val({:d}): '
                  'Loss: {:f} (avg: {:f})\t'
                  'True pos: {:f} (avg: {:f})\t'
                  'True neg: {:f} (avg: {:f})\t'
                  'Prec@1: {:f} (avg: {:f})\t'
                  'Prec@5: {:f} (avg: {:f})\t'
                  'Time: {:f}'.format(
                  batch_idx, losses.val, losses.avg, tp.val, tp.avg, tn.val, tn.avg, top1.val, top1.avg, top5.val, top5.avg, batch_time.avg))
            sys.stdout.flush()
        if breakEarly:
            break

    # overall average
    print('\n================\n'
          'Loss: {:f}\nPrec@1: {:f}\nPrec@5: {:f}'
          '\n================\n\n'.format(
          losses.avg, top1.avg, top5.avg))

    if outfile:
        np.save(outfile, out_mtrx)

    return top1.avg

# Bingary classifier: has note or not
def train_bin(model, train_loader, criterion, epoch):
    model.train()
    batch_start = time()
    avg_loss, prev_avg_loss = 0, 1000
    for batch_idx, dictionary in enumerate(train_loader):
        data = dictionary['image']
        target = dictionary['hasNote']
        data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # training log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: epoch {} iter {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime per batch: {:.6f}s'.format(
                epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],
                (time()-batch_start)/args.log_interval))
            batch_start = time()
            sys.stdout.flush()
        # update lr
        if cfg['update_lr_in_epoch'] and batch_idx % args.lr_interval == 0:
            args.lr /= 10
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
                param_group['lr'] = args.lr
            print('Update lr to ' + str(args.lr))
        # save trained model
        if batch_idx % args.save_interval == 0:
            save_name = cfg['save_prefix'] + str(batch_idx) + '.pt'
            print('Saving model: ' + save_name)
            torch.save(model, cfg['save_dir']+save_name)

def validate_bin(data_loader, model, criterion, outfile=None, breakEarly=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # true pos, true neg
    tp, tn = AverageMeter(), AverageMeter()
    prec1 = AverageMeter()

    model.eval()

    for batch_idx, dictionary in enumerate(data_loader):
        batch_start = time()

        data, target = Variable(dictionary['image'], volatile=True).type(torch.FloatTensor), Variable(dictionary['hasNote']).type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # compute output
        output = model(data)
        # performance measure: loss & true positive & true negative
        loss = criterion(output, target)
        losses.update(loss.data[0], data.size(0))
        prec1.update(accuracy(output.data, target.data, topk=(1,))[0][0], data.size(0))
        true_pos, true_neg = recall(output.data, target.data)
        if target.data[0] > 0:
          tp.update(true_pos[0], data.size(0))
        else:
          tn.update(true_neg[0], data.size(0))
        # Save probabilities & corresponding pitch bins
        probs, pitch_bins = torch.sort(output.data, 1, True) # params: data, axis, descending

        batch_time.update(time() - batch_start)
        
        if batch_idx % (200*args.log_interval) == 0:
            print('Val({:d}): '
                  'Loss: {:f} (avg: {:f})\t'
                  'Prec@1: {:f} (avg: {:f})\t'
                  'True Pos: {:f} (avg: {:f})\t'
                  'True Neg: {:f} (avg: {:f})\t'
                  'Time: {:f}'.format(
                  batch_idx, losses.val, losses.avg, prec1.val, prec1.avg, tp.val, tp.avg, tn.val, tn.avg, batch_time.avg))
            sys.stdout.flush()
        if breakEarly:
            return data, target

    # overall average
    print('\n================\n'
          'Loss: {:f}\nPrec@1: {:f}\nTrue Pos: {:f}\nTrue Neg: {:f}'
          '\n================\n\n'.format(
          losses.avg, prec1.avg, tp.avg, tn.avg))

    if outfile and not isBin:
        np.save(outfile, out_mtrx)

    return tp.avg


if __name__ == '__main__':
    if cfg['multiple']:
        raise ValueError('Please use cnn_multi.py for multiple output.')
    elif cfg['net'] == 'conv5_bin':
        model = Net(fusion_mode=cfg['fusion_mode'])
        model_bin = Net_Bin(fusion_mode=cfg['fusion_mode'])
    else:
        raise ValueError('Unrecognized network structure: ', cfg['net'])
    if args.cuda:
        model.cuda()
        model_bin.cuda()
    print_config(args, cfg)
    model.print_archi()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    lr_update_interval = [8, 16, 24, 32]

    # input / target are floats
    criterion = F.nll_loss
    best_tp = 0
    if args.mode == 'train':
        if cfg['use_pretrained']:
            pretrained_dict = torch.load(cfg['pretrained_path'])['state_dict']
            model.load_state_dict(pretrained_dict)
            model.cuda()
        pretrained_bin_dict = torch.load(cfg['pretrained_bin_path'])['state_dict']
        model_bin.load_state_dict(pretrained_bin_dict)
        model_bin.cuda()
        if not os.path.exists(cfg['save_dir']):
          os.mkdir(cfg['save_dir'])

        # test validate before training in case it falls apart
        curr_tp = validate(val_loader, model, model_bin, criterion)
        for epoch in range(args.epochs + 1):
            if epoch > 0:
                # use epoch=0 to check saving checkpoint
                print('\n\n###############\n'
                  '    Epoch {:d}'
                  '\n###############'.format(epoch))
        
                train(model, model_bin, train_loader, criterion, epoch)
        
                # validation
                curr_tp = validate(val_loader, model, model_bin, criterion)
            is_best = False if math.isnan(curr_tp) else curr_tp>best_tp
            best_tp = max(curr_tp, best_tp)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_tp': best_tp,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=cfg['save_dir']+cfg['save_prefix']+'_epoch{:d}.pt'.format(epoch))
    
            # update lr
            if epoch in lr_update_interval:
                args.lr /= 10
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
                    param_group['lr'] = args.lr
                print('Update lr to ' + str(args.lr))
            # update momentum
            if args.update_momentum and args.momentum < 0.9:
                args.momentum += 0.1
                for param_group in optimizer.param_groups:
                    print(param_group['momentum'])
                    param_group['momentum'] = args.momentum
                print('Update momentum to ' + str(args.momentum))
    else:
        # testing
        pretrained_dict = torch.load(cfg['pretrained_path'])['state_dict']
        model.load_state_dict(pretrained_dict)
        model.cuda()
        pretrained_bin_dict = torch.load(cfg['pretrained_bin_path'])['state_dict']
        model_bin.load_state_dict(pretrained_bin_dict)
        model_bin.cuda()
        if args.mode == 'features':
          # Get features
          visualise_features(train_loader, model, criterion, outfile_features='dataset/' + cfg['save_prefix'] + '_features_mtrx.npy', 
              outfile_annotations='dataset/'+cfg['save_prefix']+'_features_annotations.npy')
        else:
          validate(val_loader, model, model_bin,criterion)
