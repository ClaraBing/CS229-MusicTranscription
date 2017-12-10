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
# our code
from PitchEstimationDataSet import *
from model import *
from model_conv7 import *
from train_util import *
from config import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mode', type=str,
                    help='Running mode: train / test')
# NOTE: please specify pretrained model in config.py
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-interval', type=int, default=5000, metavar='LR',
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
# NOTE: save_dir and save_prefix are moved to config.py

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# Configuration
cfg = config_1() # NOTE: change this if you want to use different configuration
# Fields in cfg:
#   annot_folder: where the annotations are
#   image_folder: where the spectrograms are
#   audio_type: 'RAW' or 'MIX'
#   multiple: whether to use multiple pitches at a time; if True, then the input/target will be vectors.
#   save_dir: where the trained models are saved
#   save_prefix: prefix of the file name of the saved model ('.pt' files)
#   use_pretrained: whether or not to use a pretrained model
#   pretrained_path: path to the pretrained model

# train
training_set = PitchEstimationDataSet(cfg['annot_folder']+'train/', cfg['image_folder']+'train/', sr_ratio=6, audio_type=cfg['audio_type'], multiple=cfg['multiple'])
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

# val
val_set = PitchEstimationDataSet(cfg['annot_folder']+'val/', cfg['image_folder']+'val/', sr_ratio=6, audio_type=cfg['audio_type'], multiple=cfg['multiple'])
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
    # batch_size = args.batch_size, shuffle=False, **kwargs)

# test
# test_set = PitchEstimationDataSet(annot_folder+'test/', image_folder+'test/', transform=transforms.Compose([
#                transforms.ToTensor(),
#                transforms.Normalize((0.1307,), (0.3081,))
#                ]))
#test_set = PitchEstimationDataSet(annotations_test, annotations_folder + 'test', sr_ratio=2)
#test_loader = DataLoader(test_set, shuffle=False, # do not shuffle: the original ordering is needed for matching w/ annotations (for HMM)
#    batch_size = args.test_batch_size, **kwargs) # batch = 1

# raise ValueError('should stop here')


def train(model, train_loader, criterion, epoch):
    model.train()
    batch_start = time()
    avg_loss, prev_avg_loss = 0, 1000
    for batch_idx, dictionary in enumerate(train_loader):
        data = dictionary['image']
        target = dictionary['frequency']
        data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.FloatTensor) # NOTE: may need to change target back to LongTensor for single notes
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # print('data & target:')
        # print(data.data.shape)
        # print(target.data.shape)
        optimizer.zero_grad()
        output = model(data)
        # print('output:')
        # print(output.data.shape)
        # print(output.data.max())
        # print(output.data.min())
        loss = criterion(output, target) # F.nll_loss(output, target)
        # avg_loss += loss
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
        # save trained model
        if batch_idx % args.save_interval == 0:
            save_name = cfg['save_prefix'] + str(batch_idx) + '.pt'
            print('Saving model: ' + save_name)
            # torch.save(model.state_dict(), args.save_dir+save_name)
            # TODO: save also the optimizer (right now the lr can be found in the log)
            torch.save(model, cfg['save_dir']+save_name)


def validate(data_loader, model, criterion, outfile=None):
    out_mtrx = np.empty((len(data_loader), 109, 2))

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()

    model.eval()

    for batch_idx, dictionary in enumerate(data_loader):
        batch_start = time()

        data, target = Variable(dictionary['image'], volatile=True).type(torch.FloatTensor), Variable(dictionary['frequency']).type(torch.FloatTensor) # NOTE: may need to change back to LongTensor for single note
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # compute output
        output = model(data)
        # performance measure: loss & top 1/5 accuracy
        loss = criterion(output, target)
        losses.update(loss.data[0], data.size(0))
        prec1, prec5 = accuracy(output.data, target.data, topk=(1,5))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))
        # Save probabilities & corresponding pitch bins
        probs, pitch_bins = torch.sort(output.data, 1, True) # params: data, axis, descending
        out_mtrx[batch_idx, :, 0] = np.exp(probs.view(-1).cpu().numpy())
        out_mtrx[batch_idx, :, 1] = pitch_bins.view(-1).cpu().numpy()
            # prob_list, pitch_bin_list = list(probs.view(-1)), list(pitch_bins.view(-1)) 
            # for prob, pitch_bin in zip(prob_list, pitch_bin_list):
                


        batch_time.update(time() - batch_start)
        
        if batch_idx % (10*args.log_interval) == 0:
            print('Val({:d}): '
                  'Loss: {:f} (avg: {:f})\t'
                  'Prec@1: {:f} (avg: {:f})\t'
                  'Prec@5: {:f} (avg: {:f})\t'
                  'Time: {:f}'.format(
                  batch_idx, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, batch_time.avg))
            sys.stdout.flush()

    # overall average
    print('\n================\n'
          'Loss: {:f}\nPrec@1: {:f}\nPrec@5: {:f}'
          '\n================\n\n'.format(
          losses.avg, top1.avg, top5.avg))

    if outfile:
        np.save(outfile, out_mtrx)

    return top1.avg



def visualise_features(data_loader, model, criterion, outfile_features=None, outfile_annotations=None):
    L = 50
    out_mtrx = []
    annotations = []
    sampled_pitches = {}
    model.eval()
    for batch_idx, dictionary in enumerate(data_loader):
        if len(sampled_pitches) > L * 20:
          break
        batch_start = time()
        data, target = Variable(dictionary['image'], volatile=True).type(torch.FloatTensor), Variable(dictionary['frequency']).type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if target.data[0] not in sampled_pitches:
          sampled_pitches[target.data[0]] = 1
          # compute features
          output = model.get_features(data)
          out_mtrx.append(output.data.cpu().numpy()[0])
          annotations.append(target.data[0])
        elif sampled_pitches[target.data[0]] < L:
          sampled_pitches[target.data[0]] += 1
          # compute features
          output = model.get_features(data)
          out_mtrx.append(output.data.cpu().numpy()[0])
          annotations.append(target.data[0])
        # print (target.data[0], sampled_pitches[target.data[0]])
        if outfile_features and outfile_annotations:
          np.save(outfile_features, np.array(out_mtrx))
          np.save(outfile_annotations, np.array(annotations))


# TODO: check if validate() & test() calculate the same error;
# If yes, merge the two functions
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    if cfg['net'] == 'conv5':
        model = Net()
    elif cfg['net'] == 'conv7':
        model = Net_Conv7()
    else:
        raise ValueError('Unrecognized network structure: ', cfg['net'])
    if args.cuda:
        model.cuda()
    print_config(args, cfg)
    model.print_archi()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    lr_update_interval = [5, 10, 40, 50]

    if cfg['multiple']:
        # input / target are vectors
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    else:
        # input / target are floats
        criterion = F.nll_loss
    best_prec = 0
    if args.mode == 'train':
        if cfg['use_pretrained']:
            pretrained_dict = torch.load(cfg['pretrained_path'])['state_dict']
            model.load_state_dict(pretrained_dict)
            model.cuda()
        if not os.path.exists(cfg['save_dir']):
          os.mkdir(cfg['save_dir'])
        for epoch in range(1, args.epochs + 1):
            print('\n\n###############\n'
              '    Epoch {:d}'
              '\n###############'.format(epoch))
    
            train(model, train_loader, criterion, epoch)
    
            # validation
            prec = validate(val_loader, model, criterion)
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=cfg['save_dir']+cfg['save_prefix']+'_epoch{:d}.pt'.format(epoch))
    
            # update lr
            if epoch<100:
                # print('checking for avg loss')
                # avg_loss = avg_loss / args.lr_interval
                # if prev_avg_loss - avg_loss < 0.05:
                if epoch in lr_update_interval:
                    args.lr /= 10
                    for param_group in optimizer.param_groups:
                        print(param_group['lr'])
                        param_group['lr'] = args.lr
                    print('Update lr to ' + str(args.lr))
                # prev_avg_loss, avg_loss = avg_loss, 0
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
        # Get features
        visualise_features(train_loader, model, criterion, outfile_features='train_features_mtrx.npy', outfile_annotations='train_features_annotations.npy')
        # Note: "def test" has not been tested; please use "def validate" for now: the two may be merged in the futuer)
        #validate(train_loader, model, criterion, outfile='train_result_mtrx.npy')
