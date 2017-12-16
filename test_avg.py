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
from train_util import *
from config import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mode', type=str,
                    help='Running mode: train / test')
# NOTE: please specify pretrained model in config.py
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--config', type = str, default='avg', help='which config to use, see config.py')

# NOTE: save_dir and save_prefix are moved to config.py

args = parser.parse_args()
args.cuda = True #  not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# Configuration

# NOTE: use argument --config to specify the configuration to be used
cfg = None
if args.config == 'mixed':
  cfg = config_mixed()
elif args.config == 'cqt':
  cfg = config_cqt()
elif args.config == 'context6':
  cfg = config_context6()
elif args.config == 'stacking':
  cfg = config_stacking()
elif args.config == 'early_fusion':
  cfg = config_early_fusion()
elif args.config == 'late_fusion':
  cfg = config_late_fusion()
elif args.config == 'mel':
  cfg = config_mel_test()
elif args.config == 'avg':
  cfg = config_avg_test()
print ("Using configuration %s" % args.config)
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

# train
train_set = PitchEstimationDataSet(cfg['annot_folder']+'train/', cfg['image_folder']+'train/', sr_ratio=cfg['sr_ratio'], audio_type=cfg['audio_type'], multiple=cfg['multiple'], fusion_mode='late_fusion')
train_loader = DataLoader(train_set, batch_size=1, shuffle=False, **kwargs)

# val
val_set = PitchEstimationDataSet(cfg['annot_folder']+'val/', cfg['image_folder']+'val/', sr_ratio=cfg['sr_ratio'], audio_type=cfg['audio_type'], multiple=cfg['multiple'], fusion_mode='late_fusion')
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)

# test
test_set = PitchEstimationDataSet(cfg['annot_folder']+'test/', cfg['image_folder']+'test/', sr_ratio=cfg['sr_ratio'], audio_type=cfg['audio_type'], multiple=cfg['multiple'], fusion_mode='late_fusion')
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)


def validate_avg(data_loader, model_mel, model_cqt, criterion, outfile=None):
    out_mtrx = np.empty((len(data_loader), 109, 2))

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()

    model_mel.eval()
    model_cqt.eval()

    for batch_idx, dictionary in enumerate(data_loader):
        batch_start = time()

        mel, cqt, target = Variable(dictionary['mel'], volatile=True).type(torch.FloatTensor), Variable(dictionary['cqt'], volatile=True).type(torch.FloatTensor), Variable(dictionary['frequency']).type(torch.LongTensor) # NOTE: may need to change back to LongTensor for single note
        if args.cuda:
            mel, cqt, target = mel.cuda(), cqt.cuda(), target.cuda()

        # compute output
        output_mel = model_mel(mel)
        output_cqt = model_cqt(cqt)
        output = (output_mel + output_cqt) / 2
        # performance measure: loss & top 1/5 accuracy
        loss = criterion(output, target)
        losses.update(loss.data[0], mel.size(0))
        prec1, prec5 = accuracy(output.data, target.data, topk=(1,5))
        top1.update(prec1[0], mel.size(0))
        top5.update(prec5[0], mel.size(0))
        # Save probabilities & corresponding pitch bins
        probs, pitch_bins = torch.sort(output.data, 1, True) # params: data, axis, descending
        # out_mtrx[batch_idx, :, 0] = np.exp(probs.view(-1).cpu().numpy())
        # out_mtrx[batch_idx, :, 1] = pitch_bins.view(-1).cpu().numpy()
            # prob_list, pitch_bin_list = list(probs.view(-1)), list(pitch_bins.view(-1)) 
            # for prob, pitch_bin in zip(prob_list, pitch_bin_list):
                


        batch_time.update(time() - batch_start)
        
        if batch_idx % (200*args.log_interval) == 0:
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



if __name__ == '__main__':
    model_mel = Net(fusion_mode=cfg['fusion_mode'])
    model_cqt = Net(fusion_mode=cfg['fusion_mode'])
    if args.cuda:
        model_mel.cuda()
        model_cqt.cuda()
    # print_config(args, cfg)
    # model_mel.print_archi()

    # input / target are floats
    criterion = F.nll_loss
    # testing
    pretrained_dict_mel = torch.load(cfg['pretrained_path_mel'])['state_dict']
    model_mel.load_state_dict(pretrained_dict_mel)
    model_mel.cuda()
    pretrained_dict_cqt = torch.load(cfg['pretrained_path_cqt'])['state_dict']
    model_cqt.load_state_dict(pretrained_dict_cqt)
    model_cqt.cuda()
    if args.mode == 'features':
      # Get features
      visualise_features(train_loader, model, criterion, outfile_features='dataset/' + cfg['save_prefix'] + '_features_mtrx.npy', 
          outfile_annotations='dataset/'+cfg['save_prefix']+'_features_annotations.npy')
    else:
    # Note: "def test" has not been tested; please use "def validate" for now: the two may be merged in the futuer)
      validate_avg(train_loader, model_mel, model_cqt, criterion) #, outfile='dataset/test_result_mtrx_avg.npy')
      validate_avg(val_loader, model_mel, model_cqt, criterion)
