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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mode', type=str,
                    help='Running mode: train / test')
parser.add_argument('--load-model', type=str, default=None,
                    help='Path to the pretrained model')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
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
parser.add_argument('--save-dir', type=str, default='./output_model/', metavar='N',
                    help='save directory of trained models')
parser.add_argument('--save-prefix', type=str, default='model_conv5_train', metavar='N',
                    help='prefix of trained models')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# train
annotations_train = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/new_data/train/'
training_set = PitchEstimationDataSet(annotations_train, '/root/new_data/orig/train/')
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

# val
annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/new_data/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/new_data/orig/val/')
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
    # batch_size = args.batch_size, shuffle=False, **kwargs)

# test
annotations_test = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/new_data/test/'
# test_set = PitchEstimationDataSet(annotations_test, 'root/data/test', transform=transforms.Compose([
#                transforms.ToTensor(),
#                transforms.Normalize((0.1307,), (0.3081,))
#                ]))
test_set = PitchEstimationDataSet(annotations_test, '/root/new_data/orig/test')
test_loader = DataLoader(test_set, shuffle=False, # do not shuffle: the original ordering is needed for matching w/ annotations (for HMM)
    batch_size = args.test_batch_size, **kwargs) # batch = 1

# raise ValueError('should stop here')

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(model, train_loader, criterion, epoch):
    model.train()
    batch_start = time()
    avg_loss, prev_avg_loss = 0, 1000
    for batch_idx, dictionary in enumerate(train_loader):
        data = dictionary['image']
        target = dictionary['frequency']
        data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
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
            save_name = args.save_prefix + str(batch_idx) + '.pt'
            print('Saving model: ' + save_name)
            # torch.save(model.state_dict(), args.save_dir+save_name)
            # TODO: save also the optimizer (right now the lr can be found in the log)
            torch.save(model, args.save_dir+save_name)


def validate(data_loader, model, criterion, outfile=None):
    out_mtrx = np.empty((len(data_loader), 109, 2))

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()

    model.eval()

    for batch_idx, dictionary in enumerate(data_loader):
        batch_start = time()

        data, target = Variable(dictionary['image'], volatile=True).type(torch.FloatTensor), Variable(dictionary['frequency']).type(torch.LongTensor)
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
    print_config(args)
    model.print_archi()

    criterion = F.nll_loss
    best_prec = 0
    if args.mode == 'train':
        if not os.path.exists(args.save_dir):
          os.mkdir(args.save_dir)
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
            }, is_best, filename=args.save_dir+args.save_prefix+'_epoch{:d}.pt'.format(epoch))
    
            # update lr
            if epoch<100:
                # print('checking for avg loss')
                # avg_loss = avg_loss / args.lr_interval
                # if prev_avg_loss - avg_loss < 0.05:
                if epoch % 30 == 0:
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
        pretrained_dict = torch.load(args.load_model)['state_dict']
        model.load_state_dict(pretrained_dict)
        model.cuda()
        # Note: "def test" has not been tested; please use "def validate" for now: the two may be merged in the futuer)
        validate(train_loader, model, criterion, outfile='train_result_mtrx.npy')
