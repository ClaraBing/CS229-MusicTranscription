# utils
from __future__ import print_function
import argparse
from time import time
# torch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
# our code
from PitchEstimationDataSet import *
from model import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=500, metavar='N',
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
annotations_train="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/"
# annotations_test="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/onefiletest/"

# train
# annotations_train = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/'
training_set = PitchEstimationDataSet(annotations_train, '/root/data/train/')
# print (training_set[150]['image'].shape, training_set[150]['frequency'])
train_loader = DataLoader(training_set,
    batch_size = args.batch_size, shuffle = True, **kwargs)

# test
# annotations_test = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/test/'
# test_loader = DataLoader(
#     PitchEstimationDataSet(annotations_test, '../data/onefiletest', transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(model, train_loader, epoch):
    model.train()
    batch_start = time()
    for batch_idx, dictionary in enumerate(train_loader):
        data = dictionary['image']
        target = dictionary['frequency']
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
        # print('data & target:')
        # print(data.data.shape)
        # print(target.data.shape)
        optimizer.zero_grad()
        output = model(data)
        # print('output:')
        # print(output.data.shape)
        # print(output.data.max())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # training log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime per batch: {:.6f}s'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],
                (time()-batch_start)/args.log_interval))
            batch_start = time()
        # save trained model
        if batch_idx % args.save_interval == 0:
            save_name = args.save_prefix + str(batch_idx) + '.pt'
            print('Saving model: ' + save_name)
            # torch.save(model.state_dict(), args.save_dir+save_name)
            torch.save(model, args.save_dir+save_name)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, epoch)
        # model = torch.load('model_conv2_onefile_full_15.pt')
        # test(model, test_loader)
