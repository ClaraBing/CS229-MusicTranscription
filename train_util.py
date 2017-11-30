import torch
import shutil

def print_config(args):
    print('Using ' + ('GPU' if args.cuda else 'CPU'))
    print('batch size: {:d}'.format(args.batch_size))
    print('epochs: {:d}'.format(args.epochs))
    print('lr: {:f} (interval={:d})'.format(args.lr, args.lr_interval))
    print('momentum: {:f}'.format(args.momentum))
    print('save: dir: {:s} / prefix: {:s}'.format(args.save_dir, args.save_prefix))

# Ref: https://github.com/pytorch/examples/blob/master/imagenet/main.py

class AverageMeter(object):
    # Compute & store current & average value
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


def accuracy(output, target, topk=(1,)):
    # Computes the precision@k for the specified values of k
    maxk = max(topk)
    batch_size = target.size(0)

    prob, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return prob, pred, res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

