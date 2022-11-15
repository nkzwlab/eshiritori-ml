import torch

# Taken from timm - https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def quickdraw_compute_metrics(p):
    acc1, acc5 = accuracy(
        torch.tensor(p.predictions),
        torch.tensor(p.label_ids), topk=(1, 5)
    )
    return {'acc1': acc1, 'acc5': acc5}