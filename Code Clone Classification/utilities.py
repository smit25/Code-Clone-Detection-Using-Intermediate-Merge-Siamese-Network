import torch


def threshold_sigmoid(t):
  """prob > 0.5 --> 1 else 0"""
  threashold = t.clone()
  threashold.data.fill_(0.5)
  return (t > threashold).float()


def threshold_contrastive(input1, input2, margin = 2.0):
    """dist < m --> 1 else 0"""
    diff = input1 - input2
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    threshold = dist.clone()
    threshold.data.fill_(margin)
    return (dist < threshold).float().view(-1, 1)


def count(T):
  return torch.count_nonzero(T).item()