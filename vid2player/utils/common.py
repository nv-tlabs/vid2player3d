import numpy as np
import torch
import random
import datetime


class AverageMeter(object):

    def __init__(self, avg=None, count=1):
        self.reset()
        if avg is not None:
            self.val = avg
            self.avg = avg
            self.count = count
            self.sum = avg * count
    
    def __repr__(self) -> str:
        return f'{self.avg: .4f}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class AverageMeterUnSync(object):

    def __init__(self, dim=1):
        self.sum = torch.zeros(dim)
        self.count = torch.zeros(dim, dtype=torch.int64)
        self.avg = torch.zeros(dim)
        self.max = torch.zeros(dim)
        self.reset()
    
    def __repr__(self) -> str:
        return f'{self.avg: .4f}'

    def reset(self):
        self.sum[:] = 0
        self.count[:] = 0
        self.avg[:] = 1
        self.max[:] = 0

    def update(self, val, ids):
        val = val.cpu()
        ids = ids.cpu()
        self.sum[ids] += val
        self.count[ids] += 1
        non_zero = self.count > 0
        self.avg[non_zero] = self.sum[non_zero] / self.count[non_zero]
        self.max[ids] = torch.where(
            val > self.max[ids],
            val,
            self.max[ids],
        )


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return convert_sec_to_time(eta)


def convert_sec_to_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))


def set_seed(seed):
    random.seed(seed)
    seed = random.randint(0, 1e9)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def array2list(arr):
    if arr is None: return None
    if isinstance(arr, list) or isinstance(arr, tuple) or len(arr.shape) == 1:
        return [float(i) for i in arr]
    else:
        return arr.tolist()


def concat(a, b, axis=0):
    if a is None:
        return b
    if b is None:
        return a
    return np.concatenate((a, b), axis)


def concat_torch(a, b, dim=0):
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat((a, b), dim)


def test_point_in_bbox(pt, bbox):
    return pt[0] >= bbox[0] and pt[0] < bbox[2] and pt[1] >= bbox[1] and pt[1] < bbox[3]


def get_opponent_env_ids(env_ids):
    if len(env_ids) == 0: return env_ids
    is_even = env_ids % 2 == 0
    oppo_env_ids = torch.cat([env_ids[is_even] + 1, env_ids[~is_even] - 1])
    return oppo_env_ids.sort().values