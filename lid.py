import torch
import numpy as np
from scipy.spatial.distance import cdist

def gmean(input_x, dim=0):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def get_lid_r(data, reference):
    b = data.shape[0]
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2)
    a, idx = torch.sort(r, dim=1)
    return r, a, idx

def lid_mle(data, reference, k=20, get_idx=False, compute_mode='use_mm_for_euclid_dist_if_necessary', get_r=False):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k] / a[:, k].view(-1, 1) + 1.e-4), dim=1)
    if get_idx:
        return idx, lids
    elif get_r:
        return r, a, idx, lids
    return lids

def lid_mom_est(data, reference, k, get_idx=False, compute_mode='use_mm_for_euclid_dist_if_necessary', get_r=False):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    m = torch.mean(a[:, 1:k], dim=1)
    lids = m / (a[:, k] - m)
    if get_idx:
        return idx, lids
    elif get_r:
        return r, a, idx, lids
    return lids

def lid_mom_est_eps(data, reference, k, get_idx=False):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2)
    a, idx = torch.sort(r, dim=1)
    m = torch.mean(a[:, 1:k], dim=1)
    lids = m / ((a[:, k] - m) + 1.e-4)
    if get_idx:
        return idx, lids
    return lids

def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    def f(v): return - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


# lid of a single query point x
def mle_single(data, x, k=20, dist=True, metric='euclidean'):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    if dist:
        k = min(k, len(data)-1)

    def f(v): return - k / np.sum(np.log(v/v[-1]))
    if dist:
        a = cdist(x, data, metric=metric)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    else:
        a = data
        a = np.apply_along_axis(np.sort, axis=1, arr=a)
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]
