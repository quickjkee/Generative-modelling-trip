import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from scipy import linalg

from utils.inception import InceptionV3


def fid(loader1, fid_cache, b_size=128, device='cpu', dims=2048):
    """
    Calculation FID between two dataloader
    :param loader1: (nn.DataLoader)
    :param loader2: (nn.DataLoader)
    :param b_size: (Int), batch size
    :param device: current working device
    :param dims: (Int) Dimensionality of Inception features to use.
    :return: (Float)
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics(loader1, model, b_size,
                                dims, device, 1)
    f = np.load(fid_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_activations(dataloader, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """
    :return: np.array(num images, dims) that contains the
             activations of the given tensor when feeding inception with the
             query tensor.
    """
    model.to(device)
    model.eval()

    pred_arr = np.empty((len(dataloader.dataset), dims))

    start_idx = 0
    for batch in tqdm(dataloader):
        if isinstance(batch, list):
            batch = batch[0].to(device)
        else:
            batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(loader, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(loader, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics(loader, model, batch_size, dims, device,
                       num_workers=1):
    m, s = calculate_activation_statistics(loader, model, batch_size,
                                           dims, device, num_workers)

    return m, s
