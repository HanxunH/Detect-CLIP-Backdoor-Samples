import argparse
import mlconfig
import torch
import random
import numpy as np
import datasets
import time
import util
import models
import os
import misc
import sys
import h5py
import json
from open_clip import get_tokenizer
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from exp_mgmt import ExperimentManager
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser(description='SSL-LID')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)


def fit_gmm_to_cos_sim(cos_sim):
    """
    Fits a Gaussian Mixture Model to the given cos sim.

    Args:
    cos_sim (np.array): An array of cos sim.
    n_components (int): The number of components for GMM. Default is 2.

    Returns:
    GaussianMixture: The fitted GMM model.
    np.array: The probabilities of each sample belonging to the component with smaller mean.
    """
    # Reshape cos_sim for GMM compatibility
    cos_sim = np.array(cos_sim).reshape(-1, 1)

    # Fit the GMM
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(cos_sim)

    # Predict probabilities
    probabilities = gmm.predict_proba(cos_sim)

    # Identify the component with the larger mean
    larger_mean_index = np.argmax(gmm.means_)

    # Return the GMM model and probabilities of belonging to the component with larger mean
    return gmm, probabilities[:, larger_mean_index]


def detection_error(preds, labels, pos_label=1):
    """Return the misclassification probability when TPR is 95%.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    # Get ratios of positives to negatives
    pos_ratio = sum(np.array(labels) == pos_label) / len(labels)
    neg_ratio = 1 - pos_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

    # Calc error for a given threshold (i.e. idx)
    # Calc is the (# of negatives * FNR) + (# of positives * FPR)
    _detection_error = lambda idx: neg_ratio * (1 - tpr[idx]) + pos_ratio * fpr[idx]

    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))


def fpr_at_95_tpr(preds, labels, pos_label=1):
    """Return the FPR when TPR is at minimum 95%.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def main():
    if not os.path.exists(os.path.join(exp.exp_path, 'train_poison_info.json')):
        return

    # Set up Experiments
    logger = exp.logger
    config = exp.config
    # Prepare Data
    filename = os.path.join(exp.exp_path, 'train_poison_info.json')
    with open(filename, 'r') as json_file:
        train_backdoor_info = json.load(json_file)
    data = config.dataset(train_backdoor_info=train_backdoor_info)
    train_poison_idx = train_backdoor_info['poison_indices']
    train_clean_idx = np.setdiff1d(range(len(data.train_set)) ,train_poison_idx)
    y = np.zeros((len(data.train_set)))
    y[train_poison_idx] = 1

    detector_types = [
        'ABL', 'CD', 'LID', 'IsolationForest', 'SLOF', 'DAO', 'KDistance', 'CLIPScores',
    ]
    results = {}

    for type in detector_types:
        try:
            if type == 'ABL':
                # Process ABL scores
                train_loss = []
                for e in range(10):
                    path = os.path.join(exp.exp_path, 'train_loss_epoch{:d}.h5'.format(e))
                    hf = h5py.File(path, 'r')
                    train_loss.append(hf['data'])
                train_loss = np.array(train_loss)
                scores = np.mean(train_loss, axis=0)
                scores = np.subtract(0, scores) # lower scores for backdoor
            elif type == 'GMM':
                # Process other scores
                path = os.path.join(exp.exp_path, 'CLIPScores_scores.h5')
                hf = h5py.File(path, 'r')
                scores = hf['data']
                scores = np.array(scores)
                sample_indices = np.arange(0, len(scores))
                sorted_indices = np.argsort(scores)[::-1]
                new_indices = sample_indices[sorted_indices]
                new_sim_np = scores[sorted_indices]
                # Fit GMM 
                gmm_model, clean_probabilities = fit_gmm_to_cos_sim(new_sim_np)
                scores = [clean_probabilities[np.where(new_indices==i)[0][0]] for i in range(len(data.train_set))]
                scores = np.subtract(0, scores) # lower scores for backdoor
            else:
                # Process other scores
                path = os.path.join(exp.exp_path, '{}_scores.h5'.format(type))
                hf = h5py.File(path, 'r')
                scores = hf['data']
                if 'Density' in type or 'IsolationForest' in type or 'CLIPScores' in type:
                    # lower scores for backdoor
                    scores = np.subtract(0, scores)

            # assert len(scores) == len(data.train_set)
            fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y, scores, pos_label=1)
            pr_auc = auc(recall, precision)
            fpr95 = fpr_at_95_tpr(scores, y, pos_label=1)
            error95 = detection_error(scores, y, pos_label=1)

            results[type] = {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'fpr@95': fpr95, 'error@95': error95}

            print('Detector: {}, ROC AUC: {:.4f}, PR AUC: {:.4f} fpr@95: {:.4f} error@95: {:.8f}'.format(type, roc_auc, pr_auc, fpr95, error95))

        except Exception as e:
            print(e)
            print('Detector: {} failed'.format(type))
            continue

    # Save results
    exp.save_eval_stats(results, 'backdoor_detection_results')

    return


if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    seed = args.seed
    args.gpu = device
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    experiment.config.dataset.seed = args.seed

    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)