import numpy as np
import copy
from .calibration_metrics import CalibrationMetric

def init_config():
    """Initializer of Configration for CalibrationMetric."""

    n_bins = 10
    alpha = 1.0
    beta = 1.0
    config = {}
    config['num_reps'] = 100
    config['num_bins'] = n_bins
    config['split'] = ''
    config['norm'] = 1
    config['calibration_method'] = 'no_calibration'
    config['bin_method'] = ''
    config['d'] = 1
    config['alpha'] = alpha
    config['beta'] = beta
    config['a'] = alpha
    config['b'] = beta
    config['dataset'] = 'polynomial'
    config['ce_type'] = 'ew_ece_bin'
    config['num_samples'] = 5
    
    return config


def build_calibration_metric(config):
    """Builder of CalibrationMetric."""

    if config == None:
        config = init_config()

    ce_type = config['ce_type']
    num_bins = config['num_bins']
    bin_method = config['bin_method']
    norm = config['norm']

    # [4] Call CalibrationMetric constructor
    cm = CalibrationMetric(ce_type, num_bins, bin_method, norm)
    return cm


def calibrate(config, preds, labels_oneh):
    """Compute estimated calibration error.
        Args:
            config: configration dict
            pred: prediction score (fx)
            labels_oneh: one hot label (y)
        Return:
            ce: calibration error by using config['ce_type'] strategy
    """

    num_samples = config['num_samples']
    scores = preds.reshape((num_samples, 1))
    raw_labels = labels_oneh.reshape((num_samples, 1))

    # [3] Call build_calibration_metric function
    cm = build_calibration_metric(config)
    ce = cm.compute_error(scores, raw_labels)

    return 100 * ce


def calc_ece(config, preds, labels_oneh):
    """ A helper function to compute ECE.
        Args:
            config: configration dict
            pred: prediction score (fx)
            labels_oneh: one hot label (y)
        Return:
            ece: estimated (mean) calibration error by using config['ce_type'] strategy
    """

    saved_ece = []
    for _ in range(config['num_reps']):

        ce = calibrate(config, preds, labels_oneh)
        saved_ece.append(ce)
    ece = np.mean(saved_ece)
    return ece