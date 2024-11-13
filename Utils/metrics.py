import torch
import torch.nn.functional as F
from torcheval.metrics import R2Score
import numpy as np
from easydict import EasyDict as edict
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mse_loss(y_true, y_pred):
    mask = ~torch.isnan(y_true)
    loss = F.mse_loss(y_pred[mask], y_true[mask])
    return loss

def R2(true, pred):
    mask = ~torch.isnan(true)
    y_true, y_pred = true[mask], pred[mask]
    r2 = R2Score().to(true.device)
    r2.update(y_pred, y_true)
    r2_score = r2.compute()
    return r2_score.item()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~np.isnan(y_true)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-5)) * 100

def return_result(y_true,y_pred):
    mask = ~np.isnan(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    y_range = y_true.max() - y_true.min()
    performance = edict()
    performance.corr = stats.pearsonr(y_true, y_pred)[0]
    performance.r2 = r2_score(y_true=y_true, y_pred=y_pred)
    # performance.mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    performance.rmse = mean_squared_error(y_true=y_true,y_pred=y_pred, squared=True)
    performance.mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    performance.nrmse = performance.rmse / y_range
    performance.nmae = performance.mae / y_range
    
    return performance