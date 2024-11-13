import os, glob, tqdm, rich
import numpy as np
import pandas as pd
import torch

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Malgun Gothic')
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
sns.set_palette("Pastel1")

from Utils.metrics import return_result
from Utils.optimizer import get_scheduler

class Task(object):
    def __init__(self):
        self.checkpoint_dir = None

    def set_optimizers(self):
        if self.configs.optimizer == 'adam':
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.configs.lr,
                weight_decay=self.configs.weight_decay
            )
        elif self.configs.optimizer == 'sgd':
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.configs.lr,
                momentum=self.configs.momentum, weight_decay=self.configs.weight_decay
            )
        elif self.configs.optimizer == 'rms_prop':
            self.optim = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.configs.lr,
                weight_decay=self.configs.weight_decay
            )
        else:
            raise ValueError('In-valid optimizer choice')

        return self.optim

    def set_scheduler(self):
        assert self.configs.lr_scheduler is not None

        if self.configs.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=40, gamma=0.95)

        elif self.configs.lr_scheduler == "cosine":
            scheduler = get_scheduler(self.optim, "cosine", 40)

        elif self.configs.lr_scheduler == "lambda":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=0.5)

        elif self.configs.lr_scheduler == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=30, eta_min=0)

        elif self.configs.lr_scheduler == 'cosine_annealing_warm':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10,
                                                                                  T_mult=1, eta_min=0.00001)
        else:
            raise ValueError('In-valid lr_scheduler choice')

        return scheduler
    
    def reset_grad(self):
        self.optim.zero_grad()

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_model_from_checkpoint(self):
        raise NotImplementedError

    def load_history_from_checkpoint(self):
        raise NotImplementedError

    def save_vm_testing_result(self, wafer_id, y_true, y_pred):
        backbone = self.configs.backbone
        rich.print(f"Testing Done.. Save VM Testing Results ({backbone})")

        result_dir, vm_performance = os.path.join(self.ckpt_dir, 'VM_result_plot'), []
        os.makedirs(result_dir, exist_ok=True)

        for vm in tqdm.tqdm(y_true.columns):
            true, pred = y_true[vm], y_pred[vm]
            vmin, vmax = 0.0, 1.0
            
            plt.figure(figsize=(10, 10))
            p = return_result(true, pred)
            v = list(p.values())
            perf_legend = f'\ncorr : {v[0]:.2f} \nr2 : {v[1]:.2f} \nrmse : {v[2]:.2f} \nmae : {v[3]:.2f} \nnmae : {v[-1]:.2f}'
            plt.scatter(true, pred, label=perf_legend)
            plt.plot([vmin, vmax], [vmin, vmax], color='grey', linestyle='--')
            plt.xlabel('True')
            plt.ylabel('Pred')
            plt.legend(loc='upper left')
            plt.title(f"{vm}_{backbone}")
            save_path = os.path.join(result_dir, f"{vm}_{backbone}_result.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=350)
            plt.close('all')
            
            p['VM'] = vm
            performance = pd.DataFrame(p, index=[0])
            vm_performance.append(performance)
        
        performance = pd.concat(vm_performance).reset_index(drop=True)
        avg_performance = dict(performance.loc[:, :"nmae"].mean())
        avg_performance['VM'] = 'VM_mean'
        all_performance = return_result(y_true.values, y_pred.values)
        all_performance['VM'] = 'VM_all'
        final_performance = pd.concat([performance, 
                                       pd.DataFrame(avg_performance, index=[0]),
                                       pd.DataFrame(all_performance, index=[0])], axis=0).reset_index(drop=True)
        final_performance.to_csv(os.path.join(self.ckpt_dir, f'{backbone}_VM_performance.csv'))

        true, pred = y_true.agg('mean'), y_pred.agg('mean')
        
        vmin = np.min(np.concatenate([pred, true])) * 0.95
        vmax = np.max(np.concatenate([pred, true])) * 1.05

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = sns.color_palette("tab20", n_colors=len(y_true.columns))

        p = return_result(y_true.values, y_pred.values)
        v = list(p.values())
        perf_legend = f'\ncorr : {v[0]:.2f} \nr2 : {v[1]:.2f} \nrmse : {v[2]:.2f} \nmae : {v[3]:.2f} \nnmae : {v[-1]:.2f}'
        ax.plot([vmin, vmax], [vmin, vmax], color='grey', linestyle='dashed')

        for idx, vm in enumerate(y_true.columns):
            ax.scatter(true.loc[vm],
                       pred.loc[vm],
                       color=colors[idx], label=vm)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(perf_legend, fontsize=12)
        
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

        save_path = os.path.join(result_dir, f"{backbone}_VM_results.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=350)
        plt.close('all')

        y_pred.columns = ['Pred_'+vm for vm in y_pred.columns]
        info = pd.concat([wafer_id, y_true, y_pred], axis=1)
        info.to_csv(os.path.join(self.ckpt_dir, f"{backbone}_VM_pred_info.csv"), index=False)

    
    def save_et_testing_result(self, wafer_id, y_true, y_pred):
        backbone = self.configs.backbone
        rich.print(f"Testing Done.. Save ET Testing Results ({backbone})")

        result_dir, et_performance = os.path.join(self.ckpt_dir, 'ET_result_plot'), []
        os.makedirs(result_dir, exist_ok=True)

        for et in tqdm.tqdm(y_true.columns):
            true, pred = y_true[et], y_pred[et]
            vmin, vmax = 0.0, 1.0
            
            plt.figure(figsize=(10, 10))
            p = return_result(true, pred)
            v = list(p.values())
            perf_legend = f'\ncorr : {v[0]:.2f} \nr2 : {v[1]:.2f} \nrmse : {v[2]:.2f} \nmae : {v[3]:.2f} \nnmae : {v[-1]:.2f}'
            plt.scatter(true, pred, label=perf_legend)
            plt.plot([vmin, vmax], [vmin, vmax], color='grey', linestyle='--')
            plt.xlabel('True')
            plt.ylabel('Pred')
            plt.legend(loc='upper left')
            plt.title(f"{et}_{backbone}")
            save_path = os.path.join(result_dir, f"{et}_{backbone}_result.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=350)
            plt.close('all')
            
            p['ET'] = et
            performance = pd.DataFrame(p, index=[0])
            et_performance.append(performance)
        
        performance = pd.concat(et_performance).reset_index(drop=True)
        avg_performance = dict(performance.loc[:, :"nmae"].mean())
        avg_performance['ET'] = 'ET_mean'
        all_performance = return_result(y_true.values, y_pred.values)
        all_performance['ET'] = 'ET_all'
        final_performance = pd.concat([performance, 
                                       pd.DataFrame(avg_performance, index=[0]),
                                       pd.DataFrame(all_performance, index=[0])], axis=0).reset_index(drop=True)
        final_performance.to_csv(os.path.join(self.ckpt_dir, f'{backbone}_ET_performance.csv'))

        true, pred = y_true.agg('mean'), y_pred.agg('mean')
        
        vmin = np.min(np.concatenate([pred, true])) * 0.95
        vmax = np.max(np.concatenate([pred, true])) * 1.05

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = sns.color_palette("tab20", n_colors=len(y_true.columns))

        p = return_result(y_true.values, y_pred.values)
        v = list(p.values())
        perf_legend = f'\ncorr : {v[0]:.2f} \nr2 : {v[1]:.2f} \nrmse : {v[2]:.2f} \nmae : {v[3]:.2f} \nnmae : {v[-1]:.2f}'
        ax.plot([vmin, vmax], [vmin, vmax], color='grey', linestyle='dashed')

        for idx, et in enumerate(y_true.columns):
            ax.scatter(true.loc[et],
                       pred.loc[et],
                       color=colors[idx], label=et)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(perf_legend, fontsize=12)
        
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

        save_path = os.path.join(result_dir, f"{backbone}_ET_results.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=350)
        plt.close('all')

        y_pred.columns = ['Pred_'+et for et in y_pred.columns]
        info = pd.concat([wafer_id, y_true, y_pred], axis=1)
        info.to_csv(os.path.join(self.ckpt_dir, f"{backbone}_ET_pred_info.csv"), index=False)