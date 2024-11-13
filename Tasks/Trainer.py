import os, rich, collections, pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress

from Dataloaders import get_loader
import Models
from Utils.metrics import mse_loss, R2
from Utils.logger import make_epoch_description
from Utils.earlystop import EarlyStopping
from Tasks.Base import Task

class Trainer(Task):
    def __init__(self, configs, device, ckpt_dir, **kwargs):
        super(Trainer, self).__init__()
        self.configs = configs
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.epochs = configs.epochs

        self.__dict__.update(kwargs)

    def prepare(self):
        writer = SummaryWriter(os.path.join(self.ckpt_dir))
        loader_dict = get_loader(self.configs)

        earlystopping = EarlyStopping(patience=self.configs.patience,
                                          path=os.path.join(self.ckpt_dir, "best_model.pth"))
        
        input_dim = loader_dict['train'].dataset.input_dim
        num_et = loader_dict['train'].dataset.num_out
        model = getattr(Models, self.configs.backbone)(self.configs, 
                                                       input_dim, 
                                                       num_et)
        
        return loader_dict, model.to(self.device), writer, earlystopping
    
    def run(self, logger):
        loader_dict, self.model, writer, earlystopping = self.prepare()
        train_loader = loader_dict['train']
        if self.configs.et_weight == "var":
            self.et_var = torch.nanquantile(train_loader.dataset.ET_y, 0.75, dim=0) - torch.nanquantile(train_loader.dataset.ET_y, 0.25, dim=0)
        else:
            self.et_var = torch.nanquantile(train_loader.dataset.ET_y, 1, dim=0) - torch.nanquantile(train_loader.dataset.ET_y, 0, dim=0)

        self.optim = self.set_optimizers()
        if self.configs.lr_scheduler is not None:
            lr_scheduler = self.set_scheduler()

        best_loss, best_epoch = float('inf'), 0
        for epoch in range(1, self.epochs+1):
            # training and evaluating           
            train_results = self.train(train_loader)
            eval_results = self.evaluate(loader_dict['valid'])

            # step learning rate scheduler
            if self.configs.lr_scheduler is not None:
                lr_scheduler.step()
                rich.print('lr', lr_scheduler.get_last_lr()[0])

            # training history
            self.epoch_history = collections.defaultdict(dict)
            for k, v1 in train_results.items():
                self.epoch_history[k]['train'] = v1
            for k, v2 in eval_results.items():
                self.epoch_history[k]['valid'] = v2

            # write tensorboard summary
            if writer is not None:
                for k, v in self.epoch_history.items():
                    for k_, v_ in v.items():
                        writer.add_scalar(f'{k}_{k_}', v_, global_step=epoch)

            early_thred = self.epoch_history['ET_loss']['valid']
            earlystopping(early_thred, self.model)
            if best_loss > early_thred:
                best_loss = early_thred
                best_epoch = epoch

            # logging
            log = make_epoch_description(
                    history=self.epoch_history,
                    current=epoch,
                    total=self.epochs,
                    best = best_epoch
                )
            logger.info(log)

            if earlystopping.early_stop:
                break

        # save checkpoint when training ends
        best_ckpt = torch.load(earlystopping.path)
        self.model.load_state_dict(best_ckpt)

        # testing
        ET_true, ET_pred, ET_id = self.test(loader_dict['test'])
        self.save_et_testing_result(ET_id, ET_true, ET_pred)

    def train(self, train_loader):
        self.model.train()
        result = {
            'loss':torch.zeros(len(train_loader)),
            'ET_loss':torch.zeros(len(train_loader)),
            'ET_r2':torch.zeros(len(train_loader)),
        }

        with Progress(transient=True, auto_refresh=False) as pg:
            task = pg.add_task(f"[bold red] Training...", total=len(train_loader))
            for i, et_batch in enumerate(train_loader):
                et_x, et_eqp, et_y, et_idx = et_batch
                et_x, et_eqp, et_y = et_x.to(self.device), et_eqp.to(self.device), et_y.to(self.device)
                
                """Predict ET"""
                # 1. forward
                et_pred = self.model(et_x, et_eqp)

                # 2. compute loss per ET
                et_loss_ = torch.stack([mse_loss(et_y[:,i], et_pred[:,i])/(self.et_var[i]+1e-6) for i in range(et_y.size(1))])
                et_loss_ = et_loss_[~torch.isnan(et_loss_)]
                et_loss = mse_loss(et_y, et_pred) if self.configs.et_weight == "same" else torch.mean(et_loss_)

                """Compute Final Loss"""
                loss = et_loss

                loss.backward(retain_graph=True)
                self.optim.step()
                self.reset_grad()

                result['loss'][i] = loss.item()
                result['ET_loss'][i] = et_loss.item()
                result['ET_r2'][i] = R2(et_y, et_pred)

                desc = f"[bold skyblue] [{i+1}/{len(train_loader)}]: "
                for k, v in result.items():
                    desc += f" {k} : {v[:i+1].mean():.4f} |"
                pg.update(task, advance=1., description=desc)
                pg.refresh()

        return {k: v.mean().item() for k, v in result.items()}
    
    @torch.no_grad()
    def evaluate(self, valid_loader):
        self.model.eval()

        num_iters = len(valid_loader)
        if self.configs.et_weight == "var":
            et_var = torch.nanquantile(valid_loader.dataset.ET_y, 0.75, dim=0) - torch.nanquantile(valid_loader.dataset.ET_y, 0.25, dim=0)
        else:
            et_var = torch.nanquantile(valid_loader.dataset.ET_y, 1, dim=0) - torch.nanquantile(valid_loader.dataset.ET_y, 0, dim=0)
        
        result = {
            'loss':torch.zeros(num_iters),
            'ET_loss':torch.zeros(num_iters),
            'ET_r2':torch.zeros(num_iters),
        }

        with Progress(transient=True, auto_refresh=False) as pg:
            task = pg.add_task(f"[bold yellow] Evaluating...", total=num_iters)

            for i, et_batch in enumerate(valid_loader):
                et_x, et_eqp, et_y, et_idx = et_batch
                et_x, et_eqp, et_y = et_x.to(self.device), et_eqp.to(self.device), et_y.to(self.device)

                """Predict ET"""
                et_pred = self.model(et_x, et_eqp)
                et_loss_ = torch.stack([mse_loss(et_y[:,i], et_pred[:,i])/(et_var[i]+1e-6) for i in range(et_y.size(1))])
                et_loss = mse_loss(et_y, et_pred) if self.configs.et_weight == "same" else et_loss_[~torch.isnan(et_loss_)].mean()

                result['loss'][i] = et_loss.item()
                result['ET_loss'][i] = et_loss.item()
                result['ET_r2'][i] = R2(et_y, et_pred)

                desc = f"[bold green] [{i+1}/{num_iters}]: "
                for k, v in result.items():
                    desc += f" {k} : {v[:i+1].mean():.4f} |"
                pg.update(task, advance=1., description=desc)
                pg.refresh()

        return {k: v.mean().item() for k, v in result.items()}
    
    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()

        num_iters = len(test_loader)
        result = {
            'et_wafer_id':[],
            'real_et':[],
            'pred_et':[]
        }

        with Progress(transient=True, auto_refresh=False) as pg:
            task = pg.add_task(f"[bold yellow] Testing...", total=num_iters)
            for i, et_batch in enumerate(test_loader):
                et_x, et_eqp, et_y, et_idx = et_batch
                et_x, et_eqp, et_y = et_x.to(self.device), et_eqp.to(self.device), et_y.to(self.device)

                """VM 결측 -> pred 값으로 대체"""
                et_pred = self.model(et_x, et_eqp)

                result['et_wafer_id'].append(et_idx)
                result['real_et'].append(et_y.cpu().numpy())
                result['pred_et'].append(et_pred.cpu().detach().numpy())

                desc = f"[bold green] [{i+1}/{num_iters}]: "
                pg.update(task, advance=1., description=desc)
                pg.refresh()

        ET_true = pd.DataFrame(np.concatenate(result['real_et']), columns=['ET_'+str(i+1) for i in range(et_y.shape[-1])])
        ET_pred = pd.DataFrame(np.concatenate(result['pred_et']), columns=['ET_'+str(i+1) for i in range(et_y.shape[-1])])
        ET_wafer_id = pd.DataFrame(np.concatenate(result['et_wafer_id']), columns=['wafer_id'])

        return ET_true, ET_pred, ET_wafer_id