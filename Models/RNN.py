import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, configs, input_dim, vm_x_dim_list, num_et,
                 vm_num_pad, et_num_pad, et_pad_col_idx, **kwargs):
        super(RNN, self).__init__()
        self.configs = configs
        self.hidden_dim = configs.hidden_dim
        self.vm_x_dim_list = vm_x_dim_list

        self.num_layers = configs.num_layers
        self.bidirectional = configs.bidirectional
        self.num_direction = 2*self.num_layers if self.bidirectional else self.num_layers
        
        self.n_vm_vars = vm_num_pad if configs.encoding_type != 'OneHot' else 1
        self.n_et_vars = et_num_pad if configs.encoding_type != 'OneHot' else 1
        self.et_pad_col_idx = et_pad_col_idx

        # VM embedding layer
        self.emb_layer = nn.ModuleList([
            nn.Linear(self.n_vm_vars, configs.emb_dim) for _ in range(len(vm_x_dim_list))
        ])

        self.et_emb_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )

        # feature extractor
        self.backbone = nn.RNN(self.n_et_vars, self.hidden_dim,
                                self.num_layers, batch_first=True,
                                dropout=configs.dropout_rate,
                                bidirectional=self.bidirectional)

        # VM regressor
        self.vm_predictor = nn.Sequential(
                            nn.Linear(self.hidden_dim*2, self.hidden_dim//4) if self.bidirectional else \
                                nn.Linear(self.hidden_dim, self.hidden_dim//4),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim//4, 1)
                        )
        
        # ET regressor
        self.et_predictor = nn.Sequential(
                            nn.Linear(self.hidden_dim*2, self.hidden_dim//4) if self.bidirectional else \
                                nn.Linear(self.hidden_dim, self.hidden_dim//4),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim//4, num_et)
                        )
        
    def _init_hidden_state(self, batch_size, device):
        h_0 = Variable(torch.zeros(self.num_direction, batch_size, self.hidden_dim)).to(device)
        return h_0
        
    def forward(self, x):
        # predict ET
        """
        x = (batch_size, padded_seq_len)
        """
        h_0 = self._init_hidden_state(x.size(0), x.device)
        emb = self.et_emb_layer(x)
        emb = emb.view(x.size(0), -1, self.n_et_vars)
        # emb = self.et_emb_layer(x_)
        z, _ = self.backbone(emb, h_0)
        et_pred = self.et_predictor(z.mean(dim=1))
        return et_pred
    
    def predict_vm(self, x_list):
        vm_pred_list = []
        for i, x in enumerate(x_list):
            if i > 0:
                x_ = x.clone()
                vm_idx = (self.vm_x_dim_list)[:i]

                mask = torch.isnan(x_[:, vm_idx])
                x_[:, vm_idx] = x_[:, vm_idx].masked_scatter(mask, torch.cat(vm_pred_list, dim=-1)[mask].squeeze())
                x = x_
            x = x[:,:-1] if i != len(x_list)-1 else x[:,:-4]
            # x = x.unsqueeze(-1).view(x.size(0), -1, self.n_vm_vars)
            x = x.view(x.size(0), -1, self.n_vm_vars)
            emb = F.relu(self.emb_layer[i](x))
            h_0 = self._init_hidden_state(x_list[0].size(0), x_list[0].device)
            z, _ = self.backbone(emb, h_0)
            pred = self.vm_predictor(z.mean(dim=1))
            assert torch.isnan(pred).sum()==0
            vm_pred_list.append(pred)
        
        vm_pred = torch.cat(vm_pred_list, dim=-1)
        return vm_pred