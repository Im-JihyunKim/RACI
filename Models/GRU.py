import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GRU(nn.Module):
    def __init__(self, configs, input_dim, vm_x_dim_list, num_et,
                 vm_num_pad, et_num_pad, **kwargs):
        super(GRU, self).__init__()
        self.configs = configs
        self.hidden_dim = configs.hidden_dim
        self.vm_x_dim_list = vm_x_dim_list

        self.num_layers = configs.num_layers
        self.bidirectional = configs.bidirectional
        self.num_direction = 2*self.num_layers if self.bidirectional else self.num_layers
        
        self.n_vm_vars = vm_num_pad if configs.encoding_type != 'OneHot' else 1
        self.n_et_vars = et_num_pad if configs.encoding_type != 'OneHot' else 1

        # VM embedding layer
        self.emb_layer = nn.ModuleList([
            nn.Linear(self.n_vm_vars, configs.emb_dim) for _ in range(len(vm_x_dim_list))
        ])

        self.et_emb_layer = nn.Sequential(
            nn.Linear(self.n_et_vars, configs.emb_dim),
            nn.ReLU()
        )

        # feature extractor
        self.backbone = nn.GRU(1, self.hidden_dim,
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
        
    def forward(self, x, eqp):
        # predict ET
        """
        x = (batch_size, padded_seq_len)
        """
        h_0 = self._init_hidden_state(x.size(0), x.device)
        # emb_im = self.et_emb_layer(x.unsqueeze(-1))                   # [B, emb_dim]
        z1, _ = self.backbone(x.unsqueeze(-1), h_0)
        z2, _ = self.backbone(eqp.unsqueeze(-1), h_0)

        z = self.aggregation(z1.mean(dim=1), z2.mean(dim=1), self.configs.agg)
        et_pred = self.et_predictor(z)
        return et_pred
    
    @staticmethod
    def aggregation(emb1, emb2, agg:str="mean"):
        if agg == "mean":
            emb = (emb1+emb2)/2
            return emb
        elif agg == "attn":
            attn_agg = AttentionAggregation(emb1.size(-1))
            emb = attn_agg(emb1, emb2)
            return emb
        elif agg == "channel":
            return torch.stack([emb1, emb2], dim=1)
        else:
            raise NotImplementedError("aggregation 다시 확인")
    
    def predict_vm(self, x_list):
        h_0 = self._init_hidden_state(x_list[0].size(0), x_list[0].device)

        vm_pred_list = []
        for i, x in enumerate(x_list):
            if i > 0:
                x_ = x.clone()
                vm_idx = (self.vm_x_dim_list)[:i]

                mask = torch.isnan(x_[:, vm_idx])
                x_[:, vm_idx] = x_[:, vm_idx].masked_scatter(mask, torch.cat(vm_pred_list, dim=-1)[mask].squeeze())
                x = x_
            x = x[:,:-2] if i != len(x_list)-1 else x[:,:-4]
            # emb = F.relu(self.emb_layer[i](x))
            z, _ = self.backbone(x.unsqueeze(-1), h_0)
            pred = self.vm_predictor(z.mean(dim=1))
            vm_pred_list.append(pred)
        
        vm_pred = torch.cat(vm_pred_list, dim=-1)
        return vm_pred
    
class AttentionAggregation(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, emb1, emb2):
        # Stack embeddings
        embs = torch.stack([emb1, emb2], dim=1)  # (batch, 2, emb_dim)
        
        # Self-attention
        q = self.query(embs)
        k = self.key(embs)
        v = self.value(embs)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(embs.size(-1))
        attention = F.softmax(scores, dim=-1)
        
        return torch.matmul(attention, v).sum(dim=1)