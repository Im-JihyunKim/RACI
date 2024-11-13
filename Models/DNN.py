import torch, math
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, configs, input_dim, num_et,
                 **kwargs):
        super(DNN, self).__init__()
        self.configs = configs
        self.hidden_dim = configs.hidden_dim

        # embedding layer
        self.et_emb_layer = nn.Sequential(
            nn.Linear(input_dim, configs.emb_dim),
            nn.ReLU()
        )
        self.eqp_emb = nn.Sequential(
            nn.Linear(89, configs.emb_dim),
            nn.ReLU()
        )

        # feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(configs.emb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout_rate),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout_rate),

            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(configs.dropout_rate),
        )
        
        # ET regressor
        self.et_predictor = nn.Sequential(
                            nn.Linear(self.hidden_dim//2, self.hidden_dim//4),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim//4, num_et)
                        )
    
    def forward(self, x, eqp):
        emb_im = self.et_emb_layer(x)
        emb_eqp = self.eqp_emb(eqp)

        z1 = self.backbone(emb_im)
        z2 = self.backbone(emb_eqp)
        z = self.aggregation(z1, z2, self.configs.agg)
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