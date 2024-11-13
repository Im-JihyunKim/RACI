import torch, math
import torch.nn as nn
import torch.nn.functional as F

def custom_init_weights(m):
    if isinstance(m, nn.MultiheadAttention):
        head_dim = m.embed_dim // m.num_heads
        for i in range(m.num_heads):
            nn.init.xavier_uniform_(m.in_proj_weight[i*head_dim:(i+1)*head_dim], gain=0.5 + 0.1*i)
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.out_proj.bias, 0.)

class Transformer(nn.Module):
    def __init__(self, configs, input_dim, num_et,
                 **kwargs):
        super(Transformer, self).__init__()
        self.configs = configs
        self.hidden_dim = configs.hidden_dim
        self.__dict__.update(kwargs)

        # input embedding
        self.et_emb_layer = nn.Sequential(
                                    nn.Linear(input_dim, input_dim),
                                    nn.ReLU())
                                    
        self.embedding = nn.Linear(1, configs.feature_dim)
        self.pos_encoder = PositionalEncoding(configs.feature_dim, configs.dropout_rate,
                                              input_dim)
        
        # feature extractpr
        self.enc_layer = nn.TransformerEncoderLayer(configs.feature_dim,
                                                    configs.n_head,
                                                    configs.dim_feedforward,
                                                    configs.dropout_rate,
                                                    batch_first=True)
        self.backbone = nn.TransformerEncoder(self.enc_layer, configs.num_encoder_layers)
        
        # ET regressor
        self.et_predictor = nn.Sequential(
                            nn.Linear(configs.feature_dim*input_dim, self.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim, num_et))
        
        self.apply(custom_init_weights)

    def forward(self, x, eqp):
        emb = self.embedding(x.unsqueeze(-1))      # [B, seq_len, feature_dim]
        emb_pos = self.pos_encoder(emb)
        z1 = self.backbone(emb_pos)                 # z=[B, seq_len, feature_dim]
        z1 = z1.view(z1.size(0), -1)                  # z=[B, seq_len*feature_dim]

        emb_eqp = self.embedding(eqp.unsqueeze(-1))
        emb_eqp_pos = self.pos_encoder(emb_eqp)
        z2 = self.backbone(emb_eqp_pos)
        z2 = z2.view(z2.size(0), -1)
        z2 = nn.AdaptiveAvgPool1d(z1.size(1))(z2)

        z = (z1+z2)/2
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # self.pe.size(): [seq_len, B, emb_dim] -> [B, seq_len, emb_dim]
        x = x + self.pe.permute(1, 0, 2)[:, :x.size(1), :]
        return self.dropout(x)
    
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