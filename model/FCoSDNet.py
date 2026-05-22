import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

from mamba_ssm import Mamba2, Mamba

from .FreqDec import FrequencyDecomposition, Temporal_Decomposition, PerNodeOrderedBandMask1
from .Embed import TimeEmbedding, FlowEmbedding
from .MultiPeriodFusion import MultiPeriodFusion, FrequencyAdaptiveFusion
from .RevIN import RevIN, RevIN_4D
from .MambaEnc import Encoder, EncoderLayer, GRULayer
from .Mamba2 import FCoSDModel2

class FCoSD(nn.Module):
    def __init__(self, **model_args):
        super(FCoSD, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.d_model = model_args['d_model']

        self.use_norm = model_args['use_norm']
        self.emb_dropout = model_args['emb_dropout']

        self.d_state = model_args['d_state']
        self.d_conv = model_args['d_conv']
        self.expand = model_args['expand']

        self.d_ff = model_args['d_ff']
        self.ffn_dropout = model_args['ffn_dropout']

        self.freq_bands =  [(float(a), float(b)) for a, b in model_args['freq_bands']]
        self.e_layers = model_args['e_layers']
        self.ffn_activation = model_args['ffn_activation']
        self.num_bands = model_args['num_bands']
        self.rank = model_args['rank']
        self.num_bands = model_args['num_bands']

        self.headdim = model_args['headdim']
        self.d_inner = model_args['d_inner']

        self.d_band = model_args['d_band']
        self.use_adj = model_args['use_adj']
        self.shared_band = model_args['shared_band']
        self.use_mamba2 = model_args['use_mamba2']
        self.use_mvg = model_args['use_mvg']
        self.use_rnn = model_args['use_rnn']

        self.build()

    def build(self):
        self.freq_dec = FrequencyDecomposition(self.freq_bands, self.emb_dropout)
        self.ordered_band_mask = PerNodeOrderedBandMask1(num_freqs=self.history_seq_len//2+1, num_nodes=self.num_channels, num_bands=self.num_bands, min_width=1e-3, init_width=0.08, d_band=self.d_band)
        self.learnable_freq_mask = Temporal_Decomposition()

        self.embed_temporal = TimeEmbedding(self.num_channels, self.history_seq_len, self.d_model, self.emb_dropout)
        self.embed_flow = FlowEmbedding(self.history_seq_len, self.d_model, self.emb_dropout)
        self.embed_num_channels = nn.Linear(self.num_channels, self.d_model)
        self.node_emb = nn.Parameter(torch.randn(self.num_channels, self.d_model))
        self.tod_emb = nn.Parameter(torch.randn(self.num_channels, self.d_model))
        self.dow_emb = nn.Parameter(torch.randn(self.num_channels, self.d_model))
        self.adj_emb = nn.Parameter(torch.randn(self.num_channels, self.num_channels))

        self.Dropout = nn.Dropout(p=self.emb_dropout)  # Dropout layer to prevent overfitting.

        self.node_bank = nn.Parameter(torch.randn(self.rank, self.d_model))  # Learnable memory matrix.
        self.W = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))  # Bias term.

        if not self.shared_band:
            # no shared parameters for all bands, each band has its own set of parameters
            self.node_bank_list = nn.ParameterList([
                nn.Parameter(torch.randn(self.rank, self.d_model)) for _ in range(3)
            ])
            self.W_list = nn.ParameterList([
                nn.Parameter(torch.randn(self.d_model, self.d_model)) for _ in range(3)
            ])
            self.bias_list = nn.ParameterList([
                nn.Parameter(torch.zeros(self.d_model)) for _ in range(3)
            ])
            self.projection3_list = nn.ModuleList([
            nn.Linear(self.d_model*2, self.d_model, bias=True) for _ in range(3)
            ])

        self.projection = nn.Linear(self.d_model, self.future_seq_len, bias=True)
        self.projection2 = nn.Linear(self.d_model, self.num_channels, bias=True)
        self.projection3 = nn.Linear(self.d_model*2, self.d_model, bias=True)
        self.projection4 = nn.Linear(self.history_seq_len//2+1, self.d_model, bias=True)
        self.multi_period_fusion = MultiPeriodFusion(self.d_model, self.future_seq_len, self.num_channels, self.emb_dropout, self.num_bands)
        self.freq_ada_fusion = FrequencyAdaptiveFusion(self.d_model, self.num_bands)

        self.revin_layer = RevIN(num_features=self.num_channels)
        self.revin_layer_4d = RevIN_4D(num_nodes=self.num_channels, num_features=3)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba2(self.d_model, self.d_state, self.d_conv, self.expand),
                    Mamba2(self.d_model, self.d_state, self.d_conv, self.expand),
                    self.d_model,
                    self.d_ff,
                    self.ffn_dropout,
                    self.ffn_activation
                ) for layer in range(self.e_layers)
            ],
            norm_layer = nn.LayerNorm(self.d_model)
        )
        self.encoder2 = FCoSDModel2(len=self.history_seq_len, d_model=self.d_model, n_layer=self.e_layers, d_intermediate=0, aux_feature_size=1,
                            d_state=self.d_state, headdim=self.headdim, d_inner=self.d_inner, dtype=torch.float32)
        
        self.encoder3 = Encoder(
            [
                EncoderLayer(
                    Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                    Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                    self.d_model,
                    self.d_ff,
                    self.ffn_dropout,
                    self.ffn_activation
                ) for layer in range(self.e_layers)
            ],
            norm_layer = nn.LayerNorm(self.d_model)
        )
        
        self.rnn =  Encoder(
            [
                GRULayer(
                    self.d_model,
                    self.d_ff,
                    self.ffn_dropout,
                    self.ffn_activation
                ) for layer in range(self.e_layers)
            ],
            norm_layer = nn.LayerNorm(self.d_model)
        )

    def forward(self,
                history_data: torch.Tensor, laplacian, perm=None) -> torch.Tensor:

        batch_size = history_data.shape[0]

        spatial_features = self.node_emb
        spatial_features = repeat(spatial_features, 'n d -> b n d', b=batch_size)

        if self.use_mvg:
            cl, lol, mwl = 0, 0, 0
            xk = self.learnable_freq_mask(history_data[...,0])
        else:
            Xk, xk, cl, lol, mwl = self.ordered_band_mask(history_data[...,0])

        Hp = []
        for i in range(len(xk)):
            xt_flow = self.embed_flow(xk[i].permute(0, 2, 1))
            xt_dow = self.embed_temporal(history_data[...,1].permute(0, 2, 1))
            xt_tod = self.embed_temporal(history_data[...,2].permute(0, 2, 1))

            xt_time = xt_tod + xt_dow
            if self.use_adj:
                adj = self.adj_emb
                xt_time = torch.einsum('nn, bnd -> bnd', adj, xt_time)
            else:
                adj = laplacian
                xt_time = torch.einsum('nn, bnd -> bnd', adj, xt_time)

            xp = xt_flow + xt_time + spatial_features

            if self.shared_band:
                # Compute Query: Q_k = W*xp + b.
                Q_k = torch.matmul(xp, self.W) + self.bias  # (B, N, D)
                
                # Attention scores: a_k = softmax(Q_k * M_k^T).
                M_k = self.node_bank  # Node bank transposed to (D, n).
                scores = torch.matmul(Q_k, M_k.T)  # (B, N, D)
                a_k = F.softmax(scores, dim=-1)  # Normalize along the D dimension.

                # Global information aggregation: O_k = a_k * M_k.
                O_k = torch.matmul(a_k, M_k)  # (B, N, D)

                # Concatenate original and global features: H^k = H^k || O_k.
                H_prime = torch.cat([xp, O_k], dim=-1)  # (B, N, 2D)
                # Project linearly back to the original dimension.
                xp = self.projection3(H_prime)
                ######
            else:
                # Compute Query: Q_k = W_k*xp + b_k.
                W_k = self.W_list[i]
                bias_k = self.bias_list[i]
                Q_k = torch.matmul(xp, W_k) + bias_k  # (B, N, D)
                
                # Attention scores: a_k = softmax(Q_k * M_k^T).
                M_k = self.node_bank_list[i]  # Node bank transposed to (D, n).
                scores = torch.matmul(Q_k, M_k.T)  # (B, N, D)
                a_k = F.softmax(scores, dim=-1)  # Normalize along the D dimension.

                # Global information aggregation: O_k = a_k * M_k.
                O_k = torch.matmul(a_k, M_k)  # (B, N, D)

                # Concatenate original and global features: H^k = H^k || O_k.
                H_prime = torch.cat([xp, O_k], dim=-1)  # (B, N, 2D)
                # Project linearly back to the original dimension.
                projection3_layer = self.projection3_list[i]
                xp = projection3_layer(H_prime)
            
            if self.use_mamba2:
                hp_final = self.encoder(xp)
            elif self.use_rnn:
                hp_final = self.rnn(xp)
            Hp.append(hp_final)

        # MultiPeriodFusion
        if self.use_mvg:
            H_p = (Hp[0]+Hp[1])/2.0
        else:
            H_p = self.multi_period_fusion(Hp)
        H_p = self.projection(H_p).permute(0, 2, 1)
        return H_p, cl, lol, mwl
