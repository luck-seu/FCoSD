import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPeriodFusion(nn.Module):
    def __init__(self, d_model, history_seq_len, num_channels, dropout_rate, num_bands):
        super(MultiPeriodFusion, self).__init__()
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(2 * d_model, d_model)
        self.linear2 = nn.Linear(history_seq_len, history_seq_len, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.Dropout = nn.Dropout(p=dropout_rate)  # Dropout layer to prevent overfitting.
        # self.W = nn.Linear(history_seq_len, 1, bias=False)  # Used for sigmoid weight computation.
        # self.W1 = nn.Linear(d_model, 1, bias=False)  # Used for sigmoid weight computation.
        # self.W2 = nn.Linear(num_channels, 1, bias=False)  # Used for sigmoid weight computation.
        # self.W3 = nn.Parameter(torch.randn(d_model, d_model))
        self.W_list = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_bands - 1)])
        self.linear_lst = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_bands)])

    def forward(self, H_p):

        # Start fusion from the high-frequency band.
        current_fused = H_p[-1]  # Highest-frequency features.
        
        # Store all intermediate fusion results.
        fused_results = [self.linear_lst[-1](current_fused)]
        
        # Progressively fuse from the next-highest frequency toward the low-frequency bands.
        for i in range(len(H_p) - 2, -1, -1):
            # Current lower-frequency band features.
            lower_band = H_p[i]
            # Compute gating weights: a = sigmoid(W(high-frequency fusion result + low-frequency features)).
            # a = self.sigmoid(self.W1(current_fused + lower_band))  # (B, N, 1)
            # a = self.sigmoid(torch.matmul((current_fused + lower_band), self.W3))
            a = self.sigmoid(self.W_list[i](current_fused + lower_band))  # (B, N, D)
            # Fuse: low-frequency features + a * high-frequency fusion result.
            current_fused = lower_band + a * current_fused
            # Save the current fusion result.
            fused_results.append(self.linear_lst[i](current_fused))
        
        # Final fusion: sum the fusion results from all bands.
        H_p = torch.sum(torch.stack(fused_results, dim=0), dim=0)

        return H_p
    
class FrequencyAdaptiveFusion(nn.Module):
    def __init__(self, d_model, num_bands):
        """
        Frequency-band feature aggregation module.
        :param d_model: temporal embedding dimension D
        :param num_bands: number of frequency bands K
        """
        super().__init__()
        # Linear transformation layers required by self-attention.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # Linear layer that generates the final weights.
        self.score_proj = nn.Linear(d_model, 1)
        
    def forward(self, H):
        """
        :param H: input frequency-band features with shape (K, B, N, D)
                  K: number of frequency bands, B: batch size, N: number of nodes, D: temporal embedding dimension
        :return: aggregated features with shape (B, N, D)
        """
        K, B, N, D = H.shape
        
        # 1. Apply average pooling over the node dimension N, producing shape (K, B, D).
        H_pooled = torch.mean(H, dim=2)  # Average over the node dimension.
        
        # 2. Use self-attention to compute scores, producing shape (K, B, K).
        Q = self.q_proj(H_pooled)  # (K, B, D)
        K_mat = self.k_proj(H_pooled)  # (K, B, D)
        V = self.v_proj(H_pooled)  # (K, B, D)
        
        # Compute attention scores after adjusting dimensions.
        # Adjust K_mat to shape (B, D, K).
        K_mat_transposed = K_mat.permute(1, 2, 0)  # (B, D, K)
        scores = torch.bmm(Q.permute(1, 0, 2), K_mat_transposed) / (D ** 0.5)  # (B, K, K)
        scores = F.softmax(scores, dim=-1)  # Normalize attention scores, shape (B, K, K).
        
        # 3. Multiply scores by V, producing shape (B, K, D).
        # Adjust V to shape (B, K, D).
        V_reshaped = V.permute(1, 0, 2)  # (B, K, D)
        weighted_v = torch.bmm(scores, V_reshaped)  # (B, K, D)
        
        # 4. Apply a linear transformation to obtain score_fn with shape (B, K, 1).
        score_fn = self.score_proj(weighted_v)  # (B, K, 1)
        score_fn = F.softmax(score_fn, dim=1)  # Normalize weights along the frequency-band dimension.
        
        # 5. Multiply score_fn with H, then sum along frequency-band dimension K to get (B, N, D).
        # Adjust H to shape (B, K, N, D).
        H_reshaped = H.permute(1, 0, 2, 3)  # (B, K, N, D)
        # Adjust score_fn to shape (B, K, 1, 1) for broadcasting.
        score_fn_expanded = score_fn.unsqueeze(-1)  # (B, K, 1, 1)
        # Elementwise multiply: (B, K, N, D) * (B, K, 1, 1) = (B, K, N, D).
        weighted_H = H_reshaped * score_fn_expanded
        # Sum along the frequency-band dimension K: (B, N, D).
        aggregated = torch.sum(weighted_H, dim=1)
        
        return aggregated
