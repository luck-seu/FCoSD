import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9

class FrequencyDecomposition(nn.Module):
    """FFT-based multi-frequency component decomposition."""
    def __init__(self, freq_bands=[(0, 0.05), (0.05, 0.1), (0.1, 0.5)], dropout_rate=0.1):
        super(FrequencyDecomposition, self).__init__()
        self.freq_bands = freq_bands
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        """
        x: input tensor [batch_size, seq_len, num_nodes, features]
        """
        batch_size, seq_len, num_nodes, features = x.shape 
        x_in = x[...,[0]]
        
        # Apply FFT to each node's time series.
        x_fft = torch.fft.rfft(x_in, dim=1)  # [batch_size, seq_len//2+1, num_nodes, 1]
        
        # Frequency axis.
        freqs = torch.fft.rfftfreq(seq_len)

        # Decompose by frequency bands.
        components = []
        for low_freq, high_freq in self.freq_bands:
            # Create the frequency mask.
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            mask = mask.to(x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, seq_len//2+1, 1, 1]

            # Apply the mask.
            x_filtered = x_fft * mask

            # Convert back to the time domain with inverse FFT.
            x_component = torch.fft.irfft(x_filtered, n=seq_len, dim=1)  # [batch_size, seq_len, num_nodes, 1]
            x_component = self.emb_dropout(x_component)

            # Get the other features from the original input x (the second and third features).
            other_features = x[..., 1:]

            # Concatenate along the feature dimension (the last dimension).
            combined_component = torch.cat([x_component, other_features], dim=-1)
            
            components.append(combined_component)
        
        return components

class PerNodeOrderedBandMask1(nn.Module):
    """
    Module that learns frequency band masks separately for each node.
    Input x shape: [B, L, N, C] (batch_size, time steps, number of nodes, number of features).
    Output: frequency-domain decomposition results and regularization losses for each node.
    """
    def __init__(self, num_freqs, num_nodes, num_bands=3, init_deltas=None, init_width=0.08, min_width=1e-3, d_band=512):
        super().__init__()
        self.num_freqs = int(num_freqs)  # F = L//2 + 1
        self.num_nodes = int(num_nodes)  # N: number of nodes
        self.num_bands = int(num_bands)  # K: number of frequency bands
        self.min_width = float(min_width)

        self.node_random_vec = nn.Parameter(
            torch.randn(num_nodes, d_band) * 0.01  # Initialize with small random values to avoid large initial fluctuations.
        )
        # Linear layer: maps vectors to num_bands dimensions as raw parameters for the initial centers.
        self.proj_to_centers = nn.Linear(d_band, num_bands)

        # 2. Keep width parameters unchanged.
        init_widths = torch.full((num_nodes, num_bands), init_width, dtype=torch.float32)
        self.raw_widths = nn.Parameter(torch.log(torch.exp(init_widths) - 1.0 + EPS))  # inv_softplus

        # Normalized frequency axis [0,1].
        self.omega = torch.linspace(0.0, 1.0, steps=num_freqs)  # [F]

    def forward(self, x):
        B, L, N = x.shape  # [32, 96, 100, 3]

        centers_raw = self.proj_to_centers(self.node_random_vec)  # [N, num_bands]
        centers_raw = torch.sigmoid(centers_raw)  # Map to (0,1).

        # Use the cumulative sum of softmax outputs to ensure centers increase monotonically.
        centers = torch.cumsum(F.softmax(centers_raw * 10, dim=1), dim=1)  # Scale before softmax to improve separation.
        centers = centers / (centers[:, -1:] + EPS)  # Normalize to [0,1], with the last center close to 1.

        widths = F.softplus(self.raw_widths) + self.min_width  # [N, K]

        # 2. Generate masks and adjust their shape.
        omega = self.omega.to(centers.device)  # [Fq=49]
        c = centers.unsqueeze(-1)  # [N, K, 1]
        w = widths.unsqueeze(-1)   # [N, K, 1]
        omega_expand = omega.view(1, 1, -1)  # [1, 1, Fq=49]
        masks = torch.exp(-0.5 * ((omega_expand - c) **2) / (w** 2))  # [N, K, Fq]
        masks = masks / (masks.sum(dim=1, keepdim=True) + EPS)  # [N, K, Fq] (normalized within each node)

        # Adjust the mask shape to [K, 1, N, Fq, 1] to align with X_fft dimensions.
        masks = masks.permute(1, 0, 2)  # [K, N, Fq] -> transpose to put frequency bands first.
        # Flip masks along Fq first, then along K, to preserve low-to-high frequency order.
        masks = torch.flip(masks, dims=[2])  # Flip F first.
        masks = torch.flip(masks, dims=[0])  # Then flip K.
        masks = masks.unsqueeze(1) # [K, 1, N, Fq, 1]

        # 3. Apply FFT and masks after dimension alignment.
        X_fft = torch.fft.rfft(x, dim=1)  # [B, Fq=49, N=100, C=3]
        X_fft = X_fft.permute(0, 2, 1)  # Adjust to [B, N=100, Fq=49, C=3] by moving the node dimension forward.
        
        # At this point, masks [K,1,N,Fq,1] can be broadcast-multiplied with X_fft [B,N,Fq,C].
        Xk = masks * X_fft.unsqueeze(0)  # [K, B, N, Fq, C]

        # 4. Convert back to the time domain with inverse FFT while keeping dimensions consistent.
        Xk = Xk.permute(0, 1, 3, 2)  # Adjust to [K, B, Fq, N, C] for irfft.
        xk_time = torch.fft.irfft(Xk, n=L, dim=2)  # [K, B, L, N, C]

        # 5. Compute losses as before.
        cl = self.coverage_loss(masks.squeeze(1))  # Restore masks to [K, N, Fq].
        lol = self.low_overlap_loss(masks.squeeze(1))
        mwl = self.min_width_loss(widths)

        return Xk, xk_time, cl, lol, mwl

    # ---- Regularization losses adapted to per-node masks ----
    def coverage_loss(self, masks):
        """Mask coverage loss: ensures masks sum to 1 at each frequency bin, computed per node."""
        # masks shape: [K, N, F] -> sum along K to get [N, F], then compute the deviation from 1.
        return ((masks.sum(dim=0) - 1.0) **2).mean()  # Average over all nodes and frequencies.

    def low_overlap_loss(self, masks):
        """Low-overlap loss: encourages different band masks for the same node to have low overlap."""
        # masks shape: [K, N, F] -> convert to [N, K, F] for per-node computation.
        masks_per_node = masks.permute(1, 0, 2)  # [N, K, F]
        total_loss = 0.0
        for n in range(self.num_nodes):
            # Compute the cosine similarity matrix for each node.
            node_masks = masks_per_node[n]  # [K, F]
            G = F.cosine_similarity(node_masks.unsqueeze(1), node_masks.unsqueeze(0), dim=-1)  # [K, K]
            I = torch.eye(self.num_bands, device=G.device)
            total_loss += ((G - I)** 2).mean()
        return total_loss / self.num_nodes  # Average over all nodes.

    def min_width_loss(self, widths):
        """Minimum width loss: ensures widths are not smaller than min_width, averaged over all nodes and bands."""
        # widths shape: [N, K]
        return torch.clamp(self.min_width - widths, min=0.0).mean()


class Temporal_Decomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super(Temporal_Decomposition, self).__init__()

        self.moving_avg = moving_avg(kernel_size, stride=1)


    def forward(self, x):

        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        
        # res represents the seasonal component, and moving_mean represents the trend component.
        # Concatenate into 2,B,T,N.
        return torch.cat([moving_mean.unsqueeze(0), res.unsqueeze(0)], dim=0)


class moving_avg(nn.Module):
    def __init__(self, 
                 kernel_size=25, 
                 stride=1):
        super(moving_avg, self).__init__()

        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)


    def forward(self, x):
        # x:B,L,N
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x
