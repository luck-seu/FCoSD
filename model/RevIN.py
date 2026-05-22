import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, 
                 num_features: int,
                 eps=1e-5,
                 affine=True,
                 subtract_last=False):
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))


    def forward(self, 
                x, 
                mode:str):

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            self._get_statistics(x)
            x = self._denormalize(x)
        else: 
            raise NotImplementedError

        return x
    

    def _get_statistics(self, x):

        dim2reduce = tuple(range(1, x.ndim-1))

        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


    def _normalize(self, 
                   x):

        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x


    def _denormalize(self, 
                     x):

        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
            
        return x

class RevIN_4D(nn.Module):
    """
    Reversible normalization module for input shape (b, t, n, d).
    - Computes statistics along the time dimension t (dim=1) for each (b, n, d).
    """
    def __init__(self, num_nodes: int, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_nodes = num_nodes    # Number of nodes n.
        self.num_features = num_features  # Number of features d (3: flow, time of day, day of week).
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last  # Use True for non-stationary data by subtracting the last time step.

        # Affine parameters: (n, d) -> independent scaling and shift for each node and feature.
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_nodes, num_features))  # (n, d)
            self.affine_bias = nn.Parameter(torch.zeros(num_nodes, num_features))   # (n, d)

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)  # Save statistics (mean/last and standard deviation).
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Mode {mode} not supported")
        return x

    def _get_statistics(self, x):
        """
        x: (b, t, n, d)
        Statistics: compute mean/variance along the time dimension t (dim=1) for each (b, n, d).
        """
        dim2reduce = (1,)  # In the new shape, time steps are on dimension 1.

        if self.subtract_last:
            # Save the last time-step value (t=-1) for each (b, n, d).
            self.last = x[:, -1, :, :].unsqueeze(1)  # (b, 1, n, d), preserving t for broadcasting.
        else:
            # Compute the mean along t for each (b, n, d).
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # (b, 1, n, d)
        
        # Compute the standard deviation along t for each (b, n, d).
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()  # (b, 1, n, d)

    def _normalize(self, x):
        # 1. Remove the offset by subtracting the mean or last value.
        if self.subtract_last:
            x = x - self.last  # (b, t, n, d) - (b, 1, n, d), broadcast subtraction.
        else:
            x = x - self.mean
        # 2. Standardize by dividing by the standard deviation.
        x = x / self.stdev
        # 3. Optional affine transform with independent node-feature scaling and shift.
        if self.affine:
            # Broadcast from (n,d) to (1,1,n,d), aligning with x (b,t,n,d).
            x = x * self.affine_weight.unsqueeze(0).unsqueeze(0)
            x = x + self.affine_bias.unsqueeze(0).unsqueeze(0)
        return x

    def _denormalize(self, x):
        # Inverse operation: undo affine, multiply by standard deviation, then add the offset.
        if self.affine:
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(0)
            x = x / (self.affine_weight.unsqueeze(0).unsqueeze(0) + self.eps**2)  # Avoid division by zero.
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
