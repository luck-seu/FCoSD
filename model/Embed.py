import torch.nn as nn

class TimeEmbedding(nn.Module):
    """Time embedding layer."""
    def __init__(self, num_nodes, seq_len, embed_dim, dropout_rate):
        super(TimeEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        self.feature_proj = nn.Linear(seq_len, embed_dim)
        self.Dropout = nn.Dropout(p=dropout_rate)  # Dropout layer to prevent overfitting.

    def forward(self, x):
        time_embedding = self.feature_proj(x)
        time_embedding = self.Dropout(time_embedding)  # Apply dropout.
        
        return time_embedding
    

class FlowEmbedding(nn.Module):
    def __init__(self, 
                 history_seq_len, 
                 d_model, 
                 dropout):
        super(FlowEmbedding, self).__init__()

        self.ValueEmb = nn.Linear(history_seq_len, d_model)
        self.Dropout = nn.Dropout(p=dropout)

    
    def forward(self, x_in):
        x_emb = self.ValueEmb(x_in)
        x_emb = self.Dropout(x_emb)

        return x_emb
