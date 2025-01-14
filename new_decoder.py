import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, tau: float = 1.0, hard: bool = True, max_nodes: int = 50):
        super(RNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_nodes = max_nodes
        self.tau = tau
        self.hard = hard
        
        self.rnn = nn.GRU(
            input_size = latent_dim, 
            hidden_size = hidden_dim,
            num_layers = n_layers, 
            batch_first = True
        )
        self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.adj_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, z: torch.Tensor, n_nodes: int):
        batch_size = z.size(0)
        seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
        mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0).expand(batch_size, self.max_nodes) < n_nodes.unsqueeze(1)
        seq_input[~mask] = 0.0
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_input, n_nodes.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed_input)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=self.max_nodes) 
        node_emb = self.node_proj(rnn_out)
        idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1, device=z.device)
        emb_i = node_emb[:, idx[0], :]
        emb_j = node_emb[:, idx[1], :]
        pair_emb = torch.cat([emb_i, emb_j], dim=-1)
        logits = self.adj_mlp(pair_emb)
        adjacency_values = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)[..., 0]
        adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
        adj[:, idx[0], idx[1]] = adjacency_values
        adj = adj + torch.transpose(adj, 1, 2)
        indices = torch.arange(self.max_nodes, device=z.device).unsqueeze(0)  
        mask = indices < n_nodes.unsqueeze(1) 
        mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
        adj = adj * mask2d.float()
        return adj
    


class AttentionDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_heads, max_nodes):
        super(AttentionDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_nodes, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x, nb_nodes):
        # Projeter l'entrée pour obtenir des représentations initiales des nœuds
        x = self.input_proj(x)
        
        # Étendre les représentations à max_nodes avec des zéros pour aligner les dimensions
        x = x.unsqueeze(1).repeat(1, self.max_nodes, 1)
        
        # Ajouter l'encodage de position
        x = x + self.positional_encoding
        
        # Appliquer le mécanisme d'attention
        x, _ = self.attention(x, x, x)
        
        # Projeter les sorties pour obtenir des logits d'arêtes
        logits = self.output_proj(x)
        logits = logits[:, :, 0].unsqueeze(-1)  # Extraire les logits
        
        # Reshaper pour correspondre aux arêtes
        adj_logits = torch.matmul(logits, logits.transpose(1, 2))
        
        # Appliquer le masque pour garder uniquement les nœuds actifs
        indices = torch.arange(self.max_nodes, device=x.device).unsqueeze(0)
        mask = indices < nb_nodes.unsqueeze(1)
        mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
        adj_logits = adj_logits * mask2d.float()
        
        # Conversion en matrice d'adjacence avec Gumbel-Softmax
        adj = F.gumbel_softmax(adj_logits.view(x.size(0), -1, 2), tau=1, hard=True)[:, :, 0]
        adj = torch.zeros(x.size(0), self.max_nodes, self.max_nodes, device=x.device)
        idx = torch.triu_indices(self.max_nodes, self.max_nodes, 1)
        adj[:, idx[0], idx[1]] = adj
        adj = adj + adj.transpose(1, 2)
        
        return adj
