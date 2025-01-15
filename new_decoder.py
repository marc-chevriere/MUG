import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import fixed_positional_encoding
    

class RNNDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, max_nodes: int = 50, tau: float = 1.0, hard: bool = True,  batch_size: int=256):
        super(RNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_nodes = max_nodes
        self.tau = tau
        self.hard = hard
        self.batch_size = batch_size
        
        self.rnn = nn.GRU(
            input_size = latent_dim, 
            hidden_size = hidden_dim,
            num_layers = n_layers, 
            batch_first = True
        )
        self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_pre_rnn = nn.Linear(2*latent_dim, latent_dim)
        self.adj_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        # self.pairwise_attention = PairwiseAttention(hidden_dim=self.hidden_dim, num_heads=4)
        self.embeddings = nn.Embedding(self.max_nodes, latent_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encodings = fixed_positional_encoding(max_nodes, latent_dim, device=device)
        
    def forward(self, z: torch.Tensor, n_nodes: int, n_edges: int):
        batch_size = z.size(0)
        seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
        positions = torch.arange(self.max_nodes, device=z.device)
        positional_embeddings = self.embeddings(positions).repeat(batch_size, 1, 1)
        seq_input += self.positional_encodings.unsqueeze(0)
        seq_input = torch.cat((seq_input, positional_embeddings), dim=-1)
        seq_input = self.proj_pre_rnn(seq_input)
        # seq_input += positional_embeddings

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
        # pair_emb = self.pairwise_attention(emb_i, emb_j)
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


# class RNNDecoder(nn.Module):
#     def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, max_nodes: int = 50, tau: float = 1.0, hard: bool = True,  batch_size: int=256):
#         super(RNNDecoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.max_nodes = max_nodes
#         self.tau = tau
#         self.hard = hard
#         self.batch_size = batch_size
        
#         self.rnn = nn.GRU(
#             input_size = latent_dim*2, 
#             hidden_size = hidden_dim,
#             num_layers = n_layers, 
#             batch_first = True
#         )
#         self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
#         self.adj_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#         # self.pairwise_attention = PairwiseAttention(hidden_dim=self.hidden_dim, num_heads=4)
#         self.embeddings = nn.Embedding(self.max_nodes, latent_dim)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
#             num_layers=2
#         )
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.positional_encodings = fixed_positional_encoding(max_nodes, latent_dim, device=device)
        
#     def forward(self, z: torch.Tensor, n_nodes: int, n_edges: int):
#         batch_size = z.size(0)
#         seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
#         positions = torch.arange(self.max_nodes, device=z.device)
#         positional_embeddings = self.embeddings(positions).repeat(batch_size, 1, 1)

#         seq_input = torch.cat((seq_input, positional_embeddings), dim=-1)
#         # seq_input += positional_embeddings
#         # seq_input += self.positional_encodings.unsqueeze(0)

#         mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0).expand(batch_size, self.max_nodes) < n_nodes.unsqueeze(1)
#         seq_input[~mask] = 0.0
#         packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_input, n_nodes.cpu(), batch_first=True, enforce_sorted=False)
#         rnn_out, _ = self.rnn(packed_input)
#         rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=self.max_nodes) 
#         node_emb = self.transformer(rnn_out)
#         idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1, device=z.device)
#         emb_i = node_emb[:, idx[0], :]
#         emb_j = node_emb[:, idx[1], :]
#         pair_emb = torch.cat([emb_i, emb_j], dim=-1)
#         logits = self.adj_mlp(pair_emb)
#         adjacency_values = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)[..., 0]
#         adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
#         adj[:, idx[0], idx[1]] = adjacency_values
#         adj = adj + torch.transpose(adj, 1, 2)
#         indices = torch.arange(self.max_nodes, device=z.device).unsqueeze(0)  
#         mask = indices < n_nodes.unsqueeze(1) 
#         mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
#         adj = adj * mask2d.float()
#         return adj
    
# class RNNDecoder(nn.Module):
#     def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, max_nodes: int = 50, tau: float = 1.0, hard: bool = True,  batch_size: int=256):
#         super(RNNDecoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.max_nodes = max_nodes
#         self.tau = tau
#         self.hard = hard
#         self.batch_size = batch_size
        
#         self.rnn = nn.GRU(
#             input_size = latent_dim*2, 
#             hidden_size = hidden_dim,
#             num_layers = n_layers, 
#             batch_first = True
#         )
#         self.node_proj = nn.Linear(hidden_dim, latent_dim, bias=False)
        
#         self.adj_mlp = nn.Linear(2*latent_dim, 2)
#         # self.pairwise_attention = PairwiseAttention(hidden_dim=self.hidden_dim, num_heads=4)
#         self.embeddings = nn.Embedding(self.max_nodes, latent_dim)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=2*latent_dim, nhead=2),
#             num_layers=2
#         )
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.positional_encodings = fixed_positional_encoding(max_nodes, latent_dim, device=device)
        
#     def forward(self, z: torch.Tensor, n_nodes: int, n_edges: int):
#         batch_size = z.size(0)
#         seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
#         positions = torch.arange(self.max_nodes, device=z.device)
#         positional_embeddings = self.embeddings(positions).repeat(batch_size, 1, 1)

#         seq_input = torch.cat((seq_input, positional_embeddings), dim=-1)
#         # seq_input += positional_embeddings
#         # seq_input += self.positional_encodings.unsqueeze(0)

#         mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0).expand(batch_size, self.max_nodes) < n_nodes.unsqueeze(1)
#         seq_input[~mask] = 0.0
#         packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_input, n_nodes.cpu(), batch_first=True, enforce_sorted=False)
#         rnn_out, _ = self.rnn(packed_input)
#         rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=self.max_nodes) 
#         idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1, device=z.device)
#         node_emb = self.node_proj(rnn_out)
#         emb_i = node_emb[:, idx[0], :]
#         emb_j = node_emb[:, idx[1], :]
#         pair_emb = torch.cat([emb_i, emb_j], dim=-1)
#         pair_emb = self.transformer(pair_emb)
#         logits = self.adj_mlp(pair_emb)
#         adjacency_values = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)[..., 0]
#         adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
#         adj[:, idx[0], idx[1]] = adjacency_values
#         adj = adj + torch.transpose(adj, 1, 2)
#         indices = torch.arange(self.max_nodes, device=z.device).unsqueeze(0)  
#         mask = indices < n_nodes.unsqueeze(1) 
#         mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
#         adj = adj * mask2d.float()
#         return adj

# class RNNDecoderE(nn.Module):
#     def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, tau: float = 1.0, hard: bool = True, max_nodes: int = 50):
#         super(RNNDecoderE, self).__init__()
#         self.latent_dim = latent_dim
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.max_nodes = max_nodes
#         self.tau = tau
#         self.hard = hard
        
#         self.rnn = nn.GRU(
#             input_size = latent_dim, 
#             hidden_size = hidden_dim,
#             num_layers = n_layers, 
#             batch_first = True
#         )
#         self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
#         self.adj_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#         self.embeddings = nn.Embedding(self.max_nodes, latent_dim)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.positional_encodings = fixed_positional_encoding(max_nodes, latent_dim, device=device)
        
#     def forward(self, z: torch.Tensor, n_nodes: int, n_edges: int):
#         batch_size = z.size(0)
#         seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
#         positions = torch.arange(self.max_nodes, device=seq_input.device)
#         positional_embeddings = self.embeddings(positions)
#         seq_input += positional_embeddings.unsqueeze(0)
#         # seq_input += self.positional_encodings.unsqueeze(0)

#         mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0).expand(batch_size, self.max_nodes) < n_nodes.unsqueeze(1)
#         seq_input[~mask] = 0.0
#         packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_input, n_nodes.cpu(), batch_first=True, enforce_sorted=False)
#         rnn_out, _ = self.rnn(packed_input)
#         rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=self.max_nodes) 
#         node_emb = self.node_proj(rnn_out)
#         idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1, device=z.device)
#         emb_i = node_emb[:, idx[0], :]
#         emb_j = node_emb[:, idx[1], :]
#         pair_emb = torch.cat([emb_i, emb_j], dim=-1)
#         logits = self.adj_mlp(pair_emb)

#         diff_logits = logits[:,:,0] - logits[:,:,1]
        
#         probas = F.softmax(logits.squeeze(-1))
#         mean_probas_per_batch = probas.mean(dim=1)
#         adjustment_factors = (n_edges / mean_probas_per_batch).unsqueeze(1)
#         probas_ajustes = (probas * adjustment_factors).unsqueeze(2)
#         complement_probs = 1 - probas_ajustes
#         result_probs = torch.log(torch.cat((probas_ajustes, complement_probs), dim=-1) + 1e-4)
        
#         adjacency_values = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)[..., 0]

#         adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
#         adj[:, idx[0], idx[1]] = adjacency_values
#         adj = adj + torch.transpose(adj, 1, 2)
#         indices = torch.arange(self.max_nodes, device=z.device).unsqueeze(0)  
#         mask = indices < n_nodes.unsqueeze(1) 
#         mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
#         adj = adj * mask2d.float()
#         return adj
    


# class AttDecoder(nn.Module):
#     def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, max_nodes: int = 50, tau: float = 1.0, hard: bool = True,  batch_size: int=256):
#         super(AttDecoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.max_nodes = max_nodes
#         self.tau = tau
#         self.hard = hard
#         self.batch_size = batch_size
        
#         self.rnn = nn.GRU(
#             input_size = latent_dim*2, 
#             hidden_size = hidden_dim,
#             num_layers = n_layers, 
#             batch_first = True
#         )
#         self.node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
#         self.adj_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#         self.embeddings = nn.Embedding(self.max_nodes, latent_dim)
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.positional_encodings = fixed_positional_encoding(max_nodes, latent_dim, device=device)
        
#     def forward(self, z: torch.Tensor, n_nodes: int, n_edges: int):
#         batch_size = z.size(0)
#         seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
#         positions = torch.arange(self.max_nodes, device=z.device)
#         positional_embeddings = self.embeddings(positions).repeat(batch_size, 1, 1)
#         seq_input = torch.cat((seq_input, positional_embeddings), dim=-1)
#         # seq_input += self.positional_embeddings
#         # seq_input += self.positional_encodings.unsqueeze(0)

#         mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0).expand(batch_size, self.max_nodes) < n_nodes.unsqueeze(1)
#         seq_input[~mask] = 0.0
#         packed_input = torch.nn.utils.rnn.pack_padded_sequence(seq_input, n_nodes.cpu(), batch_first=True, enforce_sorted=False)
#         rnn_out, _ = self.rnn(packed_input)
#         rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=self.max_nodes) 
#         node_emb = self.node_proj(rnn_out)
#         idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1, device=z.device)
#         emb_i = node_emb[:, idx[0], :]
#         emb_j = node_emb[:, idx[1], :]
#         pair_emb = torch.cat([emb_i, emb_j], dim=-1)
#         logits = self.adj_mlp(pair_emb)
#         adjacency_values = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)[..., 0]
#         adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
#         adj[:, idx[0], idx[1]] = adjacency_values
#         adj = adj + torch.transpose(adj, 1, 2)
#         indices = torch.arange(self.max_nodes, device=z.device).unsqueeze(0)  
#         mask = indices < n_nodes.unsqueeze(1) 
#         mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
#         adj = adj * mask2d.float()
#         return adj

        # adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, 2, device=z.device)
        # adj[:, idx[0], idx[1]] = logits
        # indices = torch.arange(self.max_nodes, device=z.device).unsqueeze(0) 
        # mask = indices < n_nodes.unsqueeze(1)
        # mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
        # adj = adj * mask2d.float().unsqueeze(-1)
        # adj = F.gumbel_softmax(adj, tau=self.tau, hard=self.hard, dim=-1)[..., 0]
        # adj = adj + torch.transpose(adj, 1, 2)


# class AttentionDecoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, max_nodes, tau: float = 1.0, hard: bool = True):
#         super(AttentionDecoder, self).__init__()
#         self.max_nodes = max_nodes
#         self.embeddings = nn.Embedding(self.max_nodes, latent_dim)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=latent_dim*2, nhead=4),
#             num_layers=2
#         )
#         self.adj_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#         self.tau = tau
#         self.hard = hard

#     def forward(self, z, n_nodes, n_edges):
#         batch_size = z.size(0)
#         seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
#         positions = torch.arange(self.max_nodes, device=z.device)
#         positional_embeddings = self.embeddings(positions).repeat(batch_size, 1, 1)
#         seq_input = torch.cat((seq_input, positional_embeddings), dim=-1)
#         node_emb = self.transformer(seq_input)
#         mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0).expand(batch_size, self.max_nodes) < n_nodes.unsqueeze(1)
#         node_emb[~mask] = 0.0
#         idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1, device=z.device)
#         emb_i = node_emb[:, idx[0], :]
#         emb_j = node_emb[:, idx[1], :]
#         pair_emb = torch.cat([emb_i, emb_j], dim=-1)
#         logits = self.adj_mlp(pair_emb)
#         adjacency_values = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)[..., 0]
#         adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
#         adj[:, idx[0], idx[1]] = adjacency_values
#         adj = adj + torch.transpose(adj, 1, 2)
#         indices = torch.arange(self.max_nodes, device=z.device).unsqueeze(0)  
#         mask = indices < n_nodes.unsqueeze(1) 
#         mask2d = mask.unsqueeze(2) & mask.unsqueeze(1)
#         adj = adj * mask2d.float()
#         return adj


class TransformerDecoderModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, max_nodes: int = 50, tau: float = 1.0, hard: bool = True, batch_size: int = 256):
        super(TransformerDecoderModel, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_nodes = max_nodes
        self.tau = tau
        self.hard = hard
        self.batch_size = batch_size

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, 
            nhead=2, 
            dim_feedforward=hidden_dim // 2  
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.proj_pre_rnn = nn.Linear(2 * latent_dim, latent_dim)
        self.adj_mlp = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.embeddings = nn.Embedding(self.max_nodes, latent_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encodings = fixed_positional_encoding(max_nodes, latent_dim, device=device)


    def forward(self, z: torch.Tensor, n_nodes: int, n_edges: int):
        batch_size = z.size(0)
        seq_input = z.unsqueeze(1).repeat(1, self.max_nodes, 1)
        positions = torch.arange(self.max_nodes, device=z.device)
        positional_embeddings = self.embeddings(positions).repeat(batch_size, 1, 1)
        seq_input += self.positional_encodings.unsqueeze(0)
        seq_input = torch.cat((seq_input, positional_embeddings), dim=-1)
        seq_input = self.proj_pre_rnn(seq_input)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_nodes).to(z.device)
        src_padding_mask = torch.arange(self.max_nodes, device=z.device).unsqueeze(0).expand(batch_size, self.max_nodes) >= n_nodes.unsqueeze(1)
        
        memory = seq_input  
        tgt = seq_input.permute(1, 0, 2)  # (seq_len, batch_size, latent_dim)
        transformer_out = self.transformer_decoder(tgt, memory.permute(1, 0, 2), tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
        node_emb = transformer_out.permute(1, 0, 2)  # Back to (batch_size, seq_len, latent_dim)
        
        # Node embeddings
        # node_emb = self.node_proj(transformer_out)
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