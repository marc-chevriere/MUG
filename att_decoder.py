import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionDecoderGlobal(nn.Module):
    def __init__(self, latent_dim, max_nodes, n_heads=3):
        super(GraphAttentionDecoderGlobal, self).__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim

        # Projection du vecteur latent pour initialiser les nœuds
        self.z_proj = nn.Linear(latent_dim, latent_dim)

        # Embeddings positionnels pour les nœuds
        self.node_pos = nn.Parameter(torch.randn(max_nodes, latent_dim))

        # Calcul d'une dimension d'attention compatible
        if latent_dim % n_heads != 0:
            att_dim = n_heads * ((latent_dim // n_heads) + 1)
        else:
            att_dim = latent_dim
        self.att_dim = att_dim

        # Projection pour adapter les dimensions avant l'attention
        self.proj_att = nn.Linear(latent_dim, att_dim) if latent_dim != att_dim else nn.Identity()

        # Couche d'attention multi-tête
        self.attention = nn.MultiheadAttention(embed_dim=att_dim, num_heads=n_heads, batch_first=True)

        # Module bilinéaire pour prédire le score d'arête
        self.edge_bilinear = nn.Bilinear(latent_dim, latent_dim, 1)

    def forward(self, z):
        batch_size = z.size(0)
        z_proj = self.z_proj(z).unsqueeze(1)
        node_features = z_proj.repeat(1, self.max_nodes, 1) + self.node_pos.unsqueeze(0)
        
        # Projection des représentations en dimension att_dim
        node_features_att = self.proj_att(node_features)
        
        # Passage par l'attention
        node_features_att, _ = self.attention(node_features_att, node_features_att, node_features_att)
        
        # Vous pouvez choisir soit d'utiliser node_features (de dimension latent_dim) soit
        # node_features_att (de dimension att_dim) pour la suite.
        # Ici, nous gardons node_features pour le calcul bilinéaire :
        node_i = node_features.unsqueeze(2).expand(batch_size, self.max_nodes, self.max_nodes, self.latent_dim)
        node_j = node_features.unsqueeze(1).expand(batch_size, self.max_nodes, self.max_nodes, self.latent_dim)
        edge_logits = self.edge_bilinear(node_i, node_j).squeeze(-1)
        
        edge_logits_stacked = torch.stack([torch.zeros_like(edge_logits), edge_logits], dim=-1)
        edge_prob = F.gumbel_softmax(edge_logits_stacked, tau=1, hard=True, dim=-1)[..., 1]

        edge_prob = torch.triu(edge_prob, diagonal=1)
        adj = edge_prob + edge_prob.transpose(1, 2)
        return adj