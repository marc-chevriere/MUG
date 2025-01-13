import torch
import torch.nn as nn

class GraphGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_nodes):
        super(GraphGenerator, self).__init__()
        self.n_nodes = n_nodes
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_nodes * n_nodes)
        )

    def forward(self, z):
        adj = self.mlp(z)
        adj = adj.view(-1, self.n_nodes, self.n_nodes)  # Reshape to adjacency matrix
        adj = torch.sigmoid(adj)  # Normalize to [0, 1]
        adj = (adj + adj.transpose(-1, -2)) / 2  # Make it symmetric
        return adj

class GraphDiscriminator(nn.Module):
    def __init__(self, n_nodes, hidden_dim):
        super(GraphDiscriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_nodes * n_nodes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, adj):
        adj = adj.view(adj.size(0), -1)  # Flatten adjacency matrix
        validity = self.mlp(adj)
        return validity
