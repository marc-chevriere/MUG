import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GATConv, GlobalAttention, TopKPooling
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter_mean


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class condEncoder(nn.Module):

    def __init__(self, latent_dim, cond_dim):
        super(condEncoder, self).__init__()
        # self.fc = nn.Linear(cond_dim, latent_dim)
        hidden_dim = (latent_dim + cond_dim)//2
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.mlp(x)
    

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


class HybridGINGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, heads=3):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # Première couche GIN
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )))

        # Alternance GIN et GAT
        for i in range(1, n_layers):
            if i % 2 == 0:
                self.convs.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2)
                )))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))

        # Normalisation batch et couche fully connected
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        # Passer les données dans les couches
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        # Pooling global pour obtenir la représentation du graphe
        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


class GINwAtt(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(0.2)
        )))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.LeakyReLU(0.2)
            ))) 

        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = self.att_pool(x, batch)
        out = self.bn(out)
        out = self.fc(out)
        return out
    

class GINWithTopKPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, pool_ratio=0.8):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            )
        ))
        self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))

        for _ in range(n_layers-1):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2)
                )
            ))
            self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))

        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch=batch)
            
        # Pool global final
        out = global_mean_pool(x, batch)
        out = self.fc(out)
        return out


class GINSkipConnections(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # Couches GIN
        self.convs = nn.ModuleList()
        # Première couche
        self.convs.append(
            GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            ))
        )
        # Couches suivantes
        for _ in range(n_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2)
                ))
            )
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # On applique les couches GIN avec skip
        for conv in self.convs:
            x_res = x  # on garde l'ancienne valeur pour la connexion résiduelle
            x = conv(x, edge_index)
            # Option : petite activation avant de faire le skip
            x = F.leaky_relu(x, negative_slope=0.2)
            # Ajout de la connexion résiduelle
            x = x + x_res
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = global_add_pool(x, batch)
        out = self.bn(out)
        out = self.fc(out)
        return out

class GINVirtualNode(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        
        # 1) Projection initiale des features des nœuds
        self.node_proj = nn.Linear(input_dim, hidden_dim)

        # 2) Couches GIN
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            ))
        )
        for _ in range(n_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2)
                ))
            )

        # 3) Embedding du nœud virtuel (un pour tout le batch, répliqué par graphe)
        self.virtualnode_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.virtualnode_embedding)

        # 4) MLP pour la mise à jour du nœud virtuel après chaque couche
        self.mlp_virtualnode = nn.ModuleList()
        for _ in range(n_layers):
            self.mlp_virtualnode.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )

        # 5) BatchNorm + Linear pour projection finale
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        """
        data.x : [total_nodes, input_dim]
        data.edge_index : [2, total_edges]
        data.batch : [total_nodes] (batch indice pour chaque noeud)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1) Projection initiale des features des nœuds
        x = self.node_proj(x)

        # 2) On duplique le vecteur virtuel pour chaque graphe du batch
        batch_size = batch.max().item() + 1
        vn_batched = self.virtualnode_embedding.repeat(batch_size, 1)  # [batch_size, hidden_dim]

        for layer_idx, conv in enumerate(self.convs):
            # 3) On “ajoute” l'info du virtual node à chaque nœud
            x = x + vn_batched[batch]

            # 4) Passage dans la couche GIN
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

            # 5) Mise à jour du virtual node
            vn_update = scatter_mean(x, batch, dim=0)  # [batch_size, hidden_dim]
            vn_batched = vn_batched + self.mlp_virtualnode[layer_idx](vn_update)

        # 6) Pooling final (on peut choisir sum, mean, etc.)
        out = global_add_pool(x, batch)
        out = self.bn(out)
        out = self.fc(out)
        return out
    

# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, lambda_contrastive):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GINVirtualNode(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim*2, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.cond_encoder = condEncoder(cond_dim=7,latent_dim=latent_dim)
        self.lambda_contrastive = lambda_contrastive

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_mu(self, x_sample, stat):
        cond = self.cond_encoder(stat)
        x_sample = torch.cat((x_sample, cond), dim=1)
        adj = self.decoder(x_sample)
        return adj
    
    def contrastive_loss(self, z, cond, temperature=0.1):
        z = F.normalize(z, p=2, dim=1)
        cond = F.normalize(cond, p=2, dim=1)
        cond_similarity = torch.mm(cond, cond.T)
        z_similarity = torch.mm(z, z.T) / temperature
        labels = torch.softmax(cond_similarity, dim=1)
        logits = torch.exp(z_similarity) / torch.sum(torch.exp(z_similarity), dim=1, keepdim=True)
        loss = -torch.sum(labels * torch.log(logits + 1e-9)) / z.size(0)
        return loss

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        cond = self.cond_encoder(data.stats)
        x_g_cat = torch.cat((x_g, cond), dim=1)
        adj = self.decoder(x_g_cat)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        contrastive_loss = self.contrastive_loss(x_g, cond)
        loss = (1-self.lambda_contrastive) * (recon + beta * kld) + self.lambda_contrastive * contrastive_loss

        return loss, recon, kld
