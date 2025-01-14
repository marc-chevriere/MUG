import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset
from numpy import errstate

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from extract_feats import extract_feats, extract_numbers

from transformers import AutoTokenizer, AutoModel


#LM TO TEST :
#bert-base-uncased
#roberta-base
#distilbert-base-uncased
#huawei-noah/TinyBERT_General_4L_312D  #moins gpurmand

#from transformers import T5Tokenizer, T5Model
#tokenizer = T5Tokenizer.from_pretrained("t5-base")
#model = T5Model.from_pretrained("t5-base")

#from transformers import XLNetTokenizer, XLNetModel
#tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
#model = XLNetModel.from_pretrained("xlnet-base-cased")


    # Charger le modèle via une connexion distante
model_name = "huawei-noah/TinyBERT_General_4L_312D"  # Exemple : TinyBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
print(f"Model {model_name} loaded on device: {next(model.parameters()).device}")

def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_' + dataset + '.pt'
        desc_file = './data/' + dataset + '/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)

                # Encode the description using the language model
                inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True).to(device)
                outputs = model(**inputs)
                feats_stats = outputs.last_hidden_state.mean(dim=1).cpu()

                data_lst.append(Data(stats=feats_stats, filename=graph_id))
            fr.close()
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')

    else:
        filename = './data/dataset_' + dataset + '.pt'
        graph_path = './data/' + dataset + '/graph'
        desc_path = './data/' + dataset + '/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            files = [f for f in os.listdir(graph_path)]
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx + 1:]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")

                if extension == "graphml":
                    G = nx.read_graphml(fread)
                    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
                else:
                    G = nx.read_edgelist(fread)

                CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                eigval, eigvecs = np.linalg.eigh(L)
                eigvecs = torch.tensor(eigvecs[:, :spectral_emb_dim], dtype=torch.float)

                edge_index = torch.nonzero(adj).t()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
                x[:, 0] = adj.sum(axis=1) / (n_max_nodes - 1)
                x[:, 1:] = eigvecs[:, :spectral_emb_dim]
                adj = F.pad(adj, [0, n_max_nodes - G.number_of_nodes(), 0, n_max_nodes - G.number_of_nodes()])
                adj = adj.unsqueeze(0)

                # Encode the description using the language model
                with open(fstats, "r") as f:
                    desc = f.read().strip()
                inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True).to(device)
                outputs = model(**inputs)
                feats_stats = outputs.last_hidden_state.mean(dim=1).cpu()

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename=filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst


        

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start