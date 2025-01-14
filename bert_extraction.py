##test feature extraction from BERT

from transformers import AutoTokenizer, AutoModel
import torch

import os
from tqdm import tqdm
import random
import re

random.seed(32)


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats

# Initialiser le tokenizer et le modèle BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Exemple de description textuelle
desc = "This graph comprises 50 nodes and 589 edges. The average degree is equal to 23.56 and there are 3702 triangles in the graph. The global clustering coefficient and the graph's maximum k-core are 0.6226034308779012 and 18 respectively. The graph consists of 3 communities."

# Tokenisation
inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True)

# Encodage avec BERT
outputs = model(**inputs)

# Extraction des caractéristiques
feats_stats_bert = outputs.last_hidden_state.mean(dim=1)

#print(feats_stats_bert)

#extraction numérique brute

feats_stats = extract_numbers(desc)
feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

#print(feats_stats)

import torch

# Vérifier si CUDA est disponible
print("CUDA available:", torch.cuda.is_available())

# Afficher le nom du GPU
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))