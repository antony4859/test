from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import data
import scipy.io

from torch import nn, optim
from torch.nn import functional as F
from pathlib import Path
from gensim.models.fasttext import FastText as FT_gensim
# import tracemalloc
from ETM import ETM
from fetch_data import fetch_data

SEED = 1

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print('\n')
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
    ## get data
# 1. vocabulary
bow_tr, bow_ts, bow_ts_h1, bow_ts_h2, bow_va, vocab, n_docs_tr, n_docs_ts, n_docs_ts_h1, n_docs_ts_h2, n_docs_va = fetch_data()
vocab_size = len(vocab)
num_topics =  512
batch_size = 20
epochs = 10

# 1. training data
num_docs_train = bow_tr.shape[0]

# 2. dev set

num_docs_valid = bow_va.shape[0]

# 3. test data

num_docs_test = bow_ts_h1.shape[0] + bow_ts_h2.shape[0]

num_docs_test_1 = bow_ts_h1.shape[0]
num_docs_test_2 = bow_ts_h2.shape[0]

# embeddings = data.read_embedding_matrix(vocab, device, load_trainned=False)
# embeddings_dim = embeddings.size()
embeddings_dim = 300
t_hidden_size = 512

model = ETM(vocab_size, num_topics, t_hidden_size, embeddings_dim)

print('model: {}'.format(model))

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

best_epoch = 0
best_val_ppl = 1e9
all_val_ppls = []
print('\n')
print('Visualizing model quality before training...', epochs)
#model.visualize(args, vocabulary = vocab)
print('\n')

loss_history = []
for epoch in range(0, epochs):
    print("I am training for epoch", epoch)
    loss = model.one_epoch(epoch, bow_tr, n_docs_tr, batch_size, optimizer)
    loss_history.append(loss)

#model.visualize_topics(top_n=10)

# Plot loss
plt.plot(range(epochs), loss_history, label='Total Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

beta = model.get_beta()
x = torch.argsort(beta, dim=1, descending=True)

for i in range(num_topics):
    print("topic: " + str(i) + str(" "))
    for j in range(2):
        print(str(vocab[x[i,j]]) + "  "  + str(beta[i, j].item()) + " " , end=" ")

    print()
