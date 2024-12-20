# 1. Imports
import numpy as np
import random
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# I think this works // Johan
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


# # 2. Hyperparameters
# V = 2000  # Max 10k needed
# K = 20    # Number of topics
# embedding_dim = 300  # Dimension of word embeddings
# batch_size = 32
# epochs = 10
# learning_rate = 1e-3

# # We might want to fetch these from the data instead.
# rho_size = 100     # The dimensionality of the word embeddings
# t_hidden_size = 512  # Hidden layer size for the variational network


# # Fetch data, Johan just vibed here, this should maybe be included in the class hmmm.
# training_set = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# # Return list of raw text documents d
# documents = training_set.data

# # To test if the fetch work run:
# # from pprint import pprint
# # pprint(list(newsgroups_train.target_names))

# # Take the V most frekuent words, remove stop_words
# vectorizer = CountVectorizer(max_features=V, stop_words='english')
# # make this a np.array
# bow_matrix = vectorizer.fit_transform(documents).toarray()  # [num_docs, V]
# # get out the common words
# vocab = vectorizer.get_feature_names_out()  # Vocabulary list
# # convert to ratios in each row (why? normalize = good)
# bow_matrix = normalize(bow_matrix, norm='l1', axis=1)  # Normalize to [0, 1]

# # test
# print(bow_matrix)

class BowDataset(Dataset):
    "Bag of Word data, so far not used anywhere"
    def __init__(self, bow_matrix):
        """
        Args:
            bow_matrix (np.ndarray): Bag-of-words matrix with shape [num_docs, V].
        """
        self.bow_matrix = torch.tensor(bow_matrix, dtype=torch.float32)  # Convert to PyTorch tensors

    def __len__(self):
        """
        Returns the number of documents in the dataset.
        """
        return self.bow_matrix.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the document to retrieve.

        Returns:
            torch.Tensor: The bag-of-words representation of the document.
        """
        return self.bow_matrix[idx]


# # Create a PyTorch Dataset from the bag-of-words matrix
# dataset = BowDataset(bow_matrix)


# This might be better than the get_minibatch, but I haven't tested yet
# Use DataLoader to generate minibatches
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


############ ETM model ########################

class ETM(nn.Module):
    def __init__(self, vocab_size, num_topics, embedding_dim, t_hidden_size, theta_act='relu'):
        super(ETM, self).__init__()
        # Define model and variational parameters
        # super(ETM, self).__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        print(vocab_size, num_topics, t_hidden_size, embedding_dim)
        # Word embedding matrix (rho)
        self.rho = nn.Parameter(torch.randn(vocab_size, embedding_dim))  # [V, D]
        self.rho_shape = np.shape(self.rho)
        self.rho_size = np.size(self.rho)

        # Topic embedding matrix (alpha)
        self.alpha = nn.Parameter(torch.randn(num_topics, embedding_dim))  # [K, D]

        # Initial optimizer
        self.optimizer = nn.ReLU()  # ReLU seemed to work a bit better I think /Ebba

        # Neural network for variational parameters (mu_d, sigma_d)
        self.nn_mu = nn.Sequential(
            nn.Linear(t_hidden_size, num_topics),
            nn.ReLU()
        )
        self.nn_logsigma = nn.Sequential(
            nn.Linear(t_hidden_size, num_topics),
            nn.ReLU()
        )

        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, t_hidden_size),
            self.get_activation('relu'),
            nn.Linear(t_hidden_size, t_hidden_size),
            self.get_activation('relu'),
        )

    def get_activation(self, type):
        if type == 'tanh':
            return nn.Tanh()
        elif type == 'relu':
            return nn.ReLU()
        else:
            return nn.Identity()

    def soft_max(self, x):
        """Pure definition"""
        return np.exp(x) / sum(np.exp(x))

    def get_beta(self):
        """from algorithm 1 in paper"""
        #rho = self.rho()
        rhoT = self.rho.t()
        alpha = self.alpha #check here 
        # TODO: check dims here once all is coded
        # print(alpha.shape)
        # print(rhoT.shape)
        beta = F.softmax(torch.matmul(alpha , rhoT), dim = 0)    # alpha's rows indices k
        return beta

    def get_alpha(self, rho_size, num_topics):
        """Returns a neural net for all topic embedings
        I set (affine) linear on input
        parameters are input size, output size, and to have bias or not
        the ELBO is unbiased so remove bias here so this just becomes a matrix/vector?"""
        alphas = nn.Linear(rho_size, num_topics, bias = False)
        #self.alphas = alphas

    def get_batch(self, docs, splits):
        """I think we can outperform their paper by using K-folds
        input: docs as np.array, indices for splits
        return minibtach as torch tensor
        This is the Bag-of-Words bog from the paper"""
        batch = docs[splits, :]
        batch = torch.from_numpy(batch.toarray()).float()
        return batch

    def NN_prop(current_batch, modified_batch):
        """Propagate forward"""
        loss = 0
        kld_over_theta = 0
        pass

    def get_theta(self, modified_batch):
        q_theta = self.q_theta(modified_batch)
        nn_mu = self.nn_mu(q_theta)
        nn_logsigma = self.nn_logsigma(q_theta)
        dirac = nn_mu + torch.exp(nn_logsigma) * torch.randn_like(nn_mu)
        theta = F.softmax(dirac,dim=1)


        # kl_theta = torch.sum(theta * (torch.log(theta) - torch.log(q_theta)))
        kl_theta = 0.5 * (torch.sum(torch.exp(nn_logsigma)) + torch.sum(nn_mu ** 2) - torch.sum(nn_logsigma) - self.num_topics)  # KL divergence, not sure if this is correct
        return theta, kl_theta

    def forward(self, modified_batch):
        beta  = self.get_beta()

        theta, kl_theta = self.get_theta(modified_batch)

        p_wdn = torch.matmul(theta, beta)  # Might need to be transposed or regularize

        # They compute the reconstruction loss as the negative log-likelihood for the elbo. Not sure what in the paper it is reffering to.

        reconstruction_loss = -torch.sum(p_wdn * modified_batch, dim=1)

        return reconstruction_loss, kl_theta


    def visualize_topics(self, top_n=10):
        # Get the topic-word distribution (beta matrix)
        """ This does not work yet :) // Johan """
        beta = self.get_beta()  # [num_topics, vocab_size]
        
        # Get the top N words for each topic
        for i, topic in enumerate(beta):
            top_words_idx = torch.topk(topic, top_n).indices  # Get indices of top N words
            top_words = [self.vocab[idx] for idx in top_words_idx]
            print(f"Topic {i}: {', '.join(top_words)}")

    def one_epoch(self, epoch, training_set, num_docs, batch_size,
                  optimizer):
        batch_i = torch.split(torch.arange(num_docs), batch_size)
        all_loss_re = []
        all_loss_kl_theta = []
        for batch_partition, enumera in enumerate(batch_i):
            current_batch = self.get_batch(training_set, splits = enumera)
            optimizer.zero_grad()   #reset SGD gradients
            self.optimizer.zero_grad()
            # I do not undrstand why but they do this:
            modified_batch = current_batch
            reconstruction_loss, kl_theta = self.forward(modified_batch)
            elbo = reconstruction_loss.mean() + kl_theta.mean()
        
            elbo.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            optimizer.step()

            all_loss_re.append(reconstruction_loss.mean().item())
            all_loss_kl_theta.append(kl_theta.mean().item())

        total_loss = (np.sum(all_loss_re) + np.sum(all_loss_kl_theta))/len(all_loss_re)
        print(total_loss, all_loss_re, all_loss_kl_theta)
        return total_loss


