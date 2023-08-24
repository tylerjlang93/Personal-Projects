"""
Transformer model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=4, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.embeddingL = nn.Embedding(self.input_size,self.hidden_dim)      #initialize word embedding layer
        self.posembeddingL  = nn.Embedding(self.max_length,self.hidden_dim)      #initialize positional embedding layer

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################


        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################

        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)

        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        # Head #3
        self.k3 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v3 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q3 = nn.Linear(self.hidden_dim, self.dim_q)

        # Head #4
        self.k4 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v4 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q4 = nn.Linear(self.hidden_dim, self.dim_q)


        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)


        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        #
        # Don't forget the layer normalization.                                      #
        ##############################################################################

        self.lin1 = nn.Linear(self.hidden_dim,self.dim_feedforward)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.dim_feedforward,self.hidden_dim)
        self.norm_final = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################


        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################

        self.l_final = nn.Linear(self.hidden_dim,self.output_size)
        self.CE_Loss = nn.CrossEntropyLoss()


        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################


    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################

        embedded = self.embed(inputs)
        attent = self.multi_head_attention(embedded)
        forward = self.feedforward_layer(attent)
        outputs = self.final_layer(forward)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs


    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        embeddings = None
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################

        word_emb = self.embeddingL(inputs)
        pos = torch.arange(0, self.max_length).unsqueeze(0).to(self.device)
        pos_emb = self.posembeddingL(pos)
        embeddings = torch.add(word_emb,pos_emb)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings

    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)

        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """


        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################

        k1 = self.k1(inputs)
        v1 = self.v1(inputs)
        q1 = self.q1(inputs)

        k2 = self.k2(inputs)
        v2 = self.v2(inputs)
        q2 = self.q2(inputs)

        k3 = self.k3(inputs)
        v3 = self.v3(inputs)
        q3 = self.q3(inputs)

        k4 = self.k4(inputs)
        v4 = self.v4(inputs)
        q4 = self.q4(inputs)



        d_k = q1.size()[-1]

        scores1 = self.softmax((torch.matmul(q1, k1.transpose(-1, -2))/np.sqrt(d_k)))
        att1 = torch.matmul(scores1,v1)

        scores2 = self.softmax((torch.matmul(q2, k2.transpose(-1, -2))/np.sqrt(d_k)))
        att2 = torch.matmul(scores2,v2)

        scores3 = self.softmax((torch.matmul(q3, k3.transpose(-1, -2))/np.sqrt(d_k)))
        att3 = torch.matmul(scores3,v3)
        
        scores4 = self.softmax((torch.matmul(q4, k4.transpose(-1, -2))/np.sqrt(d_k)))
        att4 = torch.matmul(scores4,v4)

        att = torch.cat((att1,att2,att3,att4),2)

        project = self.attention_head_projection(att)
        outputs = self.norm_mh(project+inputs)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs


    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################

        l1_pass = self.lin1(inputs)
        relu_pass = self.relu(l1_pass)
        l2_pass = self.lin2(relu_pass)
        outputs = self.norm_final(l2_pass+inputs)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs


    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """

        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = self.l_final(inputs)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
