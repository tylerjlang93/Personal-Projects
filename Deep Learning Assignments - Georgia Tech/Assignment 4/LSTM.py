"""
LSTM model.  (c) 2021 Georgia Tech

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
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.U_inputgate = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_inputgate = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_inputgate = nn.Parameter(torch.Tensor(self.hidden_size))
        # f_t: the forget gate
        self.U_forget = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_forget = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_forget = nn.Parameter(torch.Tensor(self.hidden_size))
        # g_t: the cell gate
        self.U_tilded = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_tilded = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_tilded = nn.Parameter(torch.Tensor(self.hidden_size))
        # o_t: the output gate
        self.U_output = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_output = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias_output = nn.Parameter(torch.Tensor(self.hidden_size))
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################

        if init_states is None:
            h_size = self.hidden_size
            c_t = torch.zeros(x.size()[0], h_size).to(x.device)
            h_t = torch.zeros(x.size()[0], h_size).to(x.device)


        else:
            h_t = init_states[0]
            c_t = init_states[1]

        for i in range(x.size()[1]):
            input = x[:, i, :]

            input_gate_final = torch.sigmoid(input@self.U_inputgate + h_t@self.V_inputgate + self.bias_inputgate)
            forget_gate_final = torch.sigmoid(input@self.U_forget + h_t@self.V_forget + self.bias_forget)
            tilded_cell_gate_final = torch.tanh(input@self.U_tilded + h_t@self.V_tilded + self.bias_tilded)
            output_gate_final = torch.sigmoid(input@self.U_output + h_t@self.V_output + self.bias_output)

            c_t = input_gate_final * tilded_cell_gate_final + forget_gate_final * c_t
            h_t = output_gate_final * torch.tanh(c_t)



        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
