"""
2d Convolution Module.  (c) 2021 Georgia Tech

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


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        x_shape = x.shape # images, channels, h, w
        w_shape = self.weight.shape  # num filters, channels, h, w
        w = self.weight
        b = self.bias

        x_pad = np.pad(x,pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')

        h_in = x_pad.shape[2]
        w_in = x_pad.shape[3]

        h_ker = w.shape[2]
        w_ker = w.shape[3]
        n_filters = w.shape[0]

        h_out = (h_in - h_ker)//self.stride + 1
        w_out = (w_in - w_ker)//self.stride + 1

        out = np.zeros((x_pad.shape[0],self.out_channels,h_out,w_out))

        for i in range(x_pad.shape[0]):
            for j in range(n_filters):
                for k in range(h_out):
                    for l in range(w_out):
                        h_initial_id = k*self.stride
                        h_stop_id = h_initial_id + h_ker
                        w_initial_id = l*self.stride
                        w_stop_id = w_initial_id + w_ker
                        temp_x = x_pad[i,np.newaxis,:,h_initial_id:h_stop_id,w_initial_id:w_stop_id] # need to create new axis for "filter number"
                        # temp_x is [photo i, all channels,width,heighth]
                        out[i,j,k,l] = np.sum(temp_x*w[np.newaxis,j,:,:,:]) + b[j] # need to create new axis for "photo number"
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        ################################
        db = np.sum(dout,axis=(0,2,3)) #Giving the right dimensions for b
        ################################

        n_photos = x.shape[0]
        c_in = x.shape[1]
        h_in = x.shape[2]
        w_in = x.shape[3]

        h_out = dout.shape[2]
        w_out = dout.shape[3]

        n_filters = self.weight.shape[0]
        h_ker = self.weight.shape[2]
        w_ker = self.weight.shape[3]

        x_pad = np.pad(x,pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')

        ###############################################################

        dx = np.zeros(x_pad.shape)

        for ho in range(h_out):
            for wo in range(w_out):
                h_start = ho
                h_end = h_start + h_ker
                w_start = wo
                w_end = w_start + w_ker
                trial = (self.weight[np.newaxis,:, :, :, :]*dout[:,:,np.newaxis, ho:ho+1, wo:wo+1])
                channel_sum = np.sum(trial,axis=1)
                dx[:,:,h_start:h_end, w_start:w_end] += channel_sum

        dx = dx[:,:,1:dx.shape[2]-1,1:dx.shape[3]-1]

        ###############################################################

        dw = np.zeros(self.weight.shape) #Will have to take into account padded x, due to order of operations in forward.

        for f in range(n_filters):
            for c in range(c_in):
                for hk in range(h_ker):
                    for wk in range(w_ker):
                        final_h_forslicing = hk + h_out*self.stride
                        affected_cells_height = slice(hk,final_h_forslicing,self.stride) # Need to skip over strided cells without hitting error
                        final_w_forslicing = wk + w_out* self.stride
                        affected_cells_width = slice(wk,final_w_forslicing,self.stride)
                        affected_x = x_pad[:,  c, affected_cells_height, affected_cells_width]
                        #print(affected_x.shape, dout[:,f,:,:].shape)
                        temp = affected_x*dout[:, f, :, :]
                        dw[f, c, hk ,wk] = np.sum(temp)

        ##################################################################





        self.dx = dx
        self.dw = dw
        self.db = db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
