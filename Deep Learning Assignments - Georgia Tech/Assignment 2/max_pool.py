"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        kernel = self.kernel_size
        stride = self.stride
        maxxes = []

        final_size = ((x.shape[0]),(x.shape[1]), ((x.shape[2]-kernel)//stride)+1, ((x.shape[2]-kernel)//stride)+1)
        H_out = final_size[2]
        W_out = final_size[3]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        rk = k*stride
                        rl = l*stride
                        if (rk+kernel <= x.shape[2]) & (rl+kernel <= x.shape[3]):
                            block = x[i,j,rk:rk+kernel,rl:rl+kernel]
                            mx = np.max(block)
                            maxxes.append(mx)
                        else:
                            break

        out = np.array(maxxes).reshape(final_size)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        kernel = self.kernel_size
        stride = self.stride
        max_args = []
        counter = 0
        #final_size = ((x.shape[0]),(x.shape[1]), ((x.shape[2]-kernel)//stride)+1, ((x.shape[2]-kernel)//stride)+1)
        sparse = np.zeros((x.shape))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        rk = k*stride
                        rl = l*stride
                        if (rk+kernel <= x.shape[2]) & (rl+kernel <= x.shape[3]):
                            block = x[i,j,rk:rk+kernel,rl:rl+kernel]
                            mx = np.argwhere(block==np.max(block))
                            counter += 1
                            final_arg = (i,j,mx[0,0]+rk,mx[0,1]+rl)
                            sparse[final_arg] = dout[i,j,k,l]
                            max_args.append(final_arg)
                        else:
                            break

        self.dx = sparse

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
