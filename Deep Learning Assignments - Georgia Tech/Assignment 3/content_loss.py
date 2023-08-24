import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################


        depth = content_current.size()[1]
        height = content_current.size()[2]
        width = content_current.size()[3]

        dims = height*width

        fig_curr = content_current.view(depth, dims)
        fig_orig = content_original.view(depth, dims)
        diff = (fig_curr - fig_orig)
        sq_diff = diff**2
        loss_sum = torch.sum(sq_diff)

        l = loss_sum*content_weight

        return l


        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
