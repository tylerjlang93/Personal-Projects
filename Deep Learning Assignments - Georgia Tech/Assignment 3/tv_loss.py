import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total varation loss function                               #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################

        width_forward = img[:,:,:,1:]
        width_backward = img[:,:,:,:-1]
        height_forward = img[:,:,1:,:]
        height_backward = img[:,:,:-1,:]
        w_diff = (width_forward - width_backward)
        h_diff = (height_forward - height_backward)
        sq_w_diff = w_diff**2
        sq_h_diff = h_diff**2
        width_var = torch.sum(sq_w_diff)
        height_var = torch.sum(sq_h_diff)
        total_var = (width_var + height_var)
        l = tv_weight * total_var

        return l


        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
