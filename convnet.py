
"""The input dimensions (1x32x32)
 The filter size is (4x2x2) where 4 is the number of filters
 The output size is (10x1)
 architecture is (input->conv->relu->fc->output)"""


from layers import *
import numpy as np
class ConvNet(object):


    def __init__(self, input_dim=(1, 32, 32), num_filters=4, filter_size=2,
                  num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}
        C, H, W = input_dim

        # Conv layer
        # The parameters of the conv is of size (F,C,HH,WW) with
        # F give the nb of filters, C,HH,WW characterize the size of
        # each filter
        # Input size : (N,C,H,W)
        # Output size : (N,F,Hc,Wc)
        F = num_filters
        filter_height = filter_size
        filter_width = filter_size
        stride_conv = 2  # stride
        Hc = (H  - filter_height) / stride_conv + 1
        Wc = (W  - filter_width) / stride_conv + 1
        W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
        b1 = np.zeros(F)

        C = num_classes
        W2 = weight_scale * np.random.randn(F * Hc * Wc, C)
        b2 = np.zeros(C)
        self.params.update({'W1': W1,'W2': W2, 'b1': b1,'b2': b2})




    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)



        N = X.shape[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        scores = None

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1}

        x = X
        w = W1
        b = b1
        conv_layer, cache_conv_layer = conv_relu_forward(x, w, b, conv_param)
        N, F, Hp, Wp = conv_layer.shape


        x = conv_layer
        w = W2
        b = b2
        scores, cache_scores = affine_forward(x, w, b)
        if y is None:
            return scores

        loss, grads = 0, {}

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        reg_loss += 0.5 * self.reg * np.sum(W2**2)
        loss = data_loss + reg_loss
        grads = {}
            # Backprop into output layer
        dx2, dW2, db2 = affine_backward(dscores, cache_scores)
        dW2 += self.reg * W2



        # Backprop into the conv layer
        dx2 = dx2.reshape(N, F, Hp, Wp)
        dx, dW1, db1 = conv_relu_backward(dx2, cache_conv_layer)
        dW1 += self.reg * W1

        grads.update({'W1': dW1,'b1': db1,'W2': dW2,'b2': db2})
        return loss, grads
