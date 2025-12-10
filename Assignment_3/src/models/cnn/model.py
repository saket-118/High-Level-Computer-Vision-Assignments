import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.in_channels   = input_size          
        self.hidden_layers = hidden_layers[:5]   
        self.num_classes   = num_classes
        self.activation    = activation
        self.norm_layer    = norm_layer
        self.drop_prob     = drop_prob

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        in_ch = self.in_channels

        for out_ch in self.hidden_layers:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))

            if self.norm_layer is not None:          
                layers.append(self.norm_layer(out_ch))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layers.append(self.activation())
            else:                                    
                layers.append(self.activation())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            if self.drop_prob > 0.0:
                layers.append(nn.Dropout(self.drop_prob))
            in_ch = out_ch


        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))

        self.model = nn.Sequential(*layers)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        first_conv = next((m for m in self.model if isinstance(m, nn.Conv2d)),
                          None)
        if first_conv is None:
            print("No Conv layer found!");  return

        # weights: [out_ch, 3, 3, 3]  →  [out_ch, 3, 3, 3]
        with torch.no_grad():
            w = first_conv.weight.cpu().numpy()

        # normalise each filter to [0,1]
        w = self._normalize(w)

        k = w.shape[-1]           # 3
        n_filters = w.shape[0]    # 128
        cols = int(np.ceil(np.sqrt(n_filters)))
        rows = int(np.ceil(n_filters / cols))

        grid = np.zeros((rows*k, cols*k, 3))
        for idx in range(n_filters):
            r, c = divmod(idx, cols)
            filt = w[idx].transpose(1, 2, 0)       # CHW → HWC
            grid[r*k:(r+1)*k, c*k:(c+1)*k, :] = filt

        plt.figure(figsize=(cols, rows))
        plt.imshow(grid)
        plt.axis('off')
        plt.title("First-layer Conv filters")
        plt.show()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return self.model(x)
