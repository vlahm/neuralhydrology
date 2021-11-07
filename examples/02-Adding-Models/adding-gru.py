#!/usr/bin/env python
# coding: utf-8

# # Adding a New Model
# 
# This tutorial shows how you can add a new model to the `neuralhydrology` modelzoo.
# As an example, we'll implement a GRU.
import inspect
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.template import TemplateModel
from neuralhydrology.utils.config import Config


# ## Template
# 
# Every model has its own file in `neuralhydrology.modelzoo` and follows a common template that you can find [here](https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/modelzoo/template.py).
# 
# The most important points about these templates are:
# 
# - All models inherit from the `BaseModel` that's implemented in `neuralhydrology.modelzoo.basemodel`.
# - All models' constructors take just one argument, an instance of the configuration class (`Config`). The constructor initializes the model and its components.
# - Finally, each model implements its own logic in the `forward` method. This is where the actual magic happens: The forward method takes the input data during training and evaluation and uses it to generate a prediction.
# 
# In the following steps, we'll go over the constructor and the forward method in more detail.

# ## Adding a GRU Model
# 
# So, let's follow that template and add a [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) model.
# Fortunately, there already exists a [GRU implementation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) in the PyTorch libary, so we can wrap our code around that existing model.
# This way, we can be pretty sure to get a correct and reasonably fast implementation without much effort.
# 
# For the sake of brevity, we'll omit the docstrings in this example. If you actually implement a model for production use, you should always write the documentation right within your code.
# 
# ### GRU Components
# 
# Every model's constructor receives a single argument: an instance of the run configuration.
# Based on this config, we'll construct the GRU.
# 
# Like most our models, the GRU will consist of three components: 
# 
# - An optional input layer that acts as an embedding network for static or dynamic features. If used, the features will be passed through a fully-connected network before we pass them to the actual GRU. If no embedding is specified, this layer will do nothing.
# - The "body" that represents the actual GRU cell.
# - The "head" that acts as a final output layer.
# 
# To maintain a modular architecture, the input and head layers should not be implemented inside the model. Instead, we should use the `InputLayer` in `neuralhydrology.modelzoo.inputlayer` and the `get_head` function in `neuralhydrology.modelzoo.head` which will automatically construct layers that fit to the run configuration.

# In[2]:


class GRU(BaseModel):

    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'gru', 'head']

    def __init__(self, cfg: Config):

        super(GRU, self).__init__(cfg=cfg)

        # retrieve the input layer
        self.embedding_net = InputLayer(cfg)

        # create the actual GRU
        self.gru = nn.GRU(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        # add dropout between GRU and head
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # retrieve the model head
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)


# ### Implementing the Forward Pass
# 
# Now we have a class called GRU, but we haven't yet told the model how to process incoming data.
# That's what we do in the `forward` method.
# 
# By convention, our models' `forward` method accepts and returns dictionaries that map names (strings) to tensors.
# The input dictionary (`data`) usually contains at least a key 'x_d' and possibly 'x_s' and 'x_one_hot'.
# We say "usually", because models that support simultaneous prediction at multiple timescales (e.g., MTS-LSTM) will
# get one 'x_d' and 'x_s' for each timescale, suffixed with the frequency identifier (e.g., 'x_d_1H' for hourly dynamic inputs).
# 
# But for this example, let's assume a single-timescale model. Let's dive deeper into what each of the input values contain:
# 
# | Key         | Shape                                     | Description                                                                                                                                                                                                                                                 |
# |:------------|:------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | 'x_d'       | `[batch size, sequence length, features]` | the dynamic input data                                                                                                                                                                                                                                      |
# | 'x_s'       | `[batch size, features]`                  | static input features. These are the concatenation of what is defined in the run configuration under 'static_attributes' and 'evolving_attributes'. If not a single static or evolving attribute is defined in the config, 'x_s' will not be present.       |
# | 'x_one_hot' | `[batch size, number of basins]`          | one-hot encoding of the basins. If 'use_basin_id_encoding' is set to False in the run configuration, 'x_one_hot' will not be present.                                                                                                                       |
# 
# Now, given these input data we're supposed to generate a prediction that we return as 'y_hat' (multi-timescale models would return 'y_hat_1H', ...).
# The returned 'y_hat' should contain a prediction for the _full_ input sequence (not just the last element), even if you're using sequence-to-one prediction.
# The loss will sort out which of these predictions actually need to be used in the current training configuration.
# All models should at least return 'y_hat', but we can return any other potentially useful information.
# In our case, we can additionally return the final hidden state that we'll receive from the PyTorch GRU implementation.
# The naming convention for hidden states is to call them 'h_n'.
# 
# So, here we go:

# In[3]:


def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    # possibly pass dynamic and static inputs through embedding layers, then concatenate them
    x_d = self.embedding_net(data, concatenate_output=True)    

    # run the actual GRU
    gru_output, h_n = self.gru(input=x_d)

    # reshape to [batch_size, 1, n_hiddens]
    h_n = h_n.transpose(0, 1)

    pred = {'h_n': h_n}
    
    # add the final output as it's returned by the head to the prediction dict
    # (this will contain the 'y_hat')
    pred.update(self.head(self.dropout(gru_output.transpose(0, 1))))

    return pred

# usually, we'd implement the forward pass right where we define the class.
# For this tutorial, we've broken it down into the constructor and the forward pass,
# so now we'll just add the forward method to the GRU class:
GRU.forward = forward


# As you see, much of the heavy lifting is being done by existing methods, so we just have to wire everything up.
# The input layer merges the static inputs (`data['x_s']` and/or `data['x_one_hot']`) to each step of the dynamic inputs (`data['x_d']`) and returns a single tensor that we can pass to the GRU cell.
# 
# ### Using the Model
# 
# That's it! We now have a working GRU model that we can use to train and evaluate models.
# The only thing left is registering the model in the `get_model` method of `neuralhydrology.modelzoo` to make sure we can specify the model in a run configuration.
# 
# Since GRU already exists in the modelzoo, it's already there:
# 

# In[4]:


print(inspect.getsource(get_model))


# Since GRU is registered as a model, you can now specify `model: gru` in the run configuration and use the model, just like any other.
# For an example of training and evaluating a model, take a look at the [introduction tutorial](https://neuralhydrology.readthedocs.io/en/latest/tutorials/introduction.html).
