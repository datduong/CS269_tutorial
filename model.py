

from __future__ import unicode_literals, print_function, division
from io import open
# import unicodedata
import string, re, sys, pickle, gzip
import numpy as np
# import pandas as pd 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_


class Feature_Layer (nn.Module): 
  def __init__(self,num_of_feature,final_layer_dim,drop_out=0.1): 
    super ( Feature_Layer, self).__init__()
    self.num_of_feature = num_of_feature
    self.final_layer_dim = final_layer_dim
    self.standard_linear = nn.Sequential ( # @nn.Sequential put many layers into one single function 
      nn.Linear( self.num_of_feature, self.num_of_feature//2, bias=True ),
      nn.Tanh(),
      nn.Dropout(p=drop_out), ## avoids overfit
      nn.Linear( self.num_of_feature//2, self.num_of_feature//4, bias=True ),
      nn.Tanh(),
      nn.Dropout(p=drop_out), 
      nn.Linear( self.num_of_feature//4, self.final_layer_dim, bias=True ),
      nn.Tanh()
      ) 
  def forward_x(self,x) : # doing f(x)
    return self.standard_linear(x)


class Function_XY (nn.Module): 
  def __init__(self,num_of_labels): 
    super ( Function_XY, self).__init__()
    self.num_of_labels = num_of_labels
  def forward_xy( self,input,label,do_print=False): 
    # @input is batch num_sample x num_feature 
    # @label is num_sample x num_feature 
    phi = torch.sum( input * label, dim=1 )
    if do_print:
      print ('\nlabel is')
      print (label)
      print ('function value is')
      print (phi)
    return phi


## define parameters
num_of_labels = 5
num_of_feature = 100
final_layer_dim = num_of_labels ## must match @num_of_labels

## define the neural network functions 
fx = Feature_Layer (num_of_feature,final_layer_dim)
phi_xy = Function_XY (num_of_labels)

## apply to data 
batch_size = 8
x = torch.randn(batch_size,num_of_feature) ## 16 random samples in 1 batch ... so to speak.
new_x = fx.forward_x(x) ## pass @x into neural network 

init_y = torch.zeros ( batch_size,num_of_labels) + 0.5 ## initial guess 

guess_label = Variable( init_y , requires_grad=True ) ## use @Variable, start at some feasible point
optimizer = optim.SGD([guess_label], lr = 0.01, momentum=0.9) ## tell the @optim to only optimize @guess_label 


for i in range(50): ## do 50 iterations 
  print ('\niteration {}'.format(i))
  optimizer.zero_grad() ## set all gradient to be zero, otherwise, gradient will get larger for each iteration. 
  function_value = -1* phi_xy.forward_xy(new_x, guess_label,do_print=True) ## notice takes -1 multiplication 
  ## using .sum() sometimes break the computation graph
  function_value = function_value.sum() ## add all the function value over all batch
  function_value.backward(retain_graph=True) ## without retain_graph=True, you will see error 
  optimizer.step()
  ## notice, output must be bounded between [0,1] so we have to project back to this space [0,1]
  guess_label.data [guess_label.data >1] = 1 # must use ".data" to access the "data" inside the @guess_label 
  guess_label.data [guess_label.data <0] = 0


