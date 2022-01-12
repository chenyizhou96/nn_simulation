import torch 
from net import Net
import config


d = 3
N_bc = 128



  checkpoint = torch.load('./nn_data/linear_elasticity/checkpoint_511.tar')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']

model.eval()
# - or -
model.train()