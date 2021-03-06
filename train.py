import matplotlib.pyplot as plt
import numpy as np
import torch
from efem import ElasticFEM
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import python_bindings.TGSL as TGSL
from net import Net
import config
import csv
import sim_example
import os

d = 3


class ConstrainedPosDataset(Dataset):
    # """Constrained Positions dataset."""
  #frames: number of totals frames
  #root_dir: directory of constrained positions
  def __init__(self, root_dir, frames, num_constrained):
    self.root_dir = root_dir
    self.frames = frames
    self.constrained_pos = torch.zeros(num_constrained*d, frames)
    for i in range(frames):
      vec = []
      filename = root_dir+'bc_frame_'+ str(i).zfill(6) + '_python.bin'
      TGSL.ReadDoubleVector(vec, bytes(filename, 'UTF-8'))
      self.constrained_pos[:, i] = torch.FloatTensor(vec)
      
  def __len__(self):
    return self.frames

  def __getitem__(self, idx):
    return self.constrained_pos[:, idx]



def energy_loss(unconstrained_pos, constrained_pos, unconstrained_node, constrained_node, efem, psi, mu, lam):
  pos = torch.zeros(d*efem.Np)
  for i in range(len(unconstrained_node)):
    for c in range(d):
      # if d*unconstrained_node[i]+c > d*efem.Np:
      pos[d*unconstrained_node[i]+c] = unconstrained_pos[d*i+c]
  for j in range(len(constrained_node)):
    for c in range(d):
      pos[d*constrained_node[j]+c] = constrained_pos[d*j+c]

  return efem.potential_energy(psi, pos, mu, lam), pos

def residual_loss(unconstrained_pos, constrained_pos, unconstrained_node, constrained_node, efem, P, mu, lam):
  pos = torch.zeros(d*efem.Np)
  for i in range(len(unconstrained_node)):
    for c in range(d):
      # if d*unconstrained_node[i]+c > d*efem.Np:
      pos[d*unconstrained_node[i]+c] = unconstrained_pos[d*i+c]
  for j in range(len(constrained_node)):
    for c in range(d):
      pos[d*constrained_node[j]+c] = constrained_pos[d*j+c]
  residual = efem.add_internal_force(pos, P, [], mu, lam)
  norm = torch.linalg.vector_norm(residual)
  return norm, pos

def batch_mean_energy_loss(unconstained_pos_batch, constrained_pos_batch, unconstrained_node, constrained_node, efem, psi, mu, lam):
  batch_size = len(unconstained_pos_batch)
  total = 0.0
  pos_batch = torch.zeros((batch_size, d*efem.Np))
  for i in range(batch_size):
    e, pos =  energy_loss(unconstained_pos_batch[i, :], constrained_pos_batch[i, :], unconstrained_node, constrained_node, efem, psi, mu, lam)
    total = total+ e
    pos_batch[i, :] = pos
  total = total/batch_size
  return total, pos_batch

def batch_mean_residual_loss(unconstained_pos_batch, constrained_pos_batch, unconstrained_node, constrained_node, efem, P, mu, lam):
  batch_size = len(unconstained_pos_batch)
  total = 0.0
  pos_batch = torch.zeros((batch_size, d*efem.Np))
  for i in range(batch_size):
    e, pos =  residual_loss(unconstained_pos_batch[i, :], constrained_pos_batch[i, :], unconstrained_node, constrained_node, efem, P, mu, lam)
    total = total+ e
    pos_batch[i, :] = pos
  total = total/batch_size
  return total, pos_batch

def assemble_pos( unconstrained_pos, constrained_pos, unconstrained_node, constrained_node, efem):
  x_mesh = torch.zeros(d*efem.Np)
  for i in range(len(unconstrained_node)):
    for c in range(d):
      # if d*unconstrained_node[i]+c > d*efem.Np:
      x_mesh[d*unconstrained_node[i]+c] = unconstrained_pos[d*i+c]
  for j in range(len(constrained_node)):
    for c in range(d):
      x_mesh[d*constrained_node[j]+c] = constrained_pos[d*j+c]
  return x_mesh


def train(argv):
  #set up:
  mu = 284.62
  lam = 576.92

  train_config = config.TrainConfig(argv)

  sim_ex = sim_example.SimExample(train_config)


  #load in our data:
  N_bc = 128
  boundary_mesh =[]
  TGSL.ReadIntVector(boundary_mesh, b'../tgsl/tools/geometry_processing/output/boundary_mesh_python.bin')
  all_bc_pos = []
  for i in range(120):
    vec = []
    filename = '../tgsl/tools/geometry_processing/output/bc_frame_'+ str(i).zfill(6) + '_python.bin'
    TGSL.ReadDoubleVector(vec, bytes(filename, 'UTF-8'))
    all_bc_pos.append(vec)
  all_bc_pos = torch.FloatTensor(all_bc_pos)
  print(all_bc_pos.shape)
  mesh = []
  TGSL.ReadIntVector(mesh, b'../tgsl/tools/geometry_processing/output/mesh_python.bin')
  X_mesh = []
  TGSL.ReadDoubleVector(X_mesh, b'../tgsl/tools/geometry_processing/output/x_mesh_python.bin')
  X_mesh = torch.FloatTensor(X_mesh)
  efem = ElasticFEM(mesh, X_mesh)
  constrained_nodes = []
  TGSL.ReadIntVector(constrained_nodes, b'../tgsl/tools/geometry_processing/output/grid_based_constrained_nodes.bin')
  #compute unconstrained nodes:
  node_used = np.zeros(int(len(X_mesh)/3), dtype = bool)
  for i in constrained_nodes:
    node_used[i] = 1
  unconstrained_nodes = [i for i in range(efem.Np) if not node_used[i]]
  #X_unconstrained = [X_mesh[i] for i in range(efem.Np*d) if not node_used[int(i/3)]]

  box_dataset = ConstrainedPosDataset('../tgsl/tools/geometry_processing/output/',train_config.frames, len(constrained_nodes))

  train_loader = DataLoader(box_dataset, batch_size = train_config.batch_size, shuffle = train_config.shuffle)

  net = Net(N_bc*d, len(unconstrained_nodes)*d, train_config)

  if train_config.optimizer == 0:
    optimizer = optim.SGD(net.parameters(), lr = train_config.learning_rate)
  else:
    optimizer = optim.Adam(net.parameters(), lr = train_config.learning_rate)

  #load model if needed:
  if train_config.load_model:
    checkpoint = torch.load(train_config.load_dir+'checkpoint_2950.tar')
    net.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
    #loss = checkpoint['loss']



  batch_size = train_config.batch_size

  output_dir = './output/optimizer_'+str(train_config.optimizer)+'_activation_'+str(train_config.activation)+'_model_'+str(train_config.model_number)+'_lr_'+str(train_config.learning_rate)+'_shuffle_'+str(train_config.shuffle)+'_use_residual_'+str(train_config.use_residual)+'_ls_'+str(train_config.layer_size)+'_f_'+str(train_config.frames)+'/'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  for epoch in range(3000):
    loss = 0.0
    print("training epoch:" + str(epoch))
    if epoch % 50 == 0 and epoch > 0:
      torch.save({
              'epoch': epoch,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss
              }, output_dir+'checkpoint_'+ str(epoch)+'.tar')
    if epoch % 50 == 0 and epoch > 0:
      filename = output_dir+"residuals_epoch_" + str(epoch)+ ".csv"
      fields = ['energy', 'newton_residual'] 
      with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(fields) 
    for i, constrained_pos_batch in enumerate(train_loader):
      optimizer.zero_grad()
      #constrained_pos = constrained_pos.reshape(d*len(constrained_nodes))
      unconstrained_pos_batch = net.forward(constrained_pos_batch)
      #print(unconstrained_pos_batch)
      #print(net.state_dict())
      loss = torch.zeros(1)
      pos_batch = torch.zeros(1)
      if train_config.use_residual:
        loss, pos_batch = batch_mean_residual_loss(unconstrained_pos_batch, constrained_pos_batch, unconstrained_nodes, constrained_nodes, efem, sim_ex.p, mu, lam)
      else:
        loss, pos_batch = batch_mean_energy_loss(unconstrained_pos_batch, constrained_pos_batch, unconstrained_nodes, constrained_nodes, efem, sim_ex.psi, mu, lam)
      
      print(loss)
      
      
      loss.backward()
      #if train_config.verbose:
      print(net.fc1.weight.grad.norm())
      
      optimizer.step()
      print("batch number" + str(i))
      #write data into file:
      if epoch % 50 == 0 and epoch > 0:
        if train_config.shuffle:
          for f in range(box_dataset.frames):
            filename = output_dir+'epoch_'+ str(epoch).zfill(3) + '_frame_' + str(f).zfill(3)+'_python.geo'
            constrained_pos = all_bc_pos[f, :]
            unconstrained_pos = net.forward(constrained_pos)
            x_mesh = assemble_pos(unconstrained_pos, constrained_pos, unconstrained_nodes, constrained_nodes, efem)
            TGSL.WriteTrisFrame(x_mesh, int(len(x_mesh)/3), boundary_mesh, len(boundary_mesh), bytes(filename, 'UTF-8'))
        else:
          for b in range(batch_size):
            filename = output_dir+'epoch_'+ str(epoch).zfill(3) + '_frame_' + str(i*batch_size+b).zfill(3)+'_python.geo'
            TGSL.WriteTrisFrame(pos_batch[b, :], int(len(pos_batch[b,:])/3), boundary_mesh, len(boundary_mesh), bytes(filename, 'UTF-8'))
            #write loss data into csv:
            residual = efem.add_internal_force(pos_batch[b,:], sim_ex.p, [], mu, lam)
            norm = torch.linalg.vector_norm(residual)
            energy = efem.potential_energy(sim_ex.psi, pos_batch[b,:], mu, lam)
            a = energy.detach().numpy()
            b = norm.detach().numpy()
            rows=[a[0],b]
            with open(output_dir+"residuals_epoch_" + str(epoch)+ ".csv",'a') as fd:
              writer = csv.writer(fd)
              writer.writerow(rows)
  
  # for i in range(box_dataset.frames):
  #   filename = './epoch_'+ str(epoch).zfill(3) + '_frame_' + str(i).zfill(3)+'_python.geo'
  #   constrained_size = len(constrained_nodes)
  #   x_mesh = assemble_pos(net.forward(all_bc_pos[i*constrained_size*d:(i+1)*constrained_size*d-1]), all_bc_pos[i*constrained_size*d:(i+1)*constrained_size*d-1], unconstrained_nodes, constrained_nodes)
  #   TGSL.WriteTrisFrame(x_mesh, int(len(x_mesh)/3), boundary_mesh, len(boundary_mesh), bytes(filename, 'UTF-8'))



