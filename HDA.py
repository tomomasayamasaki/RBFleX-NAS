import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from models import get_cell_based_tiny_net
from nats_bench import create
import time
import pandas as pd
import random
from DownsampledImageNet import ImageNet16


# ==============================================
# GLOBAL VARIABLE: 
# Batch size for RBFleX-NAS
# N_GAMMA: Number of network to detect hyperparameter for RBF kernel
# ==============================================
batch_size_NE = 3
N_GAMMA = 10
# ==============================================
# GLOBAL VARIABLE: Experiment for RBFleX-NAS
# - cifar10
# - cifar100
# - ImageNet16-120
# ==============================================
dataset = 'cifar10'
# ==============================================
# GLOBAL VARIABLE: create a searchspace
# This example is NATS-Bench-SSS
# ==============================================
benchmark_root = "./designspace/NATS-Bench-SSS/NATS-sss-v1_0-50262-simple"
# NAS Benchmark
print('Loading...NAT Bench '+"sss")
searchspace = create(benchmark_root, "sss", fast_mode=True, verbose=False)
# ==============================================
# GLOBAL VARIABLE: create a dataloader
# This example is cifar-10
# ==============================================
img_root = "./dataset"
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
imgset = torchvision.datasets.CIFAR10(
        root=img_root+'/cifar10', train=True, download=True, transform=transform_train)
img_loader = torch.utils.data.DataLoader(
        imgset, batch_size=batch_size_NE, shuffle=True, num_workers=1, pin_memory=True)


 
#######################################
# Normalization 
# - Column-wise normalization
#######################################
def normalize(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    x_max[x_max == x_min] = 1
    x_min[x_max == x_min] = 0
    return (x - x_min) / (x_max - x_min)

def main():
  # Reproducibility
  print('==> Reproducibility..')
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.set_printoptions(precision=8)
  np.random.seed(1)
  torch.manual_seed(1)

  # GPU
  # Check that CUDA is available
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # Check that MPS is available
  if torch.cuda.is_available():
    device = 'cuda:0'
  else:
    device = 'cpu'
  print('device GPU: ', device)
  tkwargs = {
    "dtype": torch.float64,
    "device": device,
    "requires_grad":False
  }
      

  # Hyperparameter
  print('==> Preparing hyperparameters..')
  
  batch_space = random.sample(range(len(searchspace)), N_GAMMA)

  ########################
  # Compute Distance and Kernel Matrix
  ########################
  def counting_forward_hook(module, inp, out):
    with torch.no_grad():
      arr = out.view(batch_size_NE, -1)
      network.K = torch.cat((network.K, arr),1)
      
  def counting_forward_hook_FC(module, inp, out):
      with torch.no_grad():
        if isinstance(inp, tuple):
            inp = inp[0]
        network.Q = inp
  
  #######################################
  # Self-detecting Hyperparameter
  #######################################
  GAMMA_K_list = []
  GAMMA_Q_list = []
  for id in range(N_GAMMA):
    uid = batch_space[id]
    config = searchspace.get_net_config(uid, dataset)
    network = get_cell_based_tiny_net(config)
    network = network.to(device)
    
    net_counter = list(network.named_modules())
    net_counter = len(net_counter)
    NC = 0
    for name, module in network.named_modules():
      NC += 1
      if 'ReLU' in str(type(module)):
        module.register_forward_hook(counting_forward_hook)
      if NC == net_counter:
        module.register_forward_hook(counting_forward_hook_FC)
        
    with torch.no_grad():
      network.K = torch.empty(0, device=device)
      network.Q = torch.empty(0, device=device)
      network(x[0:batch_size_NE,:,:,:].to(device))
      
      Output_matrix = network.K
      Last_matrix = network.Q
    
    with torch.no_grad():
      Output_matrix = Output_matrix.cpu().numpy()
      Last_matrix = Last_matrix.cpu().numpy()
      
    for i in range(batch_size_NE-1):
      for j in range(i+1,batch_size_NE):
          z1 = Output_matrix[i,:]
          z2 = Output_matrix[j,:]
          m1 = np.mean(z1)
          m2 = np.mean(z2)
          M = (m1-m2)**2
          z1 = z1-m1
          z2 = z2-m2
          s1 = np.mean(z1**2)
          s2 = np.mean(z2**2)
          if s1+s2 != 0:
            candi_gamma_K = M/((s1+s2)*2)
            GAMMA_K_list.append(candi_gamma_K)
            
    for i in range(batch_size_NE-1):
      for j in range(i+1,batch_size_NE):
          z1 = Last_matrix[i,:]
          z2 = Last_matrix[j,:]
          m1 = np.mean(z1)
          m2 = np.mean(z2)
          M = (m1-m2)**2
          z1 = z1-m1
          z2 = z2-m2
          s1 = np.mean(z1**2)
          s2 = np.mean(z2**2)
          if s1+s2 != 0:
            candi_gamma_Q = M/((s1+s2)*2)
            GAMMA_Q_list.append(candi_gamma_Q)
          
  GAMMA_K = np.min(np.array(GAMMA_K_list))
  GAMMA_Q = np.min(np.array(GAMMA_Q_list))
  print('==> Detected Hyperparameter Gamma ..')
  print('gamma_k:',GAMMA_K)
  print('gamma_q:',GAMMA_Q)