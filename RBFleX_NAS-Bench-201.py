
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
# GLOBAL VARIABLE: Batch size for RBFleX-NAS
# ==============================================
batch_size_NE = 3
# ==============================================
# GLOBAL VARIABLE: Experiment for RBFleX-NAS
# - cifar10
# - cifar100
# - ImageNet16-120
# ==============================================
dataset = 'cifar10'
# ==============================================
# GLOBAL VARIABLE: Experiment for RBFleX-NAS
# maxtrials: a number of trials
# Num_Networks: a number of networks selcted randomly from benchmark
# ==============================================
maxtrials = 10
Num_Networks = 1000



benchmark_root = "./designspace/NAS-Bench-201/NATS-tss-v1_0-3ffb9-simple"


img_root = "./dataset"
batch_size_NE = 3
if "sss" in benchmark_root:
  design_space = "sss"
  hp = "90"
  test_idx = 'test-accuracy' 
elif "tss" in benchmark_root:
  design_space = "tss"
  hp = "200"
  test_idx = 'test-accuracy'
 
def normalize(x, axis=None):
    x_min = torch.min(x, dim=axis, keepdim=True).values
    x_max = torch.max(x, dim=axis, keepdim=True).values
    x_max[x_max == x_min] = 1
    x_min[x_max == x_min] = 0
    return (x - x_min) / (x_max - x_min)

def Simularity_Mat(matrix, gamma):    
    x_norm = torch.sum(matrix ** 2, dim=-1)
    a = x_norm[:, None]
    b = x_norm[None, :]
    c = torch.matmul(matrix, matrix.T)
    simularity_matrix = torch.exp(-gamma * (a + b - 2 * c))
    return simularity_matrix

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
  

  # Image Data
  print('==> Preparing data..')
  if dataset == "ImageNet16-120":
    norma = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.ToTensor(),
        norma,
    ])
    imgset = ImageFolder(root=img_root+'/ImageNet',transform=train_transform)
    img_loader = torch.utils.data.DataLoader(imgset, batch_size=batch_size_NE, shuffle=True, num_workers=1, pin_memory=True)
    #train_data = ImageNet16(img_root+"/ImageNet16", True , train_transform, 120)
    #img_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_NE, shuffle=True, num_workers=1, pin_memory=True)
  elif dataset == "cifar10":
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
  elif dataset == "cifar100":
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    imgset = torchvision.datasets.CIFAR100(
        root=img_root+'/cifar100', train=True, download=True, transform=transform_train)
    img_loader = torch.utils.data.DataLoader(
        imgset, shuffle=True, num_workers=1, batch_size=batch_size_NE, pin_memory=True)
  data_iterator = iter(img_loader)
  x, _ = next(data_iterator)

  # NAS Benchmark
  print('Loading...NAT Bench '+design_space)
  searchspace = create(benchmark_root, design_space, fast_mode=True, verbose=False)
  
    

  # Model
  print('==> Building model..')
  ########################
  # Compute Distance and Kernel Matrix
  ########################
  def counting_forward_hook(module, inp, out):
      arr = out.view(-1)
      network.K = torch.concatenate([network.K, arr])
      
  def counting_forward_hook_FC(module, inp, out):
      arr = inp[0].view(-1)
      network.Q = torch.concatenate([network.Q, arr])
          
  GAMMA_K = 2.68690173088039e-12
  GAMMA_Q = 1.02460061284506e-11

  # compute score
  tot_gene = 0
  total_acc = list()
  total_acc_c10 = list()
  total_acc_c100 = list()
  best_acc = 0
  best_score = -100000000000
  s = time.time()
  with torch.no_grad():
    for r in range(maxtrials):
      ss = time.time()
      batch_space = random.sample(range(len(searchspace)), Num_Networks)
      ee = time.time()
      tot_gene += ee-ss

      for uid in batch_space:
        config = searchspace.get_net_config(uid, dataset)
        network = get_cell_based_tiny_net(config)
        network = network.to(device)
        
            
        net_counter = list(network.named_modules())
        net_counter = len(net_counter)
        NC = 0
        for _, module in network.named_modules():
          NC += 1
          if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
          if NC == net_counter:
            module.register_forward_hook(counting_forward_hook_FC)
        
        x2 = torch.clone(x[0:1,:,:,:])
        x2 = x2.to(device)
        network.K = torch.tensor([], **tkwargs)
        network.Q = torch.tensor([], **tkwargs)
        network(x2)
        LA = len(network.K)
        LAQ = len(network.Q)
        
        Output_matrix = torch.zeros([batch_size_NE, LA], **tkwargs)
        Last_matrix = torch.zeros([batch_size_NE, LAQ], **tkwargs)
        for i in range(batch_size_NE):
          x2 = torch.clone(x[i:i+1,:,:,:])
          x2 = x2.to(device)
          network.K = torch.tensor([], **tkwargs)
          network.Q = torch.tensor([], **tkwargs)
          network(x2)
          Output_matrix[i,:] = network.K
          Last_matrix[i,:] = network.Q
          
        # Normalization
        Output_matrix = normalize(Output_matrix, axis=0)
        Last_matrix = normalize(Last_matrix, axis=0)
        
        # RBF kernel
        K_Matrix = Simularity_Mat(Output_matrix, GAMMA_K)
        Q_Matrix = Simularity_Mat(Last_matrix, GAMMA_Q)
        _, K = torch.linalg.slogdet(K_Matrix)
        _, Q = torch.linalg.slogdet(Q_Matrix)
        score = batch_size_NE*(K+Q)
        ss = time.time()
        #print(uid)
        if score > best_score:
          best_score = score
          best_acc = searchspace.get_more_info(uid, dataset,None, hp, True)[test_idx] #searchspace.simulate_train_eval(uid, dataset=dataset, hp=hp)
          best_acc_c10 = searchspace.get_more_info(uid, 'cifar10',None, hp, True)[test_idx] #searchspace.simulate_train_eval(uid, dataset='ImageNet16-120', hp=hp)
          best_acc_c100 = searchspace.get_more_info(uid, 'cifar100',None, hp, True)[test_idx] #searchspace.simulate_train_eval(uid, dataset='cifar100', hp=hp)
        ee = time.time()
        tot_gene += ee-ss
      total_acc.append(best_acc)
      total_acc_c10.append(best_acc_c10)
      total_acc_c100.append(best_acc_c100)
    e = time.time()

  print('dataset:{}, Bench:{} trial:{} time:{}'.format(dataset, 'sss', maxtrials, e-s-tot_gene))
  print('each trial time: {}'.format((e-s-tot_gene)/maxtrials))
  print('IM AVE. accuracy:{} std:{}'.format(np.mean(total_acc), np.std(total_acc)))
  print('C10 AVE. accuracy:{} std:{}'.format(np.mean(total_acc_c10), np.std(total_acc_c10)))  
  print('C100 AVE. accuracy:{} std:{}'.format(np.mean(total_acc_c100), np.std(total_acc_c100)))

  print("total_accraucy -------")
  print(total_acc)

if __name__ == '__main__':
  
  main()
  

