
import numpy as np
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
import time
import random
import designspace.Trans_Macro.utils as utils
from designspace.Trans_Macro.model_wrapper.cnn_wrapper import CNNWrapper
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr


# ==============================================
# GLOBAL VARIABLE: DESIGN SPACE CONFIGURATION
# - config_macs_transmicrosegmentsemantic
# - config_macs_transmicroautoencoder
# - config_macs_transmicroclassobject
# - config_macs_transmicroclassscene
# - config_macs_transmicrojigsaw
# - config_macs_transmicronormal
# - config_macs_transmicroroomlayout
# ==============================================
CONFIG_PATH = "config_macs_transmicrosegmentsemantic"

# ==============================================
# GLOBAL VARIABLE: Batch size for RBFleX-NAS
# ==============================================
batch_size_NE = 3




CONFIG = utils.get_config_tranmicro(CONFIG_PATH)
searchspace = utils.get_searchspace_tranmicro(CONFIG)
Num_Networks = len(searchspace)
INDEX_ACC = 1
INDEX_ARCH = 0
maxtrials = 1

 
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
  img_root = "./dataset/TASKONOMY/rgb"
  

  # Image Data
  print('==> Preparing data..')
  train_transform = transforms.Compose([
      transforms.ToTensor()
  ])
  imgset = ImageFolder(root=img_root,transform=train_transform)
  img_loader = torch.utils.data.DataLoader(imgset, batch_size=batch_size_NE, shuffle=True, num_workers=5)
  data_iterator = iter(img_loader)
  x, _ = next(data_iterator)
  
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
  score_list = list()
  accuracy_list = list()
  best_acc = 0
  best_score = -1000000000000000
  s = time.time()
  with torch.no_grad():
    for r in range(maxtrials):
      ss = time.time()
      batch_space = random.sample(range(len(searchspace)), Num_Networks)
      ee = time.time()
      tot_gene += ee-ss
      cn = 1
      for uid in batch_space:
        ss = time.time()
        arch = searchspace[uid][INDEX_ARCH]
        CONFIG.backbone_config.arch = arch
        network = CNNWrapper(CONFIG.backbone_config, CONFIG.head_config)
        network = network.to(device)
        ee = time.time()
        tot_gene += ee-ss
         
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
        score = score.item()
        if np.isinf(score):
           score = -100000000
        score_list.append(score)
        accuracy_list.append(searchspace[uid][INDEX_ACC])
        print("{}/{} ({}%)".format(cn, Num_Networks, np.round(100*(cn/Num_Networks))))
        cn += 1
        if score > best_score:
          best_score = score
          best_acc = searchspace[uid][INDEX_ACC]
          print("trial:{} best_acc: {}".format(r, best_acc))
        ee = time.time()
        tot_gene += ee-ss
      total_acc.append(best_acc)
    e = time.time()

  print("=================================================")
  print("Task: ", CONFIG_PATH)
  print('dataset:{}, Bench:{} trial:{} time:{}'.format("taskonomy", 'TransNAS-Bench-101', maxtrials, e-s-tot_gene))
  print('each trial time: {}'.format((e-s-tot_gene)/maxtrials))
  print('AVE. accuracy:{}'.format(total_acc))

  # Calculate correlations
  spearman_corr, spearman_p = spearmanr(score_list, accuracy_list)
  print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p}")
  pearson_corr, pearson_p = pearsonr(score_list, accuracy_list)
  print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p}")
  kendall_corr, kendall_p = kendalltau(score_list, accuracy_list)
  print(f"Kendall correlation: {kendall_corr}, p-value: {kendall_p}")


  # Save Result
  df = pd.DataFrame(score_list)
  df.to_csv("./RBFleX_score.csv", index=False)
  df = pd.DataFrame(accuracy_list)
  df.to_csv("./RBFleX_accuracy.csv", index=False)




if __name__ == '__main__':
  
  main()
  

