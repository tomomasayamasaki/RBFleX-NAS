import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

import pandas as pd
import os
from dataset.SST2.data_sst2 import DataPrecessForSentence
from BERT_model import BertModel



# ==============================================
# GLOBAL VARIABLE: Batch size for RBFleX-NAS
# ==============================================
batch_size=3





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
max_seq_len=50

data_path = "./dataset/SST2/data"
train_df = pd.read_csv(os.path.join(data_path,"train.tsv"),sep='\t',header=None, names=['similarity','s1'])






ACTIVATIONS = ['ReLU',
              'ELU',
              'Hardtanh',
              'Hardswish',
              'LeakyReLU',
              'ReLU6',
              'SELU',
              'CELU',
              'GELU',
              'SiLU',
              'Mish']



########################
# Compute Distance and Kernel Matrix
########################
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

def counting_forward_hook(module, inp, out):
      arr = out.view(-1)
      network.K = torch.concatenate([network.K, arr])
      
def counting_forward_hook_FC(module, inp, out):
    arr = inp[0].view(-1)
    network.Q = torch.concatenate([network.Q, arr])
        
def hook_nested_layers(model, prefix=""):
    for name, module in model.named_children():
        #print('module: ', module)
        if 'Bert' in str(module) or isinstance(module, torch.nn.ModuleList) or isinstance(module, torch.nn.Sequential):
            hook_nested_layers(module)
        else:
            if 'activation' in str(name):
                #print('K -> name: {} module:{} '.format(name, module))
                module.register_forward_hook(counting_forward_hook)
            elif 'classifier' in str(name):
                #print('Q -> name: {} module:{} '.format(name, module))
                module.register_forward_hook(counting_forward_hook_FC)
            #else:
                #print('name: {}'.format(name))

# Detect hyperparameter GAMMA
GAMMA_K_list = []
GAMMA_Q_list = []

for activation in ACTIVATIONS:
            bertmodel = BertModel(requires_grad = True, activation=activation)
            tokenizer = bertmodel.tokenizer
            train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len = max_seq_len)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
            data_iterator = iter(train_loader)
            x = next(data_iterator)
            batch_seqs = x[0]
            batch_seq_masks = x[1]
            batch_seq_segments = x[2]
            batch_labels = x[3]
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            network = bertmodel.to(device)
            
            # linear classification layer on top.
            network.eval()
              
            net_counter = list(network.named_modules())
            net_counter = len(net_counter)
            NC = 0
            hook_nested_layers(network)
            
            with torch.no_grad():
                network.K = torch.empty(0, device=device)
                network.Q = torch.empty(0, device=device)
                network(seqs, masks, segments, labels)
                
                Output_matrix = network.K
                Last_matrix = network.Q
                
            with torch.no_grad():
                Output_matrix = Output_matrix.cpu().numpy()
                Last_matrix = Last_matrix.cpu().numpy()
                
            #print(Output_matrix.shape)
            for i in range(batch_size-1):
                for j in range(i+1,batch_size):
                    #print(Output_matrix[i,:].shape)
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
                    
            for i in range(batch_size-1):
                for j in range(i+1,batch_size):
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

GAMMA_K_arr = np.array(GAMMA_K_list)
GAMMA_Q_arr = np.array(GAMMA_Q_list)
#print(GAMMA_Q_arr)
filtered_K_arr = GAMMA_K_arr[GAMMA_K_arr > 0]
filtered_Q_arr = GAMMA_Q_arr[GAMMA_Q_arr > 0]
GAMMA_K = np.min(filtered_K_arr)
GAMMA_Q = np.min(filtered_Q_arr)
print('gamma_k:',GAMMA_K)
print('gamma_q:',GAMMA_Q)



for activation in ACTIVATIONS:
            bertmodel = BertModel(requires_grad = True, activation=activation)
            tokenizer = bertmodel.tokenizer
            train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len = max_seq_len)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
            data_iterator = iter(train_loader)
            x = next(data_iterator)
            batch_seqs = x[0]
            batch_seq_masks = x[1]
            batch_seq_segments = x[2]
            batch_labels = x[3]
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            network = bertmodel.to(device)
            
            # linear classification layer on top.
            network.eval()
              
            net_counter = list(network.named_modules())
            net_counter = len(net_counter)
            NC = 0
            hook_nested_layers(network)
            
            with torch.no_grad():
                network.K = torch.tensor([], **tkwargs)
                network.Q = torch.tensor([], **tkwargs)
                network(seqs, masks, segments, labels)

                LA = len(network.K)
                LAQ = len(network.Q)
                
                Output_matrix = torch.zeros([batch_size, LA], **tkwargs)
                Last_matrix = torch.zeros([batch_size, LAQ], **tkwargs)
                for i in range(batch_size):
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

                K_Matrix = Simularity_Mat(Output_matrix, GAMMA_K)
                Q_Matrix = Simularity_Mat(Last_matrix, GAMMA_Q)
            
                _, K = torch.linalg.slogdet(K_Matrix)
                _, Q = torch.linalg.slogdet(Q_Matrix)
                score = batch_size*(K+Q)
            
            print('Activation: {} score:{}'.format(activation,score))
                
            

        
              
              
              



