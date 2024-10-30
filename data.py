import h5py
import torch
import math
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from conf import cfg 

class RadioML_Datset(Dataset):
    def __init__(self, file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',  snr=10, train=True):
        np.random.seed(0)
        with h5py.File(file_name, 'r') as f:
            x = f['X'][f['Z'][:,0]== snr,:,:].transpose((0,2,1))
            y = f['Y'][f['Z'][:,0]== snr,:]
            # print(f"Min I : {x.min()}")
            # print(f"Max I : {x.max()}")
            indices = np.random.permutation(x.shape[0])
            split_n = int(math.floor(y.shape[0]*0.8))
            tr_idx,tst_idx = indices[:split_n].astype(int), indices[split_n:].astype(int)
            # mean = np.array([0.00085, -.0029]).reshape(1,2,1)
            # var = np.array([1.58, 0.57]).reshape(1,2,1)
            # min_i ,max_i = -6.32, 7.009
            # min_q, max_q = -10.33,14.42

            # x[:,0,:] = (x[:,0,:]- min_i)/(max_i-min_i)
            # x[:,1,:] = (x[:,1,:]- min_q)/(max_q-min_q)

        
        if train:
            self.x, self.y = x[tr_idx, :, :], y[tr_idx, :]
        else:
            self.x, self.y = x[tst_idx, :, :], y[tst_idx, :]
        

        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        iq_data = self.x[index]
        label = self.y[index]
        data = {'iq_data':iq_data, 'label':label}
        return data
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


    
def get_loader(cfg=cfg,
                file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                snr=10,
                batch_size = 64,
                train=False,
                num_workers=4,
                shuffle = False
                ):

    g = torch.Generator()
    g.manual_seed(cfg.BASE.SEED)    
    dataset = RadioML_Datset(file_name=file_name, snr=snr, train=train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            worker_init_fn=seed_worker,
                            generator=g,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader

    
def get_fisher_loader(cfg=cfg,
                file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                snr=10,
                batch_size = 64,
                train=True,
                num_workers=4,
                shuffle = False,
                ):

    g = torch.Generator()
    g.manual_seed(cfg.BASE.SEED)    
    dataset = RadioML_Datset(file_name=file_name, snr=snr, train=train)
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    np.random.seed(0)
    np.random.shuffle(dataset_indices)
    fisher_split_index = int(np.floor(0.1*dataset_size))
    fisher_idx = dataset_indices[:fisher_split_index]
    fisher_sampler = SubsetRandomSampler(fisher_idx)

    fisher_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, 
                            sampler=fisher_sampler,worker_init_fn=seed_worker, shuffle = shuffle,
                            generator=g, num_workers=num_workers)
    return fisher_dataloader



def sanity_check():

    dataloader = get_loader(file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                   snr=10,
                   batch_size = 64,
                   train=True,
                   num_workers=4)
    
    for i,data in enumerate(dataloader):
        x , y = data['iq_data'], data['label']
        print(f'data shape: {x.size()}')
        amp = torch.mean(torch.sqrt(torch.sum(torch.pow(x, 2), dim=1)))
        print(f"Am ==>{amp}")
        print(f'Maximum in I value: {x[:,0,:].max()}')
        print(f'Maximum in Q value: {x[:,1,:].max()}')
        print(f'Minimum in I value: {x[:,0,:].min()}')
        print(f'Minimum in Q value: {x[:,1,:].min()}')
        print(f'Average in I value: {torch.abs(x[:,0,:]).mean()}')
        print(f'Average in Q value: {torch.abs(x[:,1,:]).mean()}')
        print(f'label shape: {x.size()}')
        print(x[:,0,100:200])
        break

if __name__ == '__main__':

    sanity_check()

        