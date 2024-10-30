import torch
import numpy as np
import random
import argparse
from conf import cfg
import torch.optim as optim
from data import get_loader
from copy import deepcopy
from tta_algo.build import build_tta_adapter
from tta_attack.dia import DIA
from tta_attack.u_dia import U_DIA
from utils.util import accuracy, AverageMeter
from torch.profiler import profile, record_function, ProfilerActivity 


def set_deterministic(seed=42, cudnn=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = cudnn

""" def setup_optimizer(params):

    lr_adapt = cfg.OPTIM.LR_ADAPT 
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=lr_adapt,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=lr_adapt,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError """

# latency_table = wandb.Table(columns=["GPU_Latency", "CPU_Latency"])
# def trace_handler(p):
#     cpu_lat = p.key_averages().self_cpu_time_total/1000
#     variable = p.key_averages().table(sort_by='self_cuda_time_total', row_limit=0)
#     #print(type(variable))
#     str_ = 'Self CUDA time total'
#     CUDA_time = variable.find(str_)

#     #print(variable[CUDA_time + len(str_) + 2 : CUDA_time + len(str_) + 7])
#     cuda_lat = variable[CUDA_time + len(str_) + 2 : CUDA_time + len(str_) + 7]
#     print(f"Cuda Latency {cuda_lat} and CPU Latency{cpu_lat}")
#     latency_table.add_data(cpu_lat, float(cuda_lat))

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def test_tta(model,data_loader,cfg,device):
    model=model.to(device)
    tta_adapter_cls = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_cls(cfg=cfg,model=model)
    acc = AverageMeter()
    for n_batch,data in enumerate(data_loader):
        inputs, labels = data['iq_data'], data['label']
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float).argmax(dim=1)
        outputs = tta_adapter.forward(inputs).detach().cpu()
        #outputs = model(x)
        if outputs[:-cfg.DIA.MAL_NUM].size()[0]:
            acc.update(accuracy(outputs[:-cfg.DIA.MAL_NUM].cpu(),labels[:-cfg.DIA.MAL_NUM].cpu()))
        #print(f"Accuracy ==> {acc.avg}")
    return acc.avg

def test_normal(model,data_loader,cfg,device):
    model=model.to(device)
    model = model.eval()
    acc = AverageMeter()
    for n_batch,data in enumerate(data_loader):
        inputs, labels = data['iq_data'], data['label']
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float).argmax(dim=1)
        outputs = model.forward(inputs).detach().cpu()
        if outputs[:-cfg.DIA.MAL_NUM].size()[0]:
            acc.update(accuracy(outputs[:-cfg.DIA.MAL_NUM],labels[:-cfg.DIA.MAL_NUM].cpu()))
        #print(f"Accuracy ==> {acc.avg}")
    return acc.avg

def test_attack(model, data_loader, cfg,device):

    model = model.to(device)
    victim_model = deepcopy(model)
    victim_model = victim_model.to(device)
    tta_adapter_class = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_class(cfg,model)
    tta_adapter_victim = tta_adapter_class(cfg, victim_model)

    if cfg.BASE.ATTACK == "dia":
        attack = DIA(cfg=cfg)
    elif cfg.BASE.ATTACK == "u_dia":
        attack = U_DIA(cfg=cfg, layers=['avgpool'])
    else:
        raise NotImplementedError("Specify an attack Name that is implemeted")
    acc_normal = AverageMeter()
    acc_attacked = AverageMeter()

    for n_batch,data in enumerate(dataloader):

        inputs, labels = data['iq_data'], data['label']
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float).argmax(dim=1)
        
        outputs_clean = tta_adapter.forward(inputs).detach().cpu()
        # if cfg.DIA.PSEUDO:
        #     x_adv = attack.generate_attack(sur_model=tta_adapter.model,
        #                                x = inputs, y=outputs_clean)
        # else:
        x_adv = attack.generate_attack(sur_model=tta_adapter.model,
                                       x = inputs, y=labels)
        
        # amp_adv = torch.mean(torch.sqrt(torch.sum(torch.pow(x_adv[-cfg.DIA.MAL_NUM:], 2), dim=1)))  
        # print(f"Amplitude of adversarial {amp_adv}")
        # amp_normal = torch.mean(torch.sqrt(torch.sum(torch.pow(inputs[-cfg.DIA.MAL_NUM:], 2), dim=1)))  
        # print(f"Amplitude of Normal {amp_normal}")
        outputs_mal = tta_adapter_victim.forward(x_adv).detach().cpu()

        # set victim model and optimizer state to normal adapted state for next trial/batch

        model_state, optimizer_state = copy_model_and_optimizer(tta_adapter.model, tta_adapter.optimizer)
        load_model_and_optimizer(tta_adapter_victim.model, tta_adapter_victim.optimizer, 
                                 model_state, optimizer_state)
        #print(f'Size of the benign sample in batch {n_batch}: {outputs_clean[:-cfg.DIA.MAL_NUM].size()[0]}')
        if outputs_clean[:-cfg.DIA.MAL_NUM].size()[0]:
            normal_accuracy = accuracy(outputs_clean[:-cfg.DIA.MAL_NUM], labels[:-cfg.DIA.MAL_NUM].cpu())
            attacked_accuracy = accuracy(outputs_mal[:-cfg.DIA.MAL_NUM], labels[:-cfg.DIA.MAL_NUM].cpu())
            acc_normal.update(normal_accuracy)
            acc_attacked.update(attacked_accuracy)

    return acc_normal.avg, acc_attacked.avg


def update_configs(cfg,args):
    #gpu_id
    if args.gpu_id:
        cfg.BASE.GPU_ID = args.gpu_id
    #tta method 
    if args.tta:
        cfg.TTA.NAME = args.tta
    # batch size
    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    if args.pseudo:
        cfg.DIA.PSEUDO = args.pseudo
    if args.attack:
        cfg.BASE.ATTACK=args.attack
    return cfg   


if __name__ == "__main__":
    
    set_deterministic(seed=cfg.BASE.SEED, cudnn=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-bs','--batch_size',type=int, help="Specify Batch Size",default=64)
    parser.add_argument('--gpu_id',type=int, help="Specify GPU ID")
    parser.add_argument('--tta',type=str, help="Specify the TTA algorithm to be used")
    parser.add_argument('--mal_num',type=int, help="Specify the number of malicious sample in the batch")
    parser.add_argument('--pseudo',type=bool, help="Specify whether to use pseudo labels or not")
    parser.add_argument('--loss', type=str,help="Specify the loss to be used by u_dia" )
    parser.add_argument('--attack', type=str, help="Specify the attack algorithm" )
    args = parser.parse_args()

    cfg = update_configs(cfg,args)
    device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")

    results = torch.tensor([])
    for snr in range(10,22,2):
        dataloader = get_loader(cfg = cfg,
                    file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                    snr=snr,
                    batch_size = cfg.DATA.BATCH_SIZE,
                    train=False,
                    shuffle=False,
                    num_workers=cfg.BASE.NUM_WORKERS)
        
        model = torch.load(cfg.PATH.SOURCE_MODEL, map_location=torch.device("cpu"))
        #acc = test_tta(model,dataloader,cfg,device)
        # acc = test_normal(model,dataloader,cfg,device)
        # print(f"Accuracy after Adaptation for SNR={snr} --> {acc}")
        acc_normal, acc_attacked = test_attack(model,dataloader,cfg,device)
        print(f"Accuracy after Adaptation for SNR={snr} --> {acc_normal}")
        print(f"Attacked Accuracy for SNR={snr} --> {acc_attacked}")
        x1 = torch.tensor([acc_normal,acc_attacked],device='cpu').reshape(1,2)
        results = torch.cat((results, x1),dim=0)
    # torch.save(results,f'attacked_{cfg.TTA.NAME}_{cfg.DATA.BATCH_SIZE}_{cfg.BASE.ATTACK}.pt')
