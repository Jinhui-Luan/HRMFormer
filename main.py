import numpy as np
import os
import time
import random
from tqdm import tqdm
import IPython
from math import cos, pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel 
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, AdamW
from transformer import Transformer, SMPLModel_torch
from option import BaseOptionParser


class MyDataset(Dataset):
    def __init__(self, marker, theta, beta):
        self.marker = marker
        self.theta = theta
        self.beta = beta

    def __len__(self):
        return len(self.marker)

    def __getitem__(self, index):
        return {
            'marker': self.marker[index],
            'theta': self.theta[index],
            'beta': self.beta[index]
        }


class WarmUpScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.lr_mul * (d_model ** (-0.5)) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class StepScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, base_lr, step_epoch, total_epoch, clip=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_epoch = step_epoch
        self.total_epoch = total_epoch
        self.clip = clip
        self.epoch = 1

    def get_lr(self):
        if self.epoch < self.step_epoch:
            lr = self.base_lr
        elif self.epoch < self.step_epoch * 2:
            lr = self.base_lr / 10
        else:
            lr = self.base_lr / 100

        return lr

    def update_lr(self):
        ''' Learning rate scheduling per step '''
        self.epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CosineScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, base_lr, step_epoch, total_epoch, clip=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_epoch = step_epoch
        self.total_epoch = total_epoch
        self.clip = clip
        self.epoch = 1

    def get_lr(self):
        if self.epoch < self.step_epoch:
            lr = self.base_lr
        else:
            lr = self.clip + 0.5 * (self.base_lr - self.clip) * \
                (1 + cos(pi * ((self.epoch - self.step_epoch) / (self.total_epoch - self.step_epoch))))
        return lr

    def update_lr(self):
        ''' Learning rate scheduling per step '''
        self.epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        

def init_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)
    np.random.seed(seed)
    random.seed(seed)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        # nn.init.constant_(m.bias, 0)

def attach_placeholder(X):
    ''' add <start> and <end> placeholder '''
    placeholder = torch.zeros((X.shape[0], 1, X.shape[2]), dtype=torch.float32, device=X.device, requires_grad=True)
    return torch.cat((placeholder, X), dim=1)


def get_data_loader(basic_path, batch_size, mode, interval):
    data = np.load(os.path.join(basic_path, 'dataset-amass', mode + '.npy'), allow_pickle=True).item()
    print('Successfully load data from ' + mode +  '.npy!')

    marker = torch.Tensor(data['marker'])[::interval].to(torch.float32)       # (f, m, 3)
    theta = torch.Tensor(data['theta'])[::interval].to(torch.float32)         # (f, j, 3)
    beta = torch.Tensor(data['beta'])[::interval].to(torch.float32)           # (f, 10)

    print('Dataset shape: marker with size of {}, theta with size of {}.'.format(
            marker.shape, theta.shape))

    dataset = MyDataset(marker, theta, beta)
    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size, shuffle=False)

    return dataloader


def print_performances(header, loss, mpjpe, mpvpe, lr, start_time):
    print(' - {:12} loss: {:8.6f}, mpjpe: {:8.6f}, mpvpe: {:8.6f}, lr: {:10.8f}, time cost: {:3.2f} min'.format(
          f"({header})", loss, mpjpe, mpvpe, lr, (time.time()-start_time)/60))


def load_checkpoint(model, args, device, start_epoch, scheduler=None):
    if hasattr(model, 'module'):
        model = model.module
    model_path = os.path.join(args.model_save_path, 'model_' + str(start_epoch) + '.chkpt')
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(checkpoint['model'])

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 1

    # load optimizer
    if scheduler is not None:
        assert 'optimizer' in checkpoint
        scheduler.optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.epoch = epoch
        
    return epoch
    

def train(model, dataloader_train, dataloader_val, scheduler, device, args):
    criterion = nn.MSELoss().to(device)
    smpl_model_path = os.path.join(args.basic_path, 'model_m.pkl')   
    smpl_model = SMPLModel_torch(smpl_model_path, device) 

    # train from scratch or continue
    if args.resume:
        start_epoch = load_checkpoint(model, args, device, args.start_epoch, scheduler=scheduler)
        print(' - [Info] Successfully load pretrained model, start epoch =', start_epoch)
    else:
        start_epoch = 1
        model.apply(weight_init)  

    # use tensorboard to plot curves
    if not args.no_tb:
        writer = SummaryWriter(os.path.join(args.log_save_path, 'tensorboard'))

    log_train_file = os.path.join(args.log_save_path, 'train.log')
    log_valid_file = os.path.join(args.log_save_path, 'valid.log')

    print(' - [Info] Training performance will be written to file: {} and {}'.format(
          log_train_file, log_valid_file))

    if not args.resume:
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('{:6}, {:8}, {:8}, {:8}, {:8}\n'.format('epoch', 'loss', 'mpjpe', 'mpvpe', 'lr'))
            log_vf.write('{:6}, {:8}, {:8}, {:8}, {:8}\n'.format('epoch', 'loss', 'mpjpe', 'mpvpe', 'lr'))

    val_metrics = []
    
    for i in range(start_epoch, args.total_epoch):
        print('[ Epoch', i, ']')

        # train epoch
        start = time.time()
        train_loss, train_mpjpe, train_mpvpe = train_epoch(model, smpl_model, dataloader_train, scheduler, criterion, device, args)
        lr = scheduler.optimizer.param_groups[0]['lr']
        print_performances('Training', train_loss, train_mpjpe, train_mpvpe, lr, start)

        # validation epoch
        start = time.time()
        val_loss, val_mpjpe, val_mpvpe = val_epoch(model, smpl_model, dataloader_val, criterion, device, args)
        print_performances('Validation', val_loss, val_mpjpe, val_mpvpe, lr, start)

        val_metric = 0.5 * val_mpjpe + 0.5 * val_mpvpe
        val_metrics.append(val_metric)

        checkpoint = {'epoch': i, 'settings': args, 'model': model.state_dict(), 'optimizer': scheduler.optimizer.state_dict()}

        if val_metric <= min(val_metrics):
            torch.save(checkpoint, os.path.join(args.model_save_path, 'model_best.chkpt'))
            print(' - [Info] The best model file has been updated.')
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(' - [Info] The best model file has been updated.\n')
                log_vf.write(' - [Info] The best model file has been updated.\n')
                
        if i % args.interval == 0:
            torch.save(checkpoint, os.path.join(args.model_save_path, 'model_' + str(i) +'.chkpt'))
            print(' - [Info] The model file has been saved for every {} epochs.'.format(args.interval))

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{:6d}: {:8.6f}, {:8.6f}, {:8.6f}, {:10.8f}\n'.format(i, train_loss, train_mpjpe, train_mpvpe, lr))
            log_vf.write('{:6d}: {:8.6f}, {:8.6f}, {:8.6f}, {:10.8f}\n'.format(i, val_loss, val_mpjpe, val_mpvpe, lr))

        if not args.no_tb:
            writer.add_scalars('Loss',{'Train': train_loss, 'Val': val_loss}, i)
            writer.add_scalars('MPJPE',{'Train': train_mpjpe, 'Val': val_mpjpe}, i)
            writer.add_scalars('MPVPE',{'Train': train_mpvpe, 'Val': val_mpvpe}, i)
            writer.add_scalar('learning_rate', lr, i)


def train_epoch(model, smpl_model, dataloader_train, scheduler, criterion, device, args):
    model.train()
    loss = []
    MPJPE = []
    MPVPE = []

    desc = ' - (Training)   '
    for data in tqdm(dataloader_train, mininterval=2, desc=desc, leave=False, ncols=100):
        marker = data['marker'].to(device)
        theta = data['theta'].to(device)
        beta = data['beta'].to(device)

        scheduler.optimizer.zero_grad()
        theta_pred = model(marker)
        
        smpl_model(beta, theta_pred)
        joint_pred = smpl_model.joints
        vertex_pred = smpl_model.verts

        smpl_model(beta, theta)
        joint = smpl_model.joints
        vertex = smpl_model.verts

        l_data = criterion(theta_pred, theta)
        l_joint = criterion(joint_pred, joint)
        l_vertex = criterion(vertex_pred, vertex)
        l = l_data + l_joint + l_vertex
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scheduler.optimizer.step()

        mpjpe = (joint_pred - joint).pow(2).sum(dim=-1).sqrt().mean()
        mpvpe = (vertex_pred - vertex).pow(2).sum(dim=-1).sqrt().mean()
        
        loss.append(l)
        MPJPE.append(mpjpe.clone().detach())
        MPVPE.append(mpvpe.clone().detach())
    
    scheduler.update_lr()
        
    return torch.Tensor(loss).mean(), torch.Tensor(MPJPE).mean(), torch.Tensor(MPVPE).mean()


def val_epoch(model, smpl_model, dataloader_val, criterion, device, args):
    model.eval()
    loss = []
    MPJPE = []
    MPVPE = []

    desc = ' - (Validation) '
    with torch.no_grad():
        for data in tqdm(dataloader_val, mininterval=2, desc=desc, leave=False, ncols=100):
            marker = data['marker'].to(device)
            theta = data['theta'].to(device)
            beta = data['beta'].to(device)

            theta_pred = model(marker)

            smpl_model(beta, theta_pred)
            joint_pred = smpl_model.joints
            vertex_pred = smpl_model.verts

            smpl_model(beta, theta)
            joint = smpl_model.joints
            vertex = smpl_model.verts

            l_data = criterion(theta_pred, theta)
            l_joint = criterion(joint_pred, joint)
            l_vertex = criterion(vertex_pred, vertex)
            l = l_data + l_joint + l_vertex
            
            mpjpe = (joint_pred - joint).pow(2).sum(dim=-1).sqrt().mean()
            mpvpe = (vertex_pred - vertex).pow(2).sum(dim=-1).sqrt().mean()

            loss.append(l)
            MPJPE.append(mpjpe.clone().detach())
            MPVPE.append(mpvpe.clone().detach())

    return torch.Tensor(loss).mean(), torch.Tensor(MPJPE).mean(), torch.Tensor(MPVPE).mean()
    

def test(model, dataloader_test, device, args):
    criterion = nn.MSELoss().to(device)
    smpl_model_path = os.path.join(args.basic_path, 'model_m.pkl')   
    smpl_model = SMPLModel_torch(smpl_model_path, device) 

    model.eval()
    loss = []
    MPJPE = []
    MPVPE = []

    desc = ' -       (Test) '
    with torch.no_grad():
        for data in tqdm(dataloader_test, mininterval=2, desc=desc, leave=False, ncols=100):
            marker = data['marker'].to(device)
            theta = data['theta'].to(device)
            beta = data['beta'].to(device)

            theta_pred = model(marker)

            smpl_model(beta, theta_pred)
            joint_pred = smpl_model.joints
            vertex_pred = smpl_model.verts

            smpl_model(beta, theta)
            joint = smpl_model.joints
            vertex = smpl_model.verts

            l_data = criterion(theta_pred, theta)
            l_joint = criterion(joint_pred, joint)
            l_vertex = criterion(vertex_pred, vertex)
            l = l_data + l_joint + l_vertex
            
            mpjpe = (joint_pred - joint).pow(2).sum(dim=-1).sqrt().mean()
            mpvpe = (vertex_pred - vertex).pow(2).sum(dim=-1).sqrt().mean()

            loss.append(l)
            MPJPE.append(mpjpe.clone().detach())
            MPVPE.append(mpvpe.clone().detach())

    return torch.Tensor(loss).mean(), torch.Tensor(MPJPE).mean(), torch.Tensor(MPVPE).mean()


def main():
    # get parser of option.py
    parser = BaseOptionParser()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.d_h3 = args.d_model // 2
    args.d_h2 = args.d_h3 // 2
    args.d_h1 = args.d_h2 // 4
    args.d_ffn = args.d_model * 2

    model = Transformer(args).to(device)

    args.exp_path = os.path.join(args.output_path, args.exp_name)
    args.log_save_path = os.path.join(args.exp_path, 'logs')
    os.makedirs(args.log_save_path, exist_ok=True)
    args.model_save_path = os.path.join(args.exp_path, 'models')
    os.makedirs(args.model_save_path, exist_ok=True)

    # train mode
    if args.mode == 'train':
        # initialization
        if args.seed is not None:
            init_random_seed(args.seed)
            
        if not args.output_path:
            print('No experiment result will be saved.')
            raise

        print(args)
        parser.save(os.path.join(args.exp_path, 'parameters.txt'))
        with open(os.path.join(args.exp_path, 'parameters.txt'), 'a') as f:
            f.writelines('---------- model ---------' + '\n')
            f.write(str(model))
            f.writelines('----------- end ----------' + '\n')
        
        dl_train = get_data_loader(args.basic_path, args.batch_size, 'train', 20)
        dl_val = get_data_loader(args.basic_path, args.batch_size, 'val', 1)

        # create optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.base_lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = StepScheduledOptim(optimizer, args.base_lr, args.step_epoch, args.total_epoch)
        
        # training
        train(model, dl_train, dl_val, scheduler, device, args)
    elif args.mode == 'test':
        model_path = os.path.join(args.model_save_path, 'model_best.chkpt')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        dl_test = get_data_loader(args.basic_path, args.batch_size, 'test', 20)
        loss, mpjpe, mpvpe = test(model, dl_test, device, args)


if __name__ == '__main__':
    ''' 
    Usage:
    CUDA_VISIBLE_DEVICES='0' python main.py -mode 'train' -warmup 24000 -exp_name '1_d1024' -d_model 1024
    '''
    main()
    
