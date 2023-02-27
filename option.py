import argparse

class BaseOptionParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data
        self.parser.add_argument('-basic_path', type=str, default='/home/ljh20/file/data/surreal/', help='the path of surreal dataset')
        self.parser.add_argument('-data_path', type=str, default='./data/', help='the path of generated data')
        self.parser.add_argument('-m', type=int, default=67, help='the number of markers')

        # network
        self.parser.add_argument('-d_i', type=int, default=3, help='the dimision of input')
        self.parser.add_argument('-d_o', type=int, default=3, help='the dimision of output')
        self.parser.add_argument('-d_model', type=int, default=1024)
        self.parser.add_argument('-n_sample', type=int, default=8, help='the number of sampling point')
        self.parser.add_argument('-n_heads', type=int, default=8)
        self.parser.add_argument('-enc_n_layers', type=int, default=3)
        self.parser.add_argument('-dec_n_layers', type=int, default=8)
        self.parser.add_argument('-dropout', type=float, default=0.1)
        self.parser.add_argument('-n_q', type=int, default=24, help='the number of object queries')
        self.parser.add_argument('-activation', type=str, choices=['relu', 'gelu', 'leakyrelu'], default='gelu')
        self.parser.add_argument('-norm_name', type=str, choices=['bn', 'bn1d', 'id', 'ln'], default='ln')
        self.parser.add_argument('-pct_emb', action='store_true', help='pct embedding or mlp embedding')
        self.parser.add_argument('-pre_norm', action='store_true', help='pre norm or post norm')
        self.parser.add_argument('-mode', type=str, choices=['train', 'test'], default='train')

        # train and val
        self.parser.add_argument('-seed', type=int, default=100, help='the seed for random')
        self.parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size of training')
        self.parser.add_argument('-no_tb', action='store_true', help='do not use tensorboard')
        self.parser.add_argument('-output_path', type=str, default='./experiments/', help='path to save model and log')
        self.parser.add_argument('-exp_name', type=str, default='1_d1024', help='the experiment name to create path')
        self.parser.add_argument('-interval', type=int, default=10, help='epoch interval to save and validation')
        self.parser.add_argument('-resume', action='store_true', help='train from a speicfic epoch')
        self.parser.add_argument('-start_epoch', type=int, default=-1, help='start epoch of resume training')
        self.parser.add_argument('-total_epoch', type=int, default=200)
        self.parser.add_argument('-grad_clip', type=float, default=1.0, help='gradient clip')
        self.parser.add_argument('-rate', type=float, default=0.2)
        self.parser.add_argument('-lambda1', type=float, default=1.0, help='weight of l_d in loss function')
        self.parser.add_argument('-lambda2', type=float, default=5.0, help='weight of l_j in loss function')
        self.parser.add_argument('-lambda3', type=float, default=5.0, help='weight of l_v in loss function')

        # optimizer
        self.parser.add_argument('-optim', type=str, choices=['warmup', 'cosine', 'step'], default='cosine')
        self.parser.add_argument('-base_lr', type=float, default=1e-4)
        self.parser.add_argument('-step_epoch', type=int, default=20)
        self.parser.add_argument('-warmup','--n_warmup_steps', type=int, default=24000)
        self.parser.add_argument('-lr_mul', type=float, default=0.1)
        
        # test
        self.parser.add_argument('-visualize', action='store_true', help='visualize the results or not')
        self.parser.add_argument('-vis_path', type=str, default='./visualization/', help='path to save visualization result')

        # device and distributed
        self.parser.add_argument('-local_rank', type=int, default=-1, help='local rank for parallel')

    def parse_args(self, args_str=None):
        return self.parser.parse_args(args_str)

    def get_parser(self):
        return self.parser

    def save(self, filename):
        argsDict = self.parse_args().__dict__
        with open(filename, 'w') as f:
            f.writelines('---------- parser ---------' + '\n')
            for arg, value in argsDict.items():
                f.writelines(arg + ': ' + str(value) + '\n')
            f.writelines('-----------  end  ---------' + '\n')

    def load(self, filename):
        with open(filename, 'r') as file:
            args_str = file.readline()
        return self.parse_args(args_str.split())