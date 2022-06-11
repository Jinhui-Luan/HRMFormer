import argparse

class BaseOptionParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data
        self.parser.add_argument('--basic_path', type=str, default='/home/ljh20/file/data/surreal/',
                                 help='the path of surreal dataset')

        # network
        self.parser.add_argument('-d_i', type=int, default=3, help='the dimision of input')
        self.parser.add_argument('-d_o', type=int, default=3, help='the dimision of output')
        self.parser.add_argument('-nsample', type=int, default=8, help='the number of sampling point')
        self.parser.add_argument('-d_model', type=int, default=1024)
        self.parser.add_argument('-n_heads', type=int, default=8)
        self.parser.add_argument('-enc_n_layers', type=int, default=3)
        self.parser.add_argument('-dec_n_layers', type=int, default=8)
        self.parser.add_argument('-dropout', type=float, default=0.1)
        self.parser.add_argument('-num_queries', type=int, default=24)
        self.parser.add_argument('-activation', type=str, default='gelu', help='type of activation function')
        self.parser.add_argument('-position_embedding', type=str, default='fourier', help='type of positional embedding')
        self.parser.add_argument('-pre_norm', action='store_true', help='pre norm or post norm')
        self.parser.add_argument('-mode', type=str, choices=['train', 'test'], default='train')

        # train and val
        self.parser.add_argument('-seed', type=int, default=100, help='the seed for random')
        self.parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size of training')
        self.parser.add_argument('-base_lr', type=float, default=1e-5)
        self.parser.add_argument('-step_epoch', type=int, default=10)
        self.parser.add_argument('-total_epoch', type=int, default=200)
        self.parser.add_argument('-use_tb', type=bool, default=True, help='use tensorboard')
        self.parser.add_argument('-output_path', type=str, default='./experiments/', help='path to save model and log')
        self.parser.add_argument('-exp_name', type=str, default='1_d1024', help='the experiment name to create path')
        self.parser.add_argument('-interval', type=int, default=5, help='epoch interval to save and validation')
        self.parser.add_argument('-resume', action='store_true', help='train from a speicfic epoch')
        self.parser.add_argument('-start_epoch', type=int, default=-1, help='start epoch of resume training')
        self.parser.add_argument('-grad_clip', type=float, default=1.0, help='gradient clip')
        
        # test
        self.parser.add_argument('-n_steps', type=int, default=24, help='number of inference steps')
        self.parser.add_argument('-vis_path', type=str, default='./visualization/', 
                                help='path to save visualization result')

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