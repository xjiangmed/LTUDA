import argparse
import os.path as osp

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="LTUDA")
        parser.add_argument('--model', type=str, default='unet_proto', help='model_name')
        parser.add_argument("--train_data_dir", type=str, default="../data/Toy_dataset/AbdomenCT-1K/slice_npz/")
        parser.add_argument("--train_path", type=str, default="../data/Toy_dataset/train_slice.txt" )
        parser.add_argument("--val_data_dir", type=str, default="../data/Toy_dataset/AbdomenCT-1K/npz_case/")
        parser.add_argument("--val_path", type=str, default="../data/Toy_dataset/test_volume.txt")
        parser.add_argument('--save_model_path', type=str, default="./checkpoint/CDA_PDA")
        parser.add_argument("--reload_path", type=str, default='../checkpoint/CDA/model_best.pth')
        parser.add_argument("--input_size", type=str, default='256,256')
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_epochs", type=int, default=120) # toy dataset:120 partially labeled dataset:50
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--power", type=float, default=0.9)
        parser.add_argument("--num_prototype", type=int, default=5)
        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    
