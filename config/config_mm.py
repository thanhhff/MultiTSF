import numpy as np
import argparse
import shutil
import os


def parse_args():
    description = 'Weakly supervised action localization'
    parser = argparse.ArgumentParser(description=description)

    # dataset parameters
    parser.add_argument('--data_path', type=str, default='data/mm')
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
    parser.add_argument('--output_dir', type=str, default='./outputs')

    # data parameters
    parser.add_argument('--modal', type=str, default='all', choices=['rgb', 'flow', 'all'])
    parser.add_argument('--num_segments', default=50, type=int)
    parser.add_argument('--scale', default=1, type=int)
    
    # model parameters
    parser.add_argument('--model_name', required=True, type=str, help="Which model to use")
    parser.add_argument('--video_encoder', type=str, default='resnet18', help='video encoder model')
    parser.add_argument('--len_feature', type=int, default=512, help='length of feature')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--q_val', default=0.7, type=float)
    parser.add_argument('--fusion_type', type=str, default='max', help='fusion type for device level features')

    # inference parameters
    parser.add_argument('--inference_only', action='store_true', default=False)
    parser.add_argument('--class_th', type=float, default=0.25)
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')
    parser.add_argument('--gamma', type=float, default=0.2, help='Gamma for oic class confidence')
    parser.add_argument('--soft_nms', default=False, action='store_true')
    parser.add_argument('--nms_alpha', default=0.35, type=float)
    parser.add_argument('--nms_thresh', default=0.4, type=float)
    parser.add_argument('--load_weight', default=False, action='store_true')
    
    # system parameters
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42, help='random seed (-1 for no manual seed)')
    parser.add_argument('--verbose', default=False, action='store_true')

    # wandb
    parser.add_argument('--wandb', type=str, default="disabled")
    
    return init_args(parser.parse_args())


def init_args(args):

    args.model_path = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.model_path = os.path.join(args.output_dir, args.exp_name)

    return args

class_dict = {
    0: 'AdjustAC',
    1: 'Clean',
    2: 'CleanVacuum',
    3: 'CloseCurtain',
    4: 'Drink',
    5: 'Eat',
    6: 'Enter',
    7: 'Exit',
    8: 'OpenCurtain',
    9: 'ReadBook',
    10: 'Sitdown',
    11: 'Standup',
    12: 'TurnOffLamp',
    13: 'TurnOnLamp',
    14: 'UseLaptop',
    15: 'UsePhone',
}

class Config(object):
    def __init__(self, args):
        self.lr = args.lr
        self.num_classes = len(class_dict)
        self.modal = args.modal
        self.video_encoder = args.video_encoder
        self.len_feature = args.len_feature
        self.fusion_type = args.fusion_type
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.model_path = os.path.join(args.output_dir, args.exp_name)
        self.num_workers = args.num_workers
        self.class_thresh = args.class_th
        self.act_thresh = np.arange(0.1, 1.0, 0.1)
        self.q_val = args.q_val
        self.scale = args.scale
        self.gt_path = os.path.join(self.data_path, 'gt.json')
        self.model_file = args.model_file
        self.seed = args.seed
        self.fps = 2.5
        self.num_segments = args.num_segments
        self.num_epochs = args.num_epochs
        self.gamma = args.gamma
        self.inference_only = args.inference_only
        self.model_name = args.model_name
        self.soft_nms = args.soft_nms
        self.nms_alpha = args.nms_alpha
        self.nms_thresh = args.nms_thresh
        self.load_weight = args.load_weight
        self.verbose = args.verbose



