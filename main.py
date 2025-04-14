import os
import torch
import random
import numpy as np
import torch.utils.data as data
from utils import misc_utils
import torch.nn.functional as F
from dataset.mm_dataset import MMDataset
from model_factory import ModelFactory
from config.config_mm import Config, parse_args
import torch.nn as nn
from tqdm import tqdm
from utils.loss import TwoWayLoss
import wandb
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "best_model.pkl")
        print(">>> Loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)
        net.load_state_dict(pretrained_params, strict=False)
    else:
        print(">>> Training from scratch")


def get_dataloaders(config):
    # Original dataset instances
    train_dataset = MMDataset(
        data_path=config.data_path, mode='train',
        modal=config.modal, fps=config.fps,
        num_frames=config.num_segments, len_feature=config.len_feature,
        seed=config.seed, sampling='random', supervision='strong'
        )

    test_dataset = MMDataset(
        data_path=config.data_path, mode='test',
        modal=config.modal, fps=config.fps,
        num_frames=config.num_segments, len_feature=config.len_feature,
        seed=config.seed, sampling='uniform', supervision='strong'
        )
    
    print("Length of train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))

    # Calculate the first 20% subset indices for both train and test datasets
    train_subset_indices = list(range(int(len(train_dataset) * 1.0)))
    test_subset_indices = list(range(int(len(test_dataset) * 1.0)))

    # Create subsets of the datasets
    train_subset = Subset(train_dataset, train_subset_indices)
    test_subset = Subset(test_dataset, test_subset_indices)

    # Define data loaders with subsets
    train_loader = data.DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    test_loader = data.DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    ### Print length of train and test loader
    print("Length of train loader: ", len(train_loader))
    print("Length of test loader: ", len(test_loader))
    # return train_loader, val_loader, test_loader
    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


class MMTrainer():
    def __init__(self, config):
        # config
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # network
        self.net = ModelFactory.get_model(config.model_name, config)
        self.net = torch.nn.DataParallel(self.net).to(self.device)

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, 
                                           betas=(0.9, 0.999), weight_decay=0.0005)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs, eta_min=0.00001)
        self.multi_class_criterion = TwoWayLoss()
        self.human_class_criterion = nn.CrossEntropyLoss()

        # parameters
        self.total_loss_per_epoch = 0
        self.best_mAP_multi_class = -1


    def evaluate_mutli_class(self, epoch=0, mode='test'):
        self.net = self.net.eval()

        data_loader = self.test_loader if mode == 'test' else self.train_loader
        with torch.no_grad():
            targets, preds = [], []
            preds_frames, preds_cls_frame = [], []
            for _data, _data_audio, _label, _, _, _human_info in tqdm(data_loader, desc="Evaluating '{}'".format(mode)):
                _data, _data_audio, _label = _data.to(self.device), _data_audio.to(self.device), _label.to(self.device)
                x_cls, _label_frames, _, _ = self.net(_data, _data_audio, _human_info)
                ### apply softmax
                x_cls = F.softmax(x_cls, dim=1)

                targets.append(_label.cpu())
                preds.append(x_cls.cpu())
                ### Frames
                _label_frames = _label_frames.max(dim=1)[0]
                _label_frames = F.softmax(_label_frames, dim=1)

                preds_frames.append(_label_frames.cpu())
                preds_cls_frame.append(((x_cls + _label_frames) / 2).cpu())

            targets = torch.cat(targets).long()
            preds = torch.cat(preds)
            mAP_class = misc_utils.mAP(targets.numpy(), preds.numpy())
            mAP_sample = misc_utils.mAP(targets.t().numpy(), preds.t().numpy())

            preds_frames = torch.cat(preds_frames)
            preds_cls_frame = torch.cat(preds_cls_frame)
            mAP_class_frames = misc_utils.mAP(targets.numpy(), preds_frames.numpy())
            mAP_sample_frames = misc_utils.mAP(targets.t().numpy(), preds_frames.t().numpy())

            mAP_class_cls_frame = misc_utils.mAP(targets.numpy(), preds_cls_frame.numpy())
            mAP_sample_cls_frame = misc_utils.mAP(targets.t().numpy(), preds_cls_frame.t().numpy())

            # WANDB LOG
            print("Mode: {}, Epoch: {}, mAP: {:.5f}, mAP_sample: {:.5f}".format(mode, epoch, mAP_class, mAP_sample))
            print("Mode: {}, Epoch: {}, mAP_frames: {:.5f}, mAP_cls_frame: {:.5f}".format(mode, epoch, mAP_class_frames, mAP_class_cls_frame))
            if mAP_class > self.best_mAP_multi_class and mode == 'test':
                self.best_mAP_multi_class = mAP_class
                print("New best test mAP: ", self.best_mAP_multi_class)
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, 'best_model.pkl'))
            wandb.log({f'{mode}_mAP': mAP_class, f'{mode}_mAP_sample': mAP_sample}, step=epoch)
            wandb.log({f'{mode}_mAP_frames': mAP_class_frames, f'{mode}_mAP_cls_frame': mAP_class_cls_frame}, step=epoch)
            wandb.log({f'{mode}_mAP_sample_frames': mAP_sample_frames, f'{mode}_mAP_sample_cls_frame': mAP_sample_cls_frame}, step=epoch)
        self.net = self.net.train()


    def test(self):
        self.best_mAP_multi_class = 100
        load_weight(self.net, self.config)
        self.evaluate_mutli_class(epoch=0, mode='test')


    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # Training
        for epoch in range(self.config.num_epochs):
            for _data, _data_audio, _label, _label_frames, _, _human_info in tqdm(self.train_loader, desc='Training Epoch: {}'.format(epoch)):
                _data, _data_audio = _data.to(self.device), _data_audio.to(self.device)
                _label, _label_frames, _human_info = _label.to(self.device), _label_frames.to(self.device), _human_info.to(self.device)
                
                self.optimizer.zero_grad()
                # forward pass
                x_cls, x_cls_frames, _, x_human_frames = self.net(_data, _data_audio, _human_info)
                
                loss_human_frames = self.human_class_criterion(x_human_frames.squeeze(-1), _human_info.view(-1, x_human_frames.size(1)))

                ### CLS FRAMES FRAMES
                loss_cls = self.multi_class_criterion(x_cls, _label)
                loss_cls_frames = self.multi_class_criterion(x_cls_frames.reshape(-1, 16), _label_frames.view(-1, 16))
                loss = loss_cls + loss_human_frames + loss_cls_frames

                loss.backward()
                self.optimizer.step()
                self.total_loss_per_epoch += loss.item()
                print("Loss: ", loss.item())

            # Adjust learning rate
            self.scheduler.step()

            # Log train loss
            wandb.log({'train_loss': self.total_loss_per_epoch}, step=epoch)
            self.total_loss_per_epoch = 0

            self.evaluate_mutli_class(epoch=epoch, mode='test')
            if epoch % 10 == 0:
                self.evaluate_mutli_class(epoch=epoch, mode='train')


def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    ### Wandb Initialization
    wandb.login(key='#YOUR_KEY')
    wandb.init(entity="#YOUR_ACCOUNT", 
               project="#YOUR_PROJECT", 
               group=args.model_name,
               name=args.exp_name, 
               config=config, 
               mode=args.wandb)
    
    trainer = MMTrainer(config)
    if args.inference_only:
        trainer.test()
    else:
        trainer.train()

    wandb.finish()

if __name__ == '__main__':
    main()
