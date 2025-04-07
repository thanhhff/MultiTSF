import os
import torch
import random
import numpy as np
import torch.utils.data as data
from dataset.mm_dataset_object_detection import MMDataset
from model_factory import ModelFactory
from config.config_mm import Config, parse_args
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch.utils.data import Subset

from ultralytics import YOLO
import cv2
# Load the YOLOv10 model
model = YOLO("yolov10b.pt")
# Define the class index for 'person' in COCO dataset
PERSON_CLASS_ID = 0
import json


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
        self.multi_class_criterion = nn.CrossEntropyLoss()

        # parameters
        self.total_loss_per_epoch = 0
        self.best_mAP_multi_class = -1


    def object_detection(self, mode="train"):
        if mode == "train":
            loader = self.train_loader
        else:
            loader = self.test_loader

        human_save_ = {}
        for vid_name_, combined_video_data in tqdm(loader):
            # TODO: PLEASE SELECT bATCH_SIZE = 1
            vid_name_ = vid_name_[0]
            combined_video_data = combined_video_data[0]


            human_list_ = []
            for cam_num, imgs in enumerate(combined_video_data):
                human_list = torch.zeros(len(imgs))
                results = model(imgs)
                for id_img, (img, result) in enumerate(zip(imgs, results)):
                    # save img to "/home/nguyent/MM-Multi/Weekly-MM/object_detection/output"
                    # img = img.permute(1, 2, 0).numpy() * 255
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite("outputs/debug/{}-{}.jpg".format(cam_num, id_img), img)
                    labels = result.boxes.cls.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    human_score = []
                    for label, box, confidence in zip(labels, boxes, confidences):
                        if int(label) == PERSON_CLASS_ID:
                            human_score.append(confidence)
                        # Draw bounding box around detected person
                        # x1, y1, x2, y2 = map(int, box)
                        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Put label and confidence on the bounding box
                        # cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # cv2.imwrite("/home/nguyent/MM-Multi/Weekly-MM/object_detection/predict/{}.jpg".format(id_img), img)
                    human_list[id_img] = 0 if len(human_score) == 0 else torch.tensor(np.mean(human_score))
                human_list_.append(human_list)

            human_list_ = torch.stack(human_list_)
            human_save_[vid_name_] = human_list_
        # Convert tensors to lists
        human_save_serializable = {k: v.tolist() for k, v in human_save_.items()}
        with open(f'human_save_{mode}.json', 'w') as f:
            json.dump(human_save_serializable, f, indent=4)


def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    ### Wandb Initialization
    wandb.login(key='be11a63e61b81765d8f48a9af2ca7cee294be114')
    wandb.init(entity="thanhhff", 
               project="MM-HOME-WACVW", 
               group=args.model_name,
               name=args.exp_name, 
               config=config, 
               mode=args.wandb)
    
    trainer = MMTrainer(config)
    trainer.object_detection("train")
    trainer.object_detection("test")

    wandb.finish()

if __name__ == '__main__':
    main()
