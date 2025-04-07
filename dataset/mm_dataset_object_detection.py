import os
import json
import numpy as np
import math
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from config.config_mm import Config, parse_args, class_dict
from decord import VideoReader
from decord import cpu
import torchaudio
from transformers import AutoFeatureExtractor
import concurrent.futures
DEBUG_MODE = False

class MMDataset(Dataset):
    def __init__(self, data_path, mode, modal, fps, num_frames, len_feature, sampling, seed=-1, supervision='weak'):
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.deterministic = True
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.benchmark = False

        self.data_path = data_path
        self.mode = mode
        self.fps = fps
        self.num_frames = num_frames
        self.len_feature = len_feature

        # For video processing
        self.transform = self.get_transform(mode, 224)

        # Load ground truth        
        anno_path = os.path.join(data_path, '{}.json'.format(self.mode))
        with open(anno_path, 'r') as f:
            self.anno = json.load(f)

        self.class_name_to_idx = dict((v, k) for k, v in class_dict.items())
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

        ### AUDIO
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        data_video, data_audio, vid_num_frame, sample_idx = self.get_data(index)
        vid_information = self.anno[index]
        vid_name = vid_information['video_url_1'].split('/')[-1].split('-')[0]

        return vid_name, data_video
    

    def process_video(self, video_path, desired_fps):
        """Load video, adjust to desired FPS, resize frames, and return as numpy array."""
        vr = VideoReader(video_path, ctx=cpu(0))
        original_fps = vr.get_avg_fps()
        frame_interval = round(original_fps / desired_fps)
        usable_frame_count = math.ceil(len(vr) / frame_interval)

        frame_indices = range(0, len(vr), frame_interval)
        frames = vr.get_batch(frame_indices).asnumpy()
        transformed_frames = [self.transform(frame) for frame in frames]

        if DEBUG_MODE:
            print(f"Total frames: {len(vr)}")
            print(f"Original FPS: {original_fps}")
            print(f"Frame interval: {frame_interval}")
            print(f"Usable frame count: {usable_frame_count}")

        return torch.stack(transformed_frames), usable_frame_count
    

    def process_video_audio(self, video_path, desired_fps):
        """Load video and audio"""
        vr = VideoReader(video_path, ctx=cpu(0))
        original_fps = vr.get_avg_fps()
        frame_interval = round(original_fps / desired_fps)
        usable_frame_count = math.ceil(len(vr) / frame_interval)

        frame_indices = range(0, len(vr), frame_interval)
        frames = vr.get_batch(frame_indices).asnumpy()
        transformed_frames = [self.transform(frame) for frame in frames]

        if DEBUG_MODE:
            print(f"Total frames: {len(vr)}")
            print(f"Original FPS: {original_fps}")
            print(f"Frame interval: {frame_interval}")
            print(f"Usable frame count: {usable_frame_count}")

        ### FOR AUDIO
        waveform, sample_rate = torchaudio.load(video_path)
        # Ensure the waveform is mono (single channel); convert if necessary
        if waveform.shape[0] > 1:  # Check if there are multiple channels
            waveform = waveform.mean(dim=0)  # Convert to mono by averaging channels
        
        target_sample_rate = 16000
        esampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = esampler(waveform).squeeze()

        segment_length = len(waveform) // usable_frame_count
        audio_segments = [waveform[i * segment_length:(i + 1) * segment_length] for i in range(usable_frame_count)]
        audio_segments = np.stack(audio_segments)
       
        audio_features = self.audio_feature_extractor(audio_segments, sampling_rate=target_sample_rate, return_tensors="pt")['input_values']

        return torch.stack(transformed_frames), audio_features, usable_frame_count

    def get_data(self, index):
        vid_information = self.anno[index]
        vid_num_frame = 0

        # Get all filepath 
        video_paths = [os.path.join(self.data_path, value) for key, value in vid_information.items() if key.startswith('video_url_')]

        # video_paths = [video_paths[0]]

        # Using ThreadPoolExecutor to process videos
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(video_paths)) as executor:
            args = ((path, self.fps) for path in video_paths)
            future_to_video = {executor.submit(self.process_video_audio, *arg): arg[0] for arg in args}
            results = []
            audio_features = []
            frame_counts = []
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_data, audio_feature, frame_count = future.result()
                    results.append(video_data)
                    audio_features.append(audio_feature)
                    frame_counts.append(frame_count)
                    if DEBUG_MODE:
                        print(f"Processed and resized {video_path}")
                except Exception as exc:
                    print(f'{video_path} generated an exception: {exc}')

        vid_num_frame = min(frame_counts)

        if self.sampling == 'random':
            sample_idx = self.random_perturb(vid_num_frame)
        elif self.sampling == 'uniform':
            sample_idx = self.uniform_sampling(vid_num_frame)

        audio_features = [audio_feature[sample_idx] for audio_feature in audio_features]
        audio_features = torch.stack(audio_features)

        results = [result[sample_idx] for result in results]
        combined_video_data = torch.stack(results)
        return combined_video_data, audio_features, vid_num_frame, sample_idx
    

    def get_label(self, index, vid_num_frame, sample_idx):
        vid_information = self.anno[index]
        anno_list = vid_information['tricks']
        label = np.zeros([self.num_classes], dtype=np.float32)

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['labels'][0]]] = 1

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_num_frame, self.num_classes])
            t_factor = self.fps 

            for _anno in anno_list:
                tmp_start_sec = float(_anno['start'])
                tmp_end_sec = float(_anno['end'])

                tmp_start = round(tmp_start_sec * t_factor)
                tmp_end = round(tmp_end_sec * t_factor)

                class_idx = self.class_name_to_idx[_anno['labels'][0]]
                temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]
            return label, torch.from_numpy(temp_anno)


    def random_perturb(self, length):
        if self.num_frames == length:
            return np.arange(self.num_frames).astype(int)
        samples = np.arange(self.num_frames) * length / self.num_frames
        for i in range(self.num_frames):
            if i < self.num_frames - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if self.num_frames == length:
            return np.arange(self.num_frames).astype(int)
        samples = np.arange(self.num_frames) * length / self.num_frames
        samples = np.floor(samples)
        return samples.astype(int)
    
    def get_transform(self, mode, input_size):
        if mode == "train":
            return transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                    ])

        else:
            return transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                    ])
