import torch
import torch.nn as nn
import numpy as np
import cv2

import glob
import os
from tqdm import tqdm

def video_to_tensor(vid):
    '''
    Convert Video shape (T, H, W, C) to (C, T, H, W)
    '''
    return torch.from_numpy(vid.transpose([3,0,1,2]))

def load_flow_frames(vid_dir, start, num):
    frames = [
        np.asarray([
            cv2.imread(os.path.join(vid_dir, 'x_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE) / 127.5 - 1,
            cv2.imread(os.path.join(vid_dir, 'y_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE) / 127.5 - 1
        ]).transpose([1,2,0]) for i in range(start, start + num)
    ]
    return np.asarray(frames, dtype=np.float32)

def load_rgb_frames(vid_dir, start, num):
    frames = [
        cv2.imread(os.path.join(vid_dir, str(i).zfill(5)+'.jpg'))[:,:,[2,1,0]] / 127.5 - 1
        for i in range(start, start + num)
    ]
    return np.asarray(frames, dtype=np.float32)

def load_image_frames(vid_dir, start, num):
    frames = [
        cv2.imread(os.path.join(vid_dir, str(start+3+(8*i)).zfill(5)+'.jpg'))[:,:,[2,1,0]] / 127.5 - 1
        for i in range(0, (num-1)//8)
    ]
    return np.asarray(frames, dtype=np.float32)

def make_dataset(data_dir, mode, start_i, end_i):
    dataset = []
    # vids : list of '/.../frame/vid_name'
    vid_dirs = glob.glob(os.path.join(data_dir, "*"))[start_i:end_i]

    for vid_dir in tqdm(vid_dirs, desc="load data"):
        num_frames = len(glob.glob(os.path.join(vid_dir, "*")))
        if mode == 'flow':
            num_frames = num_frames // 2
        dataset.append((vid_dir, num_frames))

    return dataset

class TwentyBN(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode, start_i, end_i):
        self.data = make_dataset(data_dir, mode, start_i, end_i)
        self.mode = mode
        self.start_i = start_i
        self.end_i = end_i

    def __getitem__(self, index):
        vid_dir, num_frames = self.data[index]
        vid_name = vid_dir.split('/')[-1]
        if self.mode == 'flow':
            frames = load_flow_frames(vid_dir, 1, num_frames)
        elif self.mode == 'rgb':
            frames = load_rgb_frames(vid_dir, 2, num_frames - 1)
        elif self.mode == 'image':
            frames = load_image_frames(vid_dir, 2, num_frames - 1)

        return video_to_tensor(frames), vid_name

    def __len__(self):
        return len(self.data)
