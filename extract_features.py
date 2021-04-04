import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from networks.resnet import resnet
from networks.pytorch_i3d import InceptionI3d
from data.dataset_20bn import TwentyBN as Dataset

import argparse
import os
import glob
from tqdm import tqdm

from config import cfg

def parse():
    parser = argparse.ArgumentParser(description='Video Feature Extractor')
    parser.add_argument('--mode', type=str, help='rgb, flow or image', default='image')
    parser.add_argument('--replace', type=bool, default=False)
    parser.add_argument('--load_model', type=str, default=cfg['load_model'])
    parser.add_argument('--depth', type=int, default=152)
    parser.add_argument('--save_dir', type=str, default=cfg['save_dir'])
    parser.add_argument('--data_dir', type=str, default=cfg['data_dir'])
    parser.add_argument('--start_i', type=int)
    parser.add_argument('--end_i', type=int)
    args = parser.parse_args()
    return args

def get_model(args):
    if args.mode == 'flow':
        model = InceptionI3d(400, in_channels=2)
        model.load_state_dict(torch.load(args.load_model))
    elif args.mode == 'rgb':
        model = InceptionI3d(400, in_channels=3)
        model.load_state_dict(torch.load(args.load_model))
    elif args.mode == 'image':
        model = resnet(pretrained=True, depth=args.depth)
        model = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError()

    return model


if __name__ == '__main__':
    # parse arguments
    args = parse()

    # prepare dataloader
    dataset = Dataset(data_dir=args.data_dir,
                      mode=args.mode,
                      start_i=args.start_i,
                      end_i=args.end_i)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # setup model
    model = get_model(args)
    model.cuda() # use gpu
    model.eval() # evalutation mode

    # set save path
    save_dir = os.path.join(args.save_dir, args.mode) # .../feature/{mode}
    map_dir = save_dir + '_map'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(map_dir):
        os.mkdir(map_dir)

    # feature extract
    for data in tqdm(dataloader):
        inputs, vid_name = data
        save_name = vid_name[0] + '.npy'

        # check the video was already performed
        if (not args.replace) and os.path.exists(os.path.join(args.save_dir, save_name)):
            continue

        b,c,t,h,w = inputs.shape

        if args.mode == 'image':
            features = []
            for i in range(t):
                with torch.no_grad():
                    frame = Variable(inputs[:,:,i,:,:].cuda())
                    out = model(frame)
                # print(f'[Log] : {out.shape}')
                features.append(out.squeeze().data.cpu().numpy())
            np.save(os.path.join(save_dir, save_name), np.asarray(features))

        else:
            with torch.no_grad():
                inputs = Variable(inputs.cuda())
                map_pool, avg_pool = model.extract_features(inputs)
            # print(f'[Log] : map_pool {map_pool.shape}')
            # print(f'[Log] : avg_pool {avg_pool.shape}')
            np.save(os.path.join(save_dir, save_name), avg_pool.squeeze().permute(-1, 0).data.cpu().numpy())
            # np.save(os.path.join(map_dir, save_name), map_pool.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


