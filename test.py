import argparse
import logging
import os
import utils
import numpy as np
import torch
import cv2
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--train_target_dir', default=None,
                    help="Directory containing the train target set")
parser.add_argument('--train_lr_dir', default=None,
                    help="Directory containing the train lr set")
parser.add_argument('--train_key_dir', default=None,
                    help="Directory containing the train key set")
parser.add_argument('--val_target_dir', default=None,
                    help="Directory containing the val target set")
parser.add_argument('--val_lr_dir', default=None,
                    help="Directory containing the val lr set")
parser.add_argument('--val_key_dir', default=None,
                    help="Directory containing the val key set")
parser.add_argument('--file_fmt', default='frame%d.png',
                    help="Dataset file fmt")
parser.add_argument('--model_dir', default='./experiments/bix4_keyvsrc_attn',
                    help="Directory containing params.json")
parser.add_argument('--num_steps', default=None, type=int,
                    help="Number of batches per epoch. Full dataset when set to None.")
parser.add_argument('--eval_batch_size', default=1,
                    help="Batch size for evaluation.")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--gpus', default=None, type=int,
                    help="Number of gpus to use. Default is to use all available.")
parser.add_argument('--no_restore_optim', dest='restore_optim', action='store_false',
                    help="Don't restore optimizer state when restore file is provided.")
parser.set_defaults(restore_optim=True)
parser.add_argument('--restore_only_weights', dest='restore_all', action='store_false',
                    help="Don't restore optimizer state when restore file is provided.")
parser.set_defaults(restore_all=True)

args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
params = utils.Params(json_path)

params.eval_batch_size = args.eval_batch_size

#params.save('./test.json')


# source_path = '/home/ubuntu/data/Dataset/Vid4/GT'

# video_name = 'foliage'

# file_list =  sorted( os.listdir( os.path.join(source_path, video_name) ) )


# target_path = '/home/ubuntu/data/home/main/NeuriCam/data/key-set'
# if not os.path.exists( os.path.join( target_path, video_name) ):
#     os.makedirs( os.path.join( target_path, video_name) )

# for i in range( len(file_list) ):
#     if i % 15 == 0:
#         source_file = os.path.join(source_path, video_name, file_list[i])
#         target_file = os.path.join(target_path, video_name, 'frame{}.png'.format(i) )
#         os.system('cp {} {}'.format(source_file, target_file) )\

# h, w = 3, 4
# y, x = torch.meshgrid(torch.arange(0,h), torch.arange(0,w))

# grid = torch.stack((x,y), 2)
# #print(grid)

# gridx = 2.0 * grid[:,:,0] / max(w, 1) - 1.0
# gridy = 2.0 * grid[:,:,1] / max(h, 1) - 1.0

# gridf = torch.stack( (gridx, gridy), dim=-1)
# print(gridf)
# print(gridf.shape)
# gridf = gridf[None,...]

# x = torch.arange(h*w).reshape(1, 1, h, w).type(torch.float)
# print(x)

# output = F.grid_sample(x, gridf, mode='nearest',padding_mode='zeros')
# print(output)


x = torch.arange(1*5*4*4).reshape(1, 5, 4, 4)
x = torch.unbind( x[:,1:-1,...] )
print(x)
# y = torch.arange(1*3*4*4).reshape(1, 3, 4, 4)
# z = [x] + [y]
# print(z)
# z = torch.cat( z, dim=1)
# print(z.shape)
#print( torch.cat(z, dim=1) ) 
# z = [1,2,3,4]
# print(z[-2])