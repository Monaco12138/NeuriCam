import argparse
import logging
import os
import utils
import numpy as np
import torch

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

x = np.arange(8*3*2160*3840).reshape(8, 3, 2160, 3840).astype(np.float32)
print(x.shape)

y = x / 255.
print(y.shape)