# Main script.

import argparse
import os

from src.model import Model

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train',
                    help='running mode')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--data_path', type=str, help='path of faces')

# optimizer params
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd', 'adagrad', 'rmsprop'])
parser.add_argument('--adam_beta1', type=float, default=0.5,
                    help='value of adam beta 1')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='value of adam beta 2')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')

# training params
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--summary_steps', type=int, default=500,
                    help='summary steps')

# dataset params
parser.add_argument('--image_size', type=int, default=64,
                    help='size of cropped images')
parser.add_argument('--ids', type=int, default=50,
                    help='number of identities for training')

# evaluation dir
parser.add_argument('--log_dir', type=str, help='path of eval checkpoint')
parser.add_argument('--vgg_path', type=str, help='path of vgg model')

# test dir
parser.add_argument('--test_dir', type=str, help='path of test images')

params = parser.parse_args()

model = Model(params)

if params.mode == 'train':
    if not os.path.exists(params.log_dir):
        os.mkdir(params.log_dir)
    else:
        raise FileExistsError("log_dir is not empty!")
    model.train()

elif params.mode == 'eval':
    if not params.log_dir:
        raise ValueError("log_dir is not specified!")
    model.eval()

else:
    raise ValueError("mode must be 'train' or 'eval'")
