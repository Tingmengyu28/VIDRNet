import torch
import torch.backends.cudnn as cudnn
import os, sys
import argparse
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image
from utils.datasets import KITTIGenDataset
from VADepthNet.vadepthnet.networks.vadepthnet import VADepthNet

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='VADepthNet PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='vadepthnet')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--prior_mean',                type=float, help='prior mean of depth', default=1.54)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Eval
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = f'@{sys.argv[1]}'
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

def eval(model, dataloader_eval, save_path, post_process=False):
    for eval_sample_batched, eval_sample_batched_trans, depth_name_batched in tqdm(iter(dataloader_eval)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched_trans.cuda())

            if image.ndimension() == 3:
                image = image.unsqueeze(0)

            pred_depth = model(image)

            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[torch.isnan(pred_depth)] = args.min_depth_eval
            pred_depth[torch.isinf(pred_depth)] = args.max_depth_eval
            
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            
            save_image = Image.fromarray(eval_sample_batched)
            save_image.save(f"{os.path.join(save_path, str(depth_name_batched + 1))}.png")
            np.save(f"{os.path.join(save_path, str(depth_name_batched + 1))}.npy", pred_depth.squeeze().cpu().numpy())


def main_worker(args):

    model = VADepthNet(max_depth=args.max_depth, 
                       prior_mean=args.prior_mean, 
                       img_size=(args.input_height, args.input_width))

    model.train()

    num_params = sum(np.prod(p.size()) for p in model.parameters())
    print(f"== Total number of parameters: {num_params}")

    num_params_update = sum(
        np.prod(p.shape) for p in model.parameters() if p.requires_grad
    )
    print(f"== Total number of learning parameters: {num_params_update}")

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print(f"== Loading checkpoint '{args.checkpoint_path}'")
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print(f"== Loaded checkpoint '{args.checkpoint_path}'")
            del checkpoint
        else:
            print(f"== No checkpoint found at '{args.checkpoint_path}'")

    cudnn.benchmark = True

    for mode in ['train', 'val', 'test']:
        dataloader_eval = KITTIGenDataset("/home/cxhpc/data/azt/research/CV/Defocus/data/kitti/kitti_dp",
                                          "kitti_dp_alltrain.json",
                                          mode=mode)
        # ===== Evaluation ======
        model.eval()
        with torch.no_grad():
            eval(model, 
                dataloader_eval, 
                save_path=os.path.join("/home/cxhpc/data/azt/research/CV/Defocus/data/kitti/kitti_pro", mode), 
                post_process=False)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
