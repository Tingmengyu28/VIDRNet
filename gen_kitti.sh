export CUDA_VISIBLE_DEVICES=1

python gen_kitti.py \
    --model_name vadepthnet \
    --dataset kitti \
    --input_height 352 \
    --input_width 1216 \
    --max_depth 80 \
    --prior_mean 2.54 \
    --do_kb_crop \
    --data_path_eval /home/cxhpc/data/azt/research/CV/Defocus/data/kitti \
    --gt_path_eval ../gt/ \
    --min_depth_eval 1e-3 \
    --max_depth_eval 80 \
    --garg_crop \
    --checkpoint_path /home/cxhpc/data/azt/research/CV/Defocus/VADepthNet/vadepthnet/vadepthnet_eigen.pth
