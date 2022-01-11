CUDA_VISIBLE_DEVICES=3 python train.py \
  --num_epochs 16 --batch_size 3 \
  --n_depths 192 --interval_scale 1.06 \
  --optimizer adam --lr 1e-3 --lr_scheduler cosine \
  --num_gpus 2