CUDA_VISIBLE_DEVICES=5 python active_train.py --backbone mobilenet --lr 0.007 --epochs 150 --batch-size 5 --gpu-ids 0 --checkname evalpa_2-feature_noise_entropy_ep150-abs_60-deeplab-mobilenet-bs_5-512x512-lr_0.007 --eval-interval 5 --dataset active_pascal_image --base-size 512 --crop-size -1 --use-lr-scheduler --lr-scheduler step --active-selection-mode noise_feature --max-iterations 8 --active-batch-size 60 --use-balanced-weights --worker 5 --memory-hog --no-early-stop
CUDA_VISIBLE_DEVICES=5 python active_train.py --backbone mobilenet --lr 0.007 --epochs 150 --batch-size 5 --gpu-ids 0 --checkname evalpa_3-noise_variance_entropy_ep150-abs_60-deeplab-mobilenet-bs_5-512x512-lr_0.007 --eval-interval 5 --dataset active_pascal_image --base-size 512 --crop-size -1 --use-lr-scheduler --lr-scheduler step --active-selection-mode noise_variance --max-iterations 8 --active-batch-size 60 --use-balanced-weights --worker 5 --memory-hog --no-early-stop
