python test.py \
	--batchSize 1 \
	--nThreads 1 \
	--name places \
	--joint_train_inp \
	--dataset_mode testimage \
	--image_dirs /home/zeng/data/datasets/inpainting/places2samples1k/places2samples1k_crop256_png \
	--mask_dirs /home/zeng/data/editline_data/places1k/edge_9 \
	--image_lists /home/zeng/data/datasets/inpainting/places2samples1k/imagelist.txt \
	--image_postfix .png \
	--mask_postfix .png \
	--model editline2 \
	--netG deepfillc2 \
	--pool_type max \
	--use_cam \
        --which_epoch latest \
	--output_dir /home/zeng/data/editline_data/places1k/results/9 \
	#--output_mask_dir /home/zeng/data/editline_results/v2/placessketch512/v2_masks \
