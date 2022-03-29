python demo.py \
	--name edgetrain_places_deepfill_ij_l1_mask \
	--filelist ./static/images/general_release.txt \
	--model inpaintij2 \
	--netG deepfillc2 \
	--pool_type max \
	--use_cam \
        --which_epoch epoch1_step11250000 \
	--port 9001 \
