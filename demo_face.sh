python demo_face.py \
	--name edgetrain_celeb_deepfill_ij_l1_mask_exp \
	--filelist ./static/images/FFHQ_release.txt \
	--model inpaintij2 \
	--netG deepfillc2 \
	--pool_type max \
	--use_cam \
        --which_epoch epoch33_step1160000 \
	--port 9001 \
