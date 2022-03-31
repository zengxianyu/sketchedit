python demo_mix.py \
	--load_opt_face ./edgetrain_celeb_deepfill_ij_l1_mask_exp.pkl \
	--load_opt_img ./edgetrain_places_deepfill_ij_l1_mask.pkl \
	--filelist ./static/images/release.txt \
	--model inpaintij2 \
	--netG deepfillc2 \
	--pool_type max \
	--use_cam \
	--port 8001 \
