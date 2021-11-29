import pdb
import cv2
import os
from collections import OrderedDict

import numpy as np
import torch
import data
from options.test_options import TestOptions
import models

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    with torch.no_grad():
        generated, mask = model(data_i, mode='inference')
    mask = (mask*255).cpu().numpy().astype(np.uint8)[:,0]
    generated = (generated+1)/2*255
    generated = generated.cpu().numpy().astype(np.uint8)
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        mm = mask[b]
        output = generated[b]
        path = img_path[b]
        output = output.transpose((1,2,0))
        assert cv2.imwrite(os.path.join(opt.output_dir, path), output[:,:,::-1])
        if hasattr(opt, "output_mask_dir") and opt.output_mask_dir is not None:
            assert cv2.imwrite(os.path.join(opt.output_mask_dir, path), mm)
