import pdb
import cv2
import os
from collections import OrderedDict

import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, url_for, render_template, request, redirect, send_from_directory
from PIL import Image
import base64
import io
import random


from options.test_options import TestOptions
import models
import torch

from mp_landmark import corresponding_points_alignment, mp_landmark

opt = TestOptions().parse()
model = models.create_model(opt)
model.eval()

max_size = 2048
max_num_examples = 200
UPLOAD_FOLDER = 'static/images'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'jpeg', 'bmp'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

port = opt.port
filelist = opt.filelist#"./static/images/example.txt"
with open(filelist, "r") as f:
    list_examples = f.readlines()
list_examples = [n.strip("\n") for n in list_examples]

h_face = 256
w_face = 256
lms_avg = np.load("/home/zeng/celebhq256x256/lms_avg_dense.npy")
c_avg = lms_avg.mean(0)[None,...]
lms_avg_ctr = lms_avg-c_avg
def process_image(img, mask, name, opt, save_to_input=True):
    img =img.convert("RGB")
    image0 = np.array(img)
    img, mask, affine = get_face_image(img, mask)
    if affine is None:
        return None, None
    img_raw = np.array(img)
    w_raw, h_raw = img.size
    h_t, w_t = h_raw//8*8, w_raw//8*8

    img = img.resize((w_t, h_t))
    img = np.array(img).transpose((2,0,1))

    mask_raw = np.array(mask)[...,None]>0
    mask = mask.resize((w_t, h_t))

    mask = np.array(mask)
    mask = (torch.Tensor(mask)>0).float()
    img = (torch.Tensor(img)).float()
    img = (img/255-0.5)/0.5
    img = img[None]
    mask = mask[None,None]

    with torch.no_grad():
        generated,mask_gen = model(
                {'image':img,'mask':mask},
                mode='inference')
    generated = torch.clamp(generated, -1, 1)
    generated = (generated+1)/2*255
    generated = generated.cpu().numpy().astype(np.uint8)
    generated = generated[0].transpose((1,2,0))
    result = generated.astype(np.uint8)

    result = Image.fromarray(result).resize((w_raw, h_raw))
    result = np.array(result)
    mask_gen = mask_gen.cpu().numpy()[0,0]
    mask_gen = Image.fromarray(mask_gen).resize((w_raw, h_raw))
    mask_gen = np.array(mask_gen)

    result = warpback_image(image0, result, mask_gen, affine)
    result = Image.fromarray(result.astype(np.uint8))
    result.save(f"static/results/{name}")
    if save_to_input:
        result.save(f"static/images/{name}")
    w_out, h_out = result.size
    return h_out, w_out

def warpback_image(image0, image, mask_gen, affine):
    h,w,c = image0.shape
    R, T, s = affine
    rot_mat = np.concatenate((R/s,-R/s@T[...,None]),1)
    #mask = np.ones_like(image)
    # Y = X@R*s+T
    # X = (Y-T)@R.T*1/s
    image = cv2.warpAffine(image, rot_mat, (w,h))
    mask = cv2.warpAffine(mask_gen, rot_mat, (w,h))
    mask = mask[...,None]
    #kernel = np.ones((3, 3), np.uint8)
    #mask = cv2.erode(mask, kernel)
    #lms_rot = lms@R.T+T
    image = image*mask+image0*(1-mask)
    return image

def get_face_image(img, mask):
    # align
    mask = np.array(mask)
    img = np.array(img)
    lms = mp_landmark(img)
    if lms is None:
        return None, None, None
    R, T, s = corresponding_points_alignment(lms, lms_avg)
    rot_mat = np.concatenate((R.T*s,T[...,None]),1)
    img = cv2.warpAffine(img, rot_mat, (w_face,h_face))
    mask = cv2.warpAffine(mask, rot_mat, (w_face,h_face))
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    affine = (R, T, s)
    return img, mask, affine


@app.route(f'/{opt.sublabel}', methods=['GET', 'POST'])
def hello(name=None):
    if 'changeim' in request.form:
        idx = request.form['im_idx']
        idx = int(idx)
        idx = (idx+1)%len(list_examples)
        filename = list_examples[idx]
        image = Image.open(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        W, H = image.size
        return render_template('hello.html', name=name, image_name=filename, image_width=W,
                image_height=H,list_examples=list_examples, is_upload=True, idx=idx)
    if request.method == 'POST':
        idx = request.form['im_idx']
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image = Image.open(file)
                W, H = image.size
                if max(W, H) > max_size:
                    ratio = float(max_size) / max(W, H)
                    W = int(W*ratio)
                    H = int(H*ratio)
                    image = image.resize((W, H))
                    filename = "resize_"+filename
                image.save(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                return render_template('hello.html', name=name, image_name=filename, image_width=W,
                        image_height=H,list_examples=list_examples, is_upload=True, idx=idx)
            else:
                filename=list_examples[0]
                image = Image.open(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                W, H = image.size
                return render_template('hello.html', name=name, image_name=filename, image_width=W, image_height=H,
                        is_alert=True,list_examples=list_examples, is_upload=True, idx=idx)
        if 'mask' in request.form:
            filename = request.form['imgname']
            mask_data = request.form['mask']
            mask_data = mask_data.replace('data:image/png;base64,', '')
            mask_data = mask_data.replace(' ', '+')
            mask = base64.b64decode(mask_data)
            maskname = '.'.join(filename.split('.')[:-1]) + '.png'
            maskname = maskname.replace("/","_")
            maskname = "{}_{}".format(random.randint(0, 1000), maskname)
            with open(os.path.join('static/masks', maskname), "wb") as fh:
                fh.write(mask)
            mask = io.BytesIO(mask)
            mask = Image.open(mask).convert("L")
            image = Image.open(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
            W, H = image.size
            list_op = ["result"]
            is_alert = False
            for op in list_op:
                _ho, _wo = process_image(image, mask, f"{op}_"+maskname, op, save_to_input=True)
                if _ho is None:
                    is_alert = True
            return render_template('hello.html', name=name, image_name=filename, #f"{args.opt[0]}_"+maskname
                    mask_name=maskname, image_width=W, image_height=H, list_opt=list_op,list_examples=list_examples,
                    is_upload=True, idx=idx, is_alert=is_alert)
    else:
        filename=list_examples[0]
        image = Image.open(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        W, H = image.size
        return render_template('hello.html', name=name, image_name=filename, image_width=W, image_height=H,
                list_examples=list_examples, is_upload=True, idx=0)



if __name__ == "__main__":

    app.run(host='0.0.0.0', debug=False, port=port, threaded=True)
