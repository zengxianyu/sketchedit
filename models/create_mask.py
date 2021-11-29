import cv2
import numpy as np
import random
from PIL import Image
import os
import pdb

class MaskCreator:
    def __init__(self, list_mask_path=None, base_mask_path=None, match_size=False):
        self.match_size = match_size
        if list_mask_path is not None:
            filenames = open(list_mask_path).readlines()
            msk_filenames = list(map(lambda x: os.path.join(base_mask_path, x.strip('\n')), filenames))
            self.msk_filenames = msk_filenames
        else:
            self.msk_filenames = None


    def object_shadow(self, h, w, blur_kernel=7, noise_loc=0.5, noise_range=0.05):
        """
        img: rgb numpy
        return: rgb numpy
        """
        mask = self.object_mask(h, w)
        kernel = np.ones((blur_kernel+3,blur_kernel+3),np.float32)
        expand_mask = cv2.dilate(mask,kernel,iterations = 1)
        noise = np.random.normal(noise_loc, noise_range, mask.shape)
        noise[noise>1] = 1
        mask = mask*noise
        mask = mask + (mask==0)
        kernel = np.ones((blur_kernel,blur_kernel),np.float32)/(blur_kernel*blur_kernel)
        mask = cv2.filter2D(mask,-1,kernel)
        return mask, expand_mask


    def object_mask(self, image_height=256, image_width=256):
        if self.msk_filenames is None:
            raise NotImplementedError
        hb, wb = image_height, image_width
        # object mask as hole
        mask = Image.open(random.choice(self.msk_filenames))
        ## randomly resize
        wm, hm = mask.size
        if self.match_size:
            r = float(min(hb, wb)) / max(wm, hm)
            r = r /2
        else:
            r = 1
        scale = random.gauss(r, 0.5)
        scale = scale if scale > 0.5 else 0.5
        scale = scale if scale < 2 else 2.0
        wm, hm = int(wm*scale), int(hm*scale)
        mask = mask.resize((wm, hm))
        mask = np.array(mask)
        mask = (mask>0)
        if mask.sum() > 0:
            ## crop object region
            col_nz = mask.sum(0)
            row_nz = mask.sum(1)
            col_nz = np.where(col_nz!=0)[0]
            left = col_nz[0]
            right = col_nz[-1]
            row_nz = np.where(row_nz!=0)[0]
            top = row_nz[0]
            bot = row_nz[-1]
            mask = mask[top:bot, left:right]
        else:
            return self.object_mask(image_height, image_width)
        ## place in a random location on the extended canvas
        hm, wm = mask.shape
        canvas = np.zeros((hm+hb, wm+wb))
        y = random.randint(0, hb-1)
        x = random.randint(0, wb-1)
        canvas[y:y+hm, x:x+wm] = mask
        hole = canvas[int(hm/2):int(hm/2)+hb, int(wm/2):int(wm/2)+wb]
        th = 100 if self.match_size else 1000
        if hole.sum() < hb*wb / th:
            return self.object_mask(image_height, image_width)
        else:
            return hole.astype(np.float)

    def rectangle_mask(self, image_height=256, image_width=256, min_hole_size=64, max_hole_size=128):
        mask = np.zeros((image_height, image_width))
        hole_size = random.randint(min_hole_size, max_hole_size)
        hole_size = min(int(image_width*0.8), int(image_height*0.8), hole_size)
        x = random.randint(0, image_width-hole_size-1)
        y = random.randint(0, image_height-hole_size-1)
        mask[x:x+hole_size, y:y+hole_size] = 1
        return mask

    def stroke_mask(self, image_height=256, image_width=256, max_vertex=5, max_mask=5, max_length=128):
        max_angle = np.pi
        max_brush_width = max(1, int(max_length*0.4))
        min_brush_width = max(1, int(max_length*0.1))

        mask = np.zeros((image_height, image_width))
        for k in range(random.randint(1, max_mask)):
            num_vertex = random.randint(1, max_vertex)
            start_x = random.randint(0, image_width-1)
            start_y = random.randint(0, image_height-1)
            for i in range(num_vertex):
                angle = random.uniform(0, max_angle)
                if i % 2 == 0:
                    angle = 2*np.pi - angle
                length = random.uniform(0, max_length)
                brush_width = random.randint(min_brush_width, max_brush_width)
                end_x = min(int(start_x + length * np.cos(angle)), image_width)
                end_y = min(int(start_y + length * np.sin(angle)), image_height)
                mask = cv2.line(mask, (start_x, start_y), (end_x, end_y), color=1, thickness=brush_width)
                start_x, start_y = end_x, end_y
                mask = cv2.circle(mask, (start_x, start_y), int(brush_width/2), 1)
            if random.randint(0, 1):
                mask = mask[:, ::-1].copy()
            if random.randint(0, 1):
                mask = mask[::-1, :].copy()
        return mask


def get_spatial_discount(mask):
    H, W = mask.shape
    shift_up = np.zeros((H, W))
    shift_up[:-1, :] = mask[1:, :]
    shift_left = np.zeros((H, W))
    shift_left[:, :-1] = mask[:, 1:]

    boundary_y = mask - shift_up
    boundary_x = mask - shift_left
    
    boundary_y = np.abs(boundary_y)
    boundary_x = np.abs(boundary_x)
    boundary = boundary_x + boundary_y
    boundary[boundary != 0 ] = 1
#     plt.imshow(boundary)
#     plt.show()
    
    xx, yy = np.meshgrid(range(W), range(H))
    bd_x = xx[boundary==1]
    bd_y = yy[boundary==1]
    dis_x = xx[..., None] - bd_x[None, None, ...]
    dis_y = yy[..., None] - bd_y[None, None, ...]
    dis = np.sqrt(dis_x*dis_x + dis_y*dis_y)
    min_dis = dis.min(2)
    gamma = 0.9
    discount_map = (gamma**min_dis)*mask
    return discount_map




if __name__ == "__main__":
    import os
    from tqdm import tqdm
    #home = os.path.expanduser("~")
    #home = os.environ['SENSEI_USERSPACE_SELF']
    home = '/mnt/ilcompf9d1/user/yuzeng'
    input_file_list = "/mnt/ilcompf0d1/data/Saliency/cheliu_data/datasets/lists/mix_train_clean.txt"
    output_dir = os.path.join(home, "datasets/saliency_sample_mask")
    output_dir_image = os.path.join(home, "results/results_sal/mask/image/0")
    output_mask_list = os.path.join(home, "saliency_mask_list.txt")
    output_image_list = os.path.join(home, "saliency_image_list.txt")
    output_list = []
    mask_creator = MaskCreator(
            "/mnt/ilcompf9d1/user/yuzeng/filelist_object_mask.txt"
            )
    if not os.path.exists(output_dir_image):
        os.mkdir(output_dir_image)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    datalist = open(input_file_list, "r").readlines()
    datalist = list(map(lambda x: x.strip("\n").split(' '), datalist))
    full_datalist = list(map(lambda x: ("/mnt/ilcompf0d1/data/Saliency/cheliu_data/datasets"+x[0], 
        "/mnt/ilcompf0d1/data/Saliency/cheliu_data/datasets"+x[1]), datalist))
    image_list = []
    for imgname, insname in tqdm(full_datalist[:400]):
        img = Image.open(imgname)
        prefix = imgname.split('/')[-1]
        prefix = '.'.join(prefix.split('.')[:-1])
        ## exclude objects
        instances = Image.open(insname)
        instances = np.array(instances)
        instances = (instances>0).astype(np.float)
        w, h = img.size
        max_hole = min(int(w/2), int(h/2))
        #mask = mask_creator.stroke_mask(h, w, max_length=320) if random.randint(0, 1) else mask_creator.rectangle_mask(h, w, max_hole_size=320, min_hole_size=128)
        mask = mask_creator.object_mask(h, w)
        mask = mask*(1-instances)
        input_image = np.array(img)
        if len(input_image.shape) > 2:
            input_image = input_image * (1-mask[:, :, None])
        else:
            input_image = input_image * (1-mask)
        input_image = Image.fromarray(input_image.astype(np.uint8))
        input_image.save('%s/%s.jpg'%(output_dir_image, prefix))
        mask = (mask*255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.save('%s/%s.png'%(output_dir, prefix))
        output_list.append('%s/%s.png'%(output_dir, prefix))
        image_list.append(imgname+'\n')

    output_list = list(map(lambda x: x+'\n', output_list))
    with open(output_mask_list, "w") as f:
        f.writelines(output_list)
    with open(output_image_list, "w") as f:
        f.writelines(image_list)
