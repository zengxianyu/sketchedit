import torchvision.transforms as transforms
import torch
from data.base_dataset import get_params, get_transform, BaseDataset
from PIL import Image
import os
import pdb


class TestImageDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--image_dirs', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--mask_dirs', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--image_postfix', type=str, default=".jpg",
                            help='path to the directory that contains photo images')
        parser.add_argument('--mask_postfix', type=str, default=".png",
                            help='path to the directory that contains photo images')
        parser.add_argument('--image_lists', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--output_labels', type=str, required=False)
        parser.add_argument('--output_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--output_mask_dir', type=str, required=False,
                            help='path to the directory that contains photo images')
        return parser

    def initialize(self, opt):
        self.opt = opt
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        if opt.output_mask_dir is not None:
            if not os.path.exists(opt.output_mask_dir):
                os.makedirs(opt.output_mask_dir)


        image_paths, mask_paths, output_paths = self.get_paths(opt)

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_paths = output_paths

        size = len(self.image_paths)
        self.dataset_size = size
        transform_list = [
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                ]
        self.image_transform = transforms.Compose(transform_list)
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def get_paths(self, opt):
        image_dirs = opt.image_dirs.split(";")
        mask_dirs = opt.mask_dirs.split(";")
        image_lists = opt.image_lists.split(";")
        if opt.output_labels is not None:
            labels = opt.output_labels.split(";")
        else:
            labels = None

        image_paths = []
        mask_paths = []
        img_names = []
        output_paths = []
        for i, image_list in enumerate(image_lists):
            with open(image_list, "r") as f:
                names = f.readlines()
            filenames = list(map(lambda x: x.strip('\n').replace(opt.image_postfix, ""), names))
            image_paths += list(map(lambda x: os.path.join(image_dirs[i], x+opt.image_postfix), filenames))
            mask_paths += list(map(lambda x: os.path.join(mask_dirs[i], x+opt.mask_postfix), filenames))
            if labels is not None:
                output_paths += list(map(lambda x: labels[i]+"_"+x+opt.image_postfix, filenames))
            else:
                output_paths += list(map(lambda x: x+opt.image_postfix, filenames))

        return image_paths, mask_paths, output_paths

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        output_path = self.output_paths[index]
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        w, h = image.size
        image_tensor = self.image_transform(image)
        # mask image
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask = mask.convert("L")
        mask = mask.resize((w,h))
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor>0).float()
        input_dict = {
                      'image': image_tensor,
                      'gt': image_tensor,
                      'mask': mask_tensor,
                      'path': output_path,
                      }

        return input_dict
