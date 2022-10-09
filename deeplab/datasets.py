import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from deeplab.tps import sparse_image_warp


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        #print(label)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        #source_locs = [[0,0],[320,320], [122, 155], [202,151], [174,206], [112,240], [191,237]]
        #source_locs = np.array(source_locs)
        #dest_locs = np.minimum(np.maximum(source_locs + np.random.randn(source_locs.shape[0], source_locs.shape[1])*20, 0), 320)
  
        #source_locs = torch.Tensor(source_locs).unsqueeze(0) 
        #dest_locs = torch.Tensor(np.array(dest_locs)).unsqueeze(0) 

        #image_tensor = torch.Tensor(image.astype("float32").transpose((1, 2, 0))).unsqueeze(0) 
        #label_tensor = torch.Tensor(label.astype("float32")).unsqueeze(2).unsqueeze(0)

        #warped_image, _, image_mask = sparse_image_warp(image_tensor, source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
        #warped_label, _, label_mask = sparse_image_warp(label_tensor, source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
        #warped_image = warped_image.squeeze(0).numpy().transpose((2, 0, 1))
        #label_mask = label_mask.squeeze(0).squeeze(0).numpy()
       
        #warped_label = warped_label.squeeze(0).squeeze(2).numpy().astype(np.uint8).astype(np.float32)
        #warped_label[warped_label>20]=255
        #warped_label[label_mask==0]=255
        '''warped_label_expanddim = np.expand_dims(warped_label, 2)
        warped_label_copyconcat = np.concatenate([warped_label_expanddim, warped_label_expanddim, warped_label_expanddim], axis=2)
        warped_label_copyconcat = warped_label_copyconcat.transpose((2, 0, 1))
        image_vis = image + np.expand_dims(np.expand_dims(self.mean, 1), 2)
        image_vis[warped_label_copyconcat==255] = 0
        warped_image_vis = warped_image + np.expand_dims(np.expand_dims(self.mean, 1), 2)
        warped_image_vis[warped_label_copyconcat==255] = 0'''
        #cv2.imwrite("image.png", image.transpose((1, 2, 0)).astype(np.uint8))
        #cv2.imwrite("label.png", label.astype(np.uint8))
        #cv2.imwrite("warped_image.png", warped_image.transpose((1, 2, 0)).astype(np.uint8))
        #cv2.imwrite("warped_label.png", warped_label.astype(np.uint8))
        #cv2.imwrite("image_mask.png", image_mask.astype(np.uint8))
        #cv2.imwrite("label_mask.png", (label_mask*255).astype(np.uint8))

        return image.copy(), label.copy(), np.array(size), name, name
        #return image.copy(), label.copy(), warped_image.copy(), warped_label.copy(), source_locs, dest_locs, np.array(size), name


class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = [] 
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, name, size


if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
