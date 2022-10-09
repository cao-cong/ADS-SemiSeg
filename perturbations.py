import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
from deeplab.tps import sparse_image_warp
import copy
import math

class BasicAffineTransform(object) :
    def __init__(self):
        a1 = 1
        b1 = 1
        a2 = 1
        b2 = 1
        c1 = 0
        c2 = 0
        x_shift = np.random.uniform(-0.25,0.25)
        y_shift = np.random.uniform(-0.25,0.25)
        x_scale = np.random.uniform(0.75,1.25)
        y_scale = np.random.uniform(0.75,1.25)
        angle = np.random.uniform(0,180)
        angle = -angle*math.pi/180

        shift = np.random.randint(0,2)
        scale = np.random.randint(0,2)
        rotation = np.random.randint(0,2)
        transpose = np.random.randint(0,2)

        if rotation==1:
            a1 = a1*math.cos(angle)
            b1 = b1*math.sin(angle)
            a2 = a2*math.sin(-angle)
            b2 = b2*math.cos(angle)
        else:
            if transpose==1:
                a1 = 0
                b2 = 0
            else:
                a2 = 0
                b1 = 0

        if shift==1:
            c1 = c1+x_shift
            c2 = c2+y_shift

        if scale==1:
            a1 = a1*x_scale
            b1 = b1*y_scale
            a2 = a2*x_scale
            b2 = b2*y_scale

        #theta = torch.tensor([
        #    [x_scale,0,x_shift],
        #    [0,y_scale,y_shift]
        #], dtype=torch.float)
        #theta = torch.tensor([
        #    [math.cos(angle)*x_scale,math.sin(-angle),x_shift],
        #    [math.sin(angle),math.cos(angle)*y_scale,y_shift]
        #], dtype=torch.float)
        self.theta = torch.tensor([
            [a1,a2,c1],
            [b1,b2,c2]
        ], dtype=torch.float)
        self.theta = self.theta.unsqueeze(0)
        self.theta = self.theta.cuda()

    def process(self, img_tensor):
        # img_tensor with shape [b, h, w, 3]
        #print(img_tensor.shape)
        theta = self.theta
        theta = theta.repeat(img_tensor.shape[0],1,1)
        #print(theta.shape)
        grid = F.affine_grid(theta, img_tensor.size())
        #print(img_tensor.shape)
        output = F.grid_sample(img_tensor, grid)
        return output

def dgw(image):
    base_locs = [[0,0],[320,320], [122, 155], [202,151], [174,206], [112,240], [191,237]]
    base_locs = np.array(base_locs)
    source_locs = base_locs
    dest_locs = np.minimum(np.maximum(base_locs + np.random.randn(base_locs.shape[0], base_locs.shape[1])*20, 0), 320)
    source_locs = torch.Tensor(source_locs).unsqueeze(0).cuda()
    dest_locs = torch.Tensor(dest_locs).unsqueeze(0).cuda()
    warped_image, _ = sparse_image_warp(image, source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
    return warped_image.permute(0, 3, 1, 2), source_locs, dest_locs


class DGW(object) :
    def __init__(self, num_split=3, img_h=321, img_w=321, src_rand_range = 0.1, dst_rand_range = 0.2, fix_corner=True):

        # num_split: split img into num_split * num_split blocks
        # img_size: input img size
        # src_rand_range, random range of points in src, block_size * src_rand_range, e.g. 320/3*0.1 = 10.6
        # dst_range_range: same
        # fix_corner: whether to fix corner
        self.num_split = num_split
        self.img_h = img_h
        self.img_w = img_w
        self.src_rand_range = src_rand_range
        self.dst_rand_range = dst_rand_range
        self.block_h = int(self.img_h / self.num_split)
        self.block_w = int(self.img_w / self.num_split)
        self.fix_corner = fix_corner

        if fix_corner:
            source_centers = [[0,0], [self.img_h-1, self.img_w-1], [0, self.img_w-1], [self.img_h-1, 0]]
        else:
            source_centers = []
        for idx in range(self.num_split):
            for jdx in range(self.num_split):
                source_centers.append([int((idx+0.5)*self.block_h), int((jdx+0.5)*self.block_w)])
        self.source_centers = source_centers

    def warp(self, img_tensor):
        # img_tensor with shape [b, h, w, 3]
        base_locs = np.array(self.source_centers)
        if self.fix_corner:
            fix_idx = 4
        else:
            fix_idx = 0
        source_locs = base_locs
        source_locs[fix_idx:,0] = np.minimum(np.maximum(base_locs[fix_idx:,0]  + 
                                            np.random.randn(base_locs.shape[0] - fix_idx)*self.block_h*self.src_rand_range, 0), self.img_h-1)
        source_locs[fix_idx:,1] = np.minimum(np.maximum(base_locs[fix_idx:,1]  + 
                                            np.random.randn(base_locs.shape[0] - fix_idx)*self.block_w*self.src_rand_range, 0), self.img_w-1)                
        dest_locs = copy.copy(source_locs)
        dest_locs[fix_idx:,0]  = np.minimum(np.maximum(source_locs[fix_idx:,0]  + 
                                            np.random.randn(base_locs.shape[0] - fix_idx)*self.block_h*self.dst_rand_range, 0), self.img_h-1)
        dest_locs[fix_idx:,1]  = np.minimum(np.maximum(source_locs[fix_idx:,1]  + 
                                            np.random.randn(base_locs.shape[0] - fix_idx)*self.block_w*self.dst_rand_range, 0), self.img_w-1)
        source_locs = torch.Tensor(source_locs).unsqueeze(0).cuda()
        dest_locs = torch.Tensor(dest_locs).unsqueeze(0).cuda()
        warped_image, _ = sparse_image_warp(img_tensor, source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
        return warped_image.permute(0, 3, 1, 2), source_locs, dest_locs

def generate_cutmask(img_size):

    num = np.random.rand()*2+2
    cut_area = img_size[0] * img_size[1] /num

    w = np.random.randint(img_size[1] / num, img_size[1] + 1)
    h = np.round(cut_area / w)
    h = np.minimum(h, img_size[0])

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)

def generate_cowmix_mask(img_size, sigma_min, sigma_max, p):
    # Randomly draw sigma from log-uniform distribution
    sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))
    #p = np.random.uniform(p_min, p_max) # Randomly draw proportion p
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    # Compute threshold
    t = erfinv(p * 2 - 1) * (2 ** 0.5) * Ns.std() + Ns.mean()
    return (Ns > t).astype(float) # Apply threshold and return

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N
