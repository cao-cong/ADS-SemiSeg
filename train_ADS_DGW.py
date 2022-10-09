import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from deeplab.model import Res_Deeplab
from deeplab.discriminator import discriminator
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import VOCDataSet
#import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()
import copy
import ramps
from deeplab.tps import sparse_image_warp
#from tensorboardX import SummaryWriter
from data import get_loader, get_data_path
from data.augmentations import *
from perturbations import DGW

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
DATASET = 'pascal_voc' #pascal_voc or pascal_context
BATCH_SIZE = 5
DATA_DIRECTORY = '../data/VOC2012/VOCdevkit/VOC2012'
DATA_LIST_PATH = './dataset/list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_COCO_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './output/snapshots/'
WEIGHT_DECAY = 0.0005

SPLIT_ID = './splits/voc/split_0.pkl'
LABELED_RATIO= None # use 100% labeled data 

LAMBDA_FM = 0.1
LAMBDA_ST = 1.0
THRESHOLD_ST = 0.6 # 0.6 for PASCAL-VOC/Context / 0.7 for Cityscapes

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset to be used")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D-l", type=str, default=None,
                        help="Where restore model D_l parameters from.")
    parser.add_argument("--restore-from-D-r", type=str, default=None,
                        help="Where restore model D_r parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--split-id", type=str, default=SPLIT_ID,
                        help="name of split pickle file")
    parser.add_argument("--labeled-ratio", type=float, default=LABELED_RATIO,
                        help="ratio of labeled samples/total samples")
    parser.add_argument('--consistency_scale', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--stabilization_scale', default=None, type=float, metavar='WEIGHT',
                        help='use stabilization loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--stabilization-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the stabilization loss ramp-up')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--stable-threshold', default=0.0, type=float, metavar='THRESHOLD',
                        help='threshold for stable sample')
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="lambda_fm for feature-matching loss.")
    parser.add_argument("--lambda-st", type=float, default=LAMBDA_ST,
                        help="lambda_st for self-training.")
    parser.add_argument("--threshold-st", type=float, default=THRESHOLD_ST,
                        help="threshold_st for the self-training threshold.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
            
def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_model(num_classes, restore_from, ema):

    model = Res_Deeplab(num_classes=num_classes)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    saved_state_dict = torch.load(restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if not i_parts[1]=='layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
    #model.float()
    #model.eval() # use_global_stats = True
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.train()
    model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output

def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 

criterion = nn.BCELoss()

def main():
    """Create the model and start the training."""
    
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    #writer = SummaryWriter('./logs')

    # Create network.
    l_model = create_model(num_classes=args.num_classes, restore_from=args.restore_from, ema=False)
    r_model = create_model(num_classes=args.num_classes, restore_from=args.restore_from, ema=False)
    
    cudnn.benchmark = True

    # init D_l
    model_D_l = discriminator(num_classes=args.num_classes, dataset='pascal_voc')

    if args.restore_from_D_l is not None:
        model_D_l.load_state_dict(torch.load(args.restore_from_D_l))

    #model_D = torch.nn.DataParallel(model_D, device_ids=[0, 1])
    #cudnn.benchmark = True    
    
    model_D_l.train()
    model_D_l.cuda()

    # init D_r
    model_D_r = discriminator(num_classes=args.num_classes, dataset='pascal_voc')

    if args.restore_from_D_r is not None:
        model_D_r.load_state_dict(torch.load(args.restore_from_D_r))

    #model_D = torch.nn.DataParallel(model_D, device_ids=[0, 1])
    #cudnn.benchmark = True    
    
    model_D_r.train()
    model_D_r.cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # Create dataloader
    if args.dataset == 'pascal_voc':
        train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    elif args.dataset == 'pascal_context':
        input_transform = transform.Compose([transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        data_kwargs = {'transform': input_transform, 'base_size': 505, 'crop_size': 321}
        #train_dataset = get_segmentation_dataset('pcontext', split='train', mode='train', **data_kwargs)
        data_loader = get_loader('pascal_context')
        data_path = get_data_path('pascal_context') 
        train_dataset = data_loader(data_path, split='train', mode='train', **data_kwargs)
    elif args.dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        data_aug = Compose([RandomCrop_city((input_size[0], input_size[1])), RandomHorizontallyFlip()])
        train_dataset = data_loader( data_path, is_transform=True, img_size=(input_size[0], input_size[1]), augmentations=data_aug) 

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    partial_size = int(args.labeled_ratio * train_dataset_size)

    if args.split_id is not None:
        #print('---------------------')
        train_ids = pickle.load(open(args.split_id, 'rb'))
        print('loading train ids from {}'.format(args.split_id))
    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    pickle.dump(train_ids, open(args.snapshot_dir + 'split.pkl', 'wb'))

    train_sampler_supervised = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    if args.labeled_ratio==1:
        train_sampler_unsupervised = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    else:
        train_sampler_unsupervised = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
    train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

    trainloader_supervised = data.DataLoader(train_dataset,
                    batch_size=args.batch_size, sampler=train_sampler_supervised, num_workers=5, pin_memory=True)
    trainloader_unsupervised = data.DataLoader(train_dataset,
                    batch_size=args.batch_size, sampler=train_sampler_unsupervised, num_workers=5, pin_memory=True)
    trainloader_gt = data.DataLoader(train_dataset,
                    batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=5, pin_memory=True)

    trainloader_iter_supervised = iter(trainloader_supervised)
    trainloader_iter_unsupervised = iter(trainloader_unsupervised)
    trainloader_gt_iter = iter(trainloader_gt)
    
    # Create optimizers
    l_optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(l_model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(l_model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    l_optimizer.zero_grad()
    
    r_optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(r_model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(r_model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    r_optimizer.zero_grad()
    
    # optimizer for discriminator network
    optimizer_D_l = optim.Adam(model_D_l.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D_l.zero_grad()
    
    optimizer_D_r = optim.Adam(model_D_r.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D_r.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    y_real_, y_fake_ = Variable(torch.ones(args.batch_size, 1).cuda()), Variable(torch.zeros(args.batch_size, 1).cuda())

    i_iter=0
    while i_iter < args.num_steps:
 
        l_optimizer.zero_grad()
        adjust_learning_rate(l_optimizer, i_iter)
        r_optimizer.zero_grad()
        adjust_learning_rate(r_optimizer, i_iter)
        optimizer_D_l.zero_grad()
        adjust_learning_rate_D(optimizer_D_l, i_iter) 
        optimizer_D_r.zero_grad()
        adjust_learning_rate_D(optimizer_D_r, i_iter)

        # don't accumulate grads in D
        for param in model_D_l.parameters():
            param.requires_grad = False
        for param in model_D_r.parameters():
            param.requires_grad = False

        # When next() end, will raise StopIteration. Use try ... except ... to solve this problem.
        try:
            batch_supervised = next(trainloader_iter_supervised)
        except:
            trainloader_iter_supervised = iter(trainloader_supervised)
            batch_supervised = next(trainloader_iter_supervised)

        try:
            batch_unsupervised = next(trainloader_iter_unsupervised)
        except:
            trainloader_iter_unsupervised = iter(trainloader_unsupervised)
            batch_unsupervised = next(trainloader_iter_unsupervised)

        images, labels, _, _, _  = batch_supervised
        images_un, _, _, _, _ = batch_unsupervised
        
        images = images.cuda()
        images_un = images_un.cuda()

        #try:
        #    tps_aug = DGW(img_h=input_size[0], img_w=input_size[1])
        #    images_un_tps, source_locs, dest_locs  = tps_aug.warp(images_un.permute(0, 2, 3, 1))
        #except Exception as e:
        #    print('--->',e)
        #    continue
        tps_aug = DGW(img_h=input_size[0], img_w=input_size[1])
        images_un_tps, source_locs, dest_locs  = tps_aug.warp(images_un.permute(0, 2, 3, 1))

        images_var = Variable(images)
        #labels_var = Variable(labels.cuda(async=True))
        images_un_var = Variable(images_un)
        #images_un_var_e = Variable(images_un, requires_grad=False, volatile=True)
        images_un_tps_var = Variable(images_un_tps)
        #images_un_tps_var_e = Variable(images_un, requires_grad=False, volatile=True)
        
        # forward
        l_model_out = interp(l_model(images_var))
        r_model_out = interp(r_model(images_var))
        l_model_un_out = interp(l_model(images_un_var))
        r_model_un_out = interp(r_model(images_un_var))
        l_model_un_tps_out = interp(l_model(images_un_tps_var))
        r_model_un_tps_out = interp(r_model(images_un_tps_var))

        # ce loss
        l_ce_loss = loss_calc(l_model_out, labels)
        r_ce_loss = loss_calc(r_model_out, labels)

        l_loss = l_ce_loss
        r_loss = r_ce_loss
        
        # consistency loss
        consistency_weight = args.consistency_scale * ramps.sigmoid_rampup(i_iter, args.consistency_rampup)
        #print(l_model_un_out.detach().cpu().numpy().shape)
        l_model_un_out_tps, _ = sparse_image_warp(l_model_un_out.permute(0,2,3,1), source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
        #print(l_model_un_out_tps.detach().cpu().numpy().shape)
        l_model_un_out_tps = l_model_un_out_tps.permute(0,3,1,2)
        l_model_un_out_tps = Variable(l_model_un_out_tps.detach().data, requires_grad=False)
        #print(l_model_un_tps_out.detach().cpu().numpy().shape)
        #print(l_model_un_out_tps.detach().cpu().numpy().shape)
        l_consistency_loss = consistency_weight * torch.pow(torch.abs(F.softmax(l_model_un_tps_out, dim=1)-F.softmax(l_model_un_out_tps, dim=1)), 2).mean()       

        r_model_un_out_tps, _ = sparse_image_warp(r_model_un_out.permute(0,2,3,1), source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
        r_model_un_out_tps = r_model_un_out_tps.permute(0,3,1,2)
        r_model_un_out_tps = Variable(r_model_un_out_tps.detach().data, requires_grad=False)
        r_consistency_loss = consistency_weight * torch.pow(torch.abs(F.softmax(r_model_un_tps_out, dim=1)-F.softmax(r_model_un_out_tps, dim=1)), 2).mean() 
        
        l_loss += l_consistency_loss
        r_loss += r_consistency_loss

        # stabilization loss
        # value (cls_v) and index (cls_i) of the max probability in the prediction
        l_cls_v_map, l_cls_i_map = torch.max(F.softmax(l_model_un_tps_out, dim=1), dim=1)
        r_cls_v_map, r_cls_i_map = torch.max(F.softmax(r_model_un_tps_out, dim=1), dim=1)
        le_cls_v_map, le_cls_i_map = torch.max(F.softmax(l_model_un_out_tps, dim=1), dim=1)
        re_cls_v_map, re_cls_i_map = torch.max(F.softmax(r_model_un_out_tps, dim=1), dim=1)

        l_cls_i_map = l_cls_i_map.data.cpu().numpy()
        r_cls_i_map = r_cls_i_map.data.cpu().numpy()
        le_cls_i_map = le_cls_i_map.data.cpu().numpy()
        re_cls_i_map = re_cls_i_map.data.cpu().numpy()

        # stable prediction mask 
        l_mask = (l_cls_v_map > args.stable_threshold).data.cpu().numpy()
        r_mask = (r_cls_v_map > args.stable_threshold).data.cpu().numpy()
        le_mask = (le_cls_v_map > args.stable_threshold).data.cpu().numpy()
        re_mask = (re_cls_v_map > args.stable_threshold).data.cpu().numpy()
        
        # detach logit -> for generating stablilization target 
        in_r_cons_logit = Variable(r_model_un_tps_out.detach().data, requires_grad=False)
        tar_l_class_logit = Variable(l_model_un_tps_out.clone().detach().data, requires_grad=False)

        in_l_cons_logit = Variable(l_model_un_tps_out.detach().data, requires_grad=False)
        tar_r_class_logit = Variable(r_model_un_tps_out.clone().detach().data, requires_grad=False)
      
        # generate target for each sample
        for sdx in range(0, args.batch_size):
            shape = tar_l_class_logit.cpu().detach().numpy().shape
            l_stable_map = np.ones((shape[2], shape[3]))
            # unstable: do not satisfy the 2nd condition
            #print(l_mask[sdx] == 0 and le_mask[sdx] == 0)
            #print(sdx)
            #print(l_mask.shape)
            #print(le_mask.shape)
            if sdx>=l_mask.shape[0] or sdx>=le_mask.shape[0]:
                break
            index = (l_mask[sdx] == 0).astype(np.uint8)*(le_mask[sdx] == 0).astype(np.uint8) == 1   
            #l_stable_map[l_mask[sdx] == 0 and le_mask[sdx] == 0] = 0
            l_stable_map[index] = 0
            #print((l_stable_map==0).shape)
            #print(tar_l_class_logit[sdx, l_stable_map==0].detach().cpu().numpy().shape)
            #print(in_r_cons_logit[sdx, l_stable_map==0].detach().cpu().numpy().shape)
            tar_l_class_logit[sdx, :, l_stable_map==0] = in_r_cons_logit[sdx, :, l_stable_map==0]
            # unstable: do not satisfy the 1st condition
            index = l_cls_i_map[sdx] != le_cls_i_map[sdx]
            #l_stable_map[l_mask[sdx] == 0 and le_mask[sdx] == 0] = 2
            l_stable_map[index] = 2
            tar_l_class_logit[sdx, :, l_stable_map==2] = in_r_cons_logit[sdx, :, l_stable_map==2]
            l_stable_map[l_stable_map==2]=0
            
            shape = tar_r_class_logit.cpu().detach().numpy().shape
            r_stable_map = np.ones((shape[2], shape[3]))
            # unstable: do not satisfy the 2nd condition
            index = (r_mask[sdx] == 0).astype(np.uint8)*(re_mask[sdx] == 0).astype(np.uint8) == 1
            #r_stable_map[r_mask[sdx] == 0 and re_mask[sdx] == 0] = 0
            r_stable_map[index] = 0
            tar_r_class_logit[sdx, :, r_stable_map==0] = in_l_cons_logit[sdx, :, r_stable_map==0]
            # unstable: do not satisfy the 1st condition
            index = r_cls_i_map[sdx] != re_cls_i_map[sdx]
            #r_stable_map[r_mask[sdx] == 0 and re_mask[sdx] == 0] = 2
            r_stable_map[index] = 2
            tar_r_class_logit[sdx, :, r_stable_map==2] = in_l_cons_logit[sdx, :, r_stable_map==2]
            r_stable_map[r_stable_map==2]=0

            # calculate stability if both models are stable for a sample
            # compare by consistency
            l_sample_cons = torch.mean(torch.pow(torch.abs(l_model_un_tps_out[sdx:sdx+1, ...]-l_model_un_out_tps[sdx:sdx+1, ...]), 2), dim=1)
            r_sample_cons = torch.mean(torch.pow(torch.abs(r_model_un_tps_out[sdx:sdx+1, ...]-r_model_un_out_tps[sdx:sdx+1, ...]), 2), dim=1)
            judge1 = (l_stable_map==1).astype(np.uint8)*(r_stable_map==1).astype(np.uint8)*(l_sample_cons.data.cpu().numpy()[0] < r_sample_cons.data.cpu().numpy()[0]) == 1
            # loss: l -> r
            tar_r_class_logit[sdx, :, judge1] = in_l_cons_logit[sdx, :, judge1]
            judge2 = (l_stable_map==1).astype(np.uint8)*(r_stable_map==1).astype(np.uint8)*(l_sample_cons.data.cpu().numpy()[0] > r_sample_cons.data.cpu().numpy()[0]) == 1
            # loss: r -> l
            tar_l_class_logit[sdx, :, judge2] = in_r_cons_logit[sdx, :, judge2]
        
        # calculate stablization weight
        stabilization_weight = args.stabilization_scale * ramps.sigmoid_rampup(i_iter, args.stabilization_rampup)

        # stabilization loss for r model
        r_stabilization_loss = stabilization_weight * torch.pow(torch.abs(F.softmax(r_model_un_tps_out, dim=1)-F.softmax(tar_l_class_logit, dim=1)), 2).mean() 
        
        # stabilization loss for l model
        l_stabilization_loss = stabilization_weight * torch.pow(torch.abs(F.softmax(l_model_un_tps_out, dim=1)-F.softmax(tar_r_class_logit, dim=1)), 2).mean() 
 
        l_loss += l_stabilization_loss
        r_loss += r_stabilization_loss
        
        #adversarial loss            
        # concatenate the prediction with the input images
        images_remain = (images_un_var-torch.min(images_un_var))/(torch.max(images_un_var)- torch.min(images_un_var))    
        pred_remain_l = l_model_un_out
        #print (pred_remain.size(), images_remain.size())
        pred_cat_l = torch.cat((F.softmax(pred_remain_l, dim=1), images_remain), dim=1)             
        D_l_out_z, D_l_out_y_pred = model_D_l(pred_cat_l) # predicts the D ouput 0-1 and feature map for FM-loss               
        # find predicted segmentation maps above threshold 
        pred_sel_l, labels_sel_l, count_l = find_good_maps(D_l_out_z, pred_remain_l) 
        # training loss on above threshold segmentation predictions (Cross Entropy Loss)
        if count_l > 0 and i_iter > 0:
            l_loss_st = loss_calc(pred_sel_l, labels_sel_l)
            l_loss += args.lambda_st*l_loss_st
            
        pred_remain_r = r_model_un_out
        #print (pred_remain.size(), images_remain.size())
        pred_cat_r = torch.cat((F.softmax(pred_remain_r, dim=1), images_remain), dim=1)            
        D_r_out_z, D_r_out_y_pred = model_D_r(pred_cat_r) # predicts the D ouput 0-1 and feature map for FM-loss            
        # find predicted segmentation maps above threshold 
        pred_sel_r, labels_sel_r, count_r = find_good_maps(D_r_out_z, pred_remain_r) 
        # training loss on above threshold segmentation predictions (Cross Entropy Loss)
        if count_r > 0 and i_iter > 0:
            r_loss_st = loss_calc(pred_sel_r, labels_sel_r)
            r_loss += args.lambda_st*r_loss_st
            
        # Concatenates the input images and ground-truth maps for the Districrimator 'Real' input
        try:
            batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = iter(trainloader_gt)
            batch_gt = next(trainloader_gt_iter)

        images_gt, labels_gt, _, _, _ = batch_gt
        # Converts grounth truth segmentation into 'num_classes' segmentation maps. 
        D_gt_v = Variable(one_hot(labels_gt)).cuda()             
        images_gt = images_gt.cuda()
        images_gt = (images_gt - torch.min(images_gt))/(torch.max(images)-torch.min(images))           
        D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim=1)
        
        D_l_out_z_gt , D_l_out_y_gt = model_D_l(D_gt_v_cat)
        D_r_out_z_gt , D_r_out_y_gt = model_D_r(D_gt_v_cat)

        # L1 loss for Feature Matching Loss
        l_loss_fm = torch.mean(torch.abs(torch.mean(D_l_out_y_gt, 0) - torch.mean(D_l_out_y_pred, 0)))
        r_loss_fm = torch.mean(torch.abs(torch.mean(D_r_out_y_gt, 0) - torch.mean(D_r_out_y_pred, 0)))
        
        l_loss += args.lambda_fm*l_loss_fm
        r_loss += args.lambda_fm*r_loss_fm
        
        l_loss.backward()
        r_loss.backward()
        
        # train D
        for param in model_D_l.parameters():
            param.requires_grad = True
        for param in model_D_r.parameters():
            param.requires_grad = True

        #D_l
        # train with pred
        pred_cat_l = pred_cat_l.detach()  # detach does not allow the graddients to back propagate.      
        D_l_out_z, _ = model_D_l(pred_cat_l)
        y_l_fake_ = Variable(torch.zeros(D_l_out_z.size(0), 1).cuda())
        loss_D_l_fake = criterion(D_l_out_z, y_l_fake_) 
        # train with gt
        D_l_out_z_gt , _ = model_D_l(D_gt_v_cat)
        y_l_real_ = Variable(torch.ones(D_l_out_z_gt.size(0), 1).cuda()) 
        loss_D_l_real = criterion(D_l_out_z_gt, y_l_real_)      
        loss_D_l = (loss_D_l_fake + loss_D_l_real)/2.0
        loss_D_l.backward()
        
        #D_r
        # train with pred
        pred_cat_r = pred_cat_r.detach()  # detach does not allow the graddients to back propagate.      
        D_r_out_z, _ = model_D_r(pred_cat_r)
        y_r_fake_ = Variable(torch.zeros(D_r_out_z.size(0), 1).cuda())
        loss_D_r_fake = criterion(D_r_out_z, y_r_fake_) 
        # train with gt
        D_r_out_z_gt , _ = model_D_r(D_gt_v_cat)
        y_r_real_ = Variable(torch.ones(D_r_out_z_gt.size(0), 1).cuda()) 
        loss_D_r_real = criterion(D_r_out_z_gt, y_r_real_)      
        loss_D_r = (loss_D_r_fake + loss_D_r_real)/2.0
        loss_D_r.backward()
        
        # update model
        l_optimizer.step()       
        r_optimizer.step()
        optimizer_D_l.step()
        optimizer_D_r.step()
        
        print('iter = ', i_iter, 'of', args.num_steps,'completed, l_loss = ', l_loss.data.cpu().numpy())
        print('iter = ', i_iter, 'of', args.num_steps,'completed, r_loss = ', r_loss.data.cpu().numpy())

        if args.dataset == 'pascal_voc': 
            scene_name = 'VOC12'
        elif args.dataset == 'pascal_context':
            scene_name = 'PASCALContext'
        elif args.dataset == 'cityscapes':
            scene_name = 'Cityscapes'
        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(l_model.state_dict(), osp.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(args.num_steps)+'_l_model.pth'))
            torch.save(r_model.state_dict(), osp.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(args.num_steps)+'_r_model.pth'))
            torch.save(model_D_l.state_dict(), os.path.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(args.num_steps)+'_D_l.pth'))
            torch.save(model_D_r.state_dict(), os.path.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(args.num_steps)+'_D_r.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print('taking snapshot ...')
            torch.save(l_model.state_dict(), osp.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(i_iter)+'_l_model.pth'))
            torch.save(r_model.state_dict(), osp.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(i_iter)+'_r_model.pth'))
            torch.save(model_D_l.state_dict(), os.path.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(i_iter)+'_D_l.pth'))
            torch.save(model_D_r.state_dict(), os.path.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(i_iter)+'_D_r.pth'))
        i_iter = i_iter+1

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
