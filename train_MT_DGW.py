import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import VOCDataSet
import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()
from data import get_loader, get_data_path
from data.augmentations import *

import ramps
from perturbations import *

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATASET = 'pascal_voc' #pascal_voc or pascal_context

BATCH_SIZE = 10
DATA_DIRECTORY = '../../data/VOCdevkit/VOC2012'
DATA_LIST_PATH = './dataset/list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_COCO_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

SPLIT_ID = './splits/voc/split_0.pkl'
LABELED_RATIO= None # use 100% labeled data 

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
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument("--perturbations", type=str, default='DGW',
                        help="data augmentation for perturbations")
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


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def main():
    """Create the model and start the training."""
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Create network.
    model = create_model(num_classes=args.num_classes, restore_from=args.restore_from, ema=False)
    ema_model = create_model(num_classes=args.num_classes, restore_from=args.restore_from, ema=True)
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    #trainloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size, 
    #                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN), 
    #                batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

    if args.dataset == 'pascal_voc':
        train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    else:
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

    trainloader_supervised = data.DataLoader(train_dataset,
                    batch_size=args.batch_size, sampler=train_sampler_supervised, num_workers=5, pin_memory=True)
    trainloader_unsupervised = data.DataLoader(train_dataset,
                    batch_size=args.batch_size, sampler=train_sampler_unsupervised, num_workers=5, pin_memory=True)

    trainloader_iter_supervised = iter(trainloader_supervised)
    trainloader_iter_unsupervised = iter(trainloader_unsupervised)
 
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    i_iter = 0
    print(i_iter)
    while i_iter < args.num_steps:
    #for i_iter in range(args.num_steps):
    #for i_iter, batch in enumerate(trainloader):
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
        images_ema, _, _, _, _ = batch_unsupervised
        
        images = images.cuda()
        images_ema = images_ema.cuda()
        #for i in range(5):
        #    cv2.imwrite('img_{}.png'.format(i), np.uint8(images_ema.permute(0, 2, 3, 1)[i].cpu().numpy()))
        #print(images_ema.detach().cpu().numpy())
        #print(images_ema.cpu().numpy().shape)

        if args.perturbations == 'DGW':
            try:
                tps_images_ema, source_locs, dest_locs  = dgw(images_ema.permute(0, 2, 3, 1))
            except Exception as e:
                print('--->',e)
                continue

        ###print(tps_images_ema.cpu().numpy().shape)
        #print(tps_images_ema.permute(0, 2, 3, 1)[0].cpu().numpy().shape)
        #for i in range(5):
        #    cv2.imwrite('img_after_tps_{}.png'.format(i), np.uint8(tps_images_ema.permute(0, 2, 3, 1)[i].cpu().numpy()))

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        #print(images.cpu().numpy())
        pred = interp(model(images))
        ce_loss = loss_calc(pred, labels)

        if args.perturbations == 'DGW':
            tps_pred = F.softmax(interp(model(tps_images_ema)), dim=1)
            pred_ema = interp(ema_model(images_ema))
            pred_ema = pred_ema.permute(0,2,3,1)
            #print(pred_ema.detach().cpu().numpy().shape)
            #pred_ema_tps, _, _ = sparse_image_warp(pred_ema.cpu(), source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
            pred_ema_tps, _ = sparse_image_warp(pred_ema, source_locs, dest_locs, interpolation_order=1, regularization_weight=0.0, num_boundaries_points=0)
            #pred_ema_tps = pred_ema_tps.cuda().permute(0,3,1,2)
            pred_ema_tps = pred_ema_tps.permute(0,3,1,2)
            pred_ema_tps = Variable(pred_ema_tps.detach().data, requires_grad=False)
            #print(pred_ema_tps.detach().cpu().numpy().shape)
            pred_ema_tps = F.softmax(pred_ema_tps,dim=1)
            #print(pred_ema_tps.detach().cpu().numpy().shape)
            confidence, _ = pred_ema_tps.max(dim=1)
            #print(confidence)
            conf_fac = (confidence > 0.97).float().mean()
            consistency_weight = get_current_consistency_weight(i_iter)
            #print(tps_pred.detach().cpu().numpy().shape)
            #print(pred_ema_tps.detach().cpu().numpy().shape)
            consistency_loss = torch.pow(torch.abs(tps_pred-pred_ema_tps), 2).mean()

        consistency_loss = consistency_loss * conf_fac * consistency_weight
        #consistency_loss = consistency_loss * consistency_weight
        
        #loss = ce_loss
        loss = ce_loss + consistency_loss
        loss.backward()

        optimizer.step()
        
        update_ema_variables(model, ema_model, args.ema_decay, i_iter)
        
        print('iter = ', i_iter, 'of', args.num_steps,'completed, loss = ', loss.data.cpu().numpy())

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'VOC12_scene_'+str(args.num_steps)+'.pth'))
            torch.save(ema_model.state_dict(), osp.join(args.snapshot_dir, 'VOC12_scene_'+str(args.num_steps)+'_ema.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC12_scene_'+str(i_iter)+'.pth'))
            torch.save(ema_model.state_dict(),osp.join(args.snapshot_dir, 'VOC12_scene_'+str(i_iter)+'_ema.pth'))

        i_iter += 1

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
