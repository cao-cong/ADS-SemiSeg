import argparse

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transform
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
from data import get_loader, get_data_path
from data.augmentations import *
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATASET = 'pascal_voc' #pascal_voc or pascal_context 

BATCH_SIZE = 10
DATA_DIRECTORY = '../data/VOC2012/VOCdevkit/VOC2012'
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


def main():
    """Create the model and start the training."""
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Create network.
    model = Res_Deeplab(num_classes=args.num_classes)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[1]=='layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
    model.train()
    model.cuda()
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

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
        data_aug = Compose([RandomCrop_city((256, 512)), RandomHorizontallyFlip()])
        train_dataset = data_loader( data_path, is_transform=True, augmentations=data_aug) 
        
    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if args.labeled_ratio is None:
        trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                        num_workers=5, pin_memory=True)
    else:
        partial_size = int(args.labeled_ratio * train_dataset_size)

        if args.split_id is not None:
            train_ids = pickle.load(open(args.split_id, 'rb'))
            print('loading train ids from {}'.format(args.split_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)

        pickle.dump(train_ids, open(args.snapshot_dir + 'split.pkl', 'wb'))

        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler, num_workers=5, pin_memory=True)

    trainloader_iter = iter(trainloader)
 
    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    for i_iter in range(args.num_steps):
        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels, _, _, _ = batch
        images = Variable(images).cuda()

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        pred = interp(model(images))
        loss = loss_calc(pred, labels)
        loss.backward()
        optimizer.step()

        
        print('iter = ', i_iter, 'of', args.num_steps,'completed, loss = ', loss.data.cpu().numpy())
        if args.dataset == 'pascal_voc': 
            scene_name = 'VOC12'
        elif args.dataset == 'pascal_context':
            scene_name = 'PASCALContext'
        elif args.dataset == 'cityscapes':
            scene_name = 'Cityscapes'
        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, '{}_scene_'.format(scene_name)+str(i_iter)+'.pth'))     

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
