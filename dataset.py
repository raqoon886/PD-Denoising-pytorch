import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

#color=0 for gray, color=1 for color images
def prepare_data(data_path, output_path, patch_size, stride, aug_times=1, color=0, mode=object, label=False):
    # train
    print('process training data')
    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    files = glob.glob(os.path.join(data_path, '*.jpg'))
    if label:
        h5f = h5py.File(os.path.join(output_path, '{}_label.h5'.format(mode)), 'w')
    else:
        h5f = h5py.File(os.path.join(output_path, '{}_input.h5'.format(mode)), 'w')
    files.sort()
    train_num = 0
    for i in range(len(files)):
        print(i)
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            if color == 0:
                Img = np.expand_dims(Img[:,:,0].copy(), 0)
            elif color == 1:
                Img = np.transpose(Img, (2,0,1)) #move the channel to the first dimension
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # # val
    # print('\nprocess validation data')
    # #files.clear()
    # files = []
    # files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    # files.sort()
    # h5f = h5py.File('val.h5', 'w')
    # val_num = 0
    # for i in range(len(files)):
    #     print("file: %s" % files[i])
    #     img = cv2.imread(files[i])
    #     img = np.expand_dims(img[:,:,0], 0)
    #     img = np.float32(normalize(img))
    #     h5f.create_dataset(str(val_num), data=img)
    #     val_num += 1
    # h5f.close()
    print('training set, # samples %d\n' % train_num)
    # print('val set, # samples %d\n' % val_num)


#Prepare the data for real image and noise
def prepare_real_data(real_data_path, noise_data_path, output_path, patch_size, stride, aug_times = 1, color = 0, mode=object):
    # train
    print('process training data')
    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    if color == 0:
        real_files = glob.glob(os.path.join(real_data_path, '*.png'))
        noise_files = glob.glob(os.path.join(noise_data_path, '*png'))
        h5f = h5py.File('train.h5', 'w')
    elif color == 1:
        real_files = glob.glob(os.path.join(real_data_path, '*.png'))
        noise_files = glob.glob(os.path.join(noise_data_path, '*.png'))
        h5f = h5py.File(output_path + '/' + mode + '_c.h5', 'w')
    print(real_files)
    real_files.sort()
    noise_files.sort()
    train_num = 0
    for i in range(len(real_files)):
        real_img = cv2.imread(real_files[i])
        noise_img = cv2.imread(noise_files[i])
        h, w, c = real_img.shape
        for k in range(len(scales)):
            Img = cv2.resize(real_img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            NImg = cv2.resize(noise_img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            if color == 0:
                Img = np.expand_dims(Img[:,:,0].copy(), 0)
                NImg = np.expend_dims(NImg[:,:,0].copy(), 0)
            elif color == 1:
                Img = np.transpose(Img, (2,0,1)) #move the channel to the first dimension
                NImg = np.transpose(NImg, (2,0,1))
            Img = np.float32(normalize(Img))
            NImg = np.float32(normalize(NImg))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            Npatches = Im2Patch(NImg, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (real_files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                ndata = Npatches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                h5f.create_dataset(str(train_num)+"_noise", data=ndata)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    ndata_aug = data_augmentation(ndata, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    h5f.create_dataset(str(train_num)+"_n_aug_%d" % (m+1), data_ndata_aug)
                    train_num += 1
    h5f.close()
    # # val
    # print('\nprocess validation data')
    # #files.clear()
    # files = []
    # files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    # files.sort()
    # h5f = h5py.File('val.h5', 'w')
    # val_num = 0
    # for i in range(len(files)):
    #     print("file: %s" % files[i])
    #     img = cv2.imread(files[i])
    #     img = np.expand_dims(img[:,:,0], 0)
    #     img = np.float32(normalize(img))
    #     h5f.create_dataset(str(val_num), data=img)
    #     val_num += 1
    # h5f.close()
    print('training set, # samples %d\n' % train_num)
    # print('val set, # samples %d\n' % val_num)

def generate_noise_level_data(image, image_name, out_folder):
    '''
    Generate AWGN noisy images of different levels at different channels
    Given an image, 
    B: 0-75 at 15
    G: 0-75 at 15
    R: 0-75 at 15
    totally 216 images for one input image
    '''
    os.mkdir(os.path.join(out_folder, image_name))
    for i in range(6):
       for j in range(6):
           for k in range(6):
               noise_level_list = [i * 15, j * 15, k * 15]  #the pre-defined noise level
               noisy = generate_noisy(image/255., 3, np.array(noise_level_list) / 255.)  #generate noisy image according to the given levels
               cv2.imwrite(os.path.join(out_folder, image_name, image_name + '_%d_%d_%d.png' % (i+1, j+1, k+1)), noisy[:,:,::-1]*255.)

def prepare_noise_level_data(data_path, out_path):
    # train
    files = glob.glob(os.path.join(data_path, '*'))
    for i in range(len(files)):
        file_name = files[i].split('/')[-1].split('.')[0]
        img = cv2.imread(files[i])
        img = img[:,:,::-1]
        generate_noise_level_data(img, file_name, out_path)
        
class Dataset(udata.Dataset):
    def __init__(self, input_path, label_path=None, mode='train'):
        super(Dataset, self).__init__()
        self.input_path = input_path
        self.label_path = label_path
        self.mode = mode
        h5f = h5py.File(os.path.join(self.input_path, '{}_input.h5'.format(self.mode)), 'r')
        self.input_keys = list(h5f.keys())
        h5f.close()

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self, index):

        h5f = h5py.File(os.path.join(self.input_path, '{}_input.h5'.format(self.mode)), 'r')
        input_keys = list(h5f.keys())
        input_key = input_keys[index]
        input_data = np.array(h5f[input_key])
        h5f.close()

        h5f = h5py.File(os.path.join(self.label_path, '{}_label.h5'.format(self.mode)), 'r')
        label_keys = list(h5f.keys())
        label_key = label_keys[index]
        label_data = np.array(h5f[label_keys])
        h5f.close()

        d = {'input' : input_data,
             'label' : label_data}
        return d
