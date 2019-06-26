#!/usr/bin/env python
from __future__ import print_function, division

import os
os.environ['THEANO_FLAGS'] = "device=gpu0"
import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import argparse
import matplotlib.pyplot as plt
from os.path import join
from scipy.io import loadmat, savemat

from utils import compressed_sensing as cs

from utils.metric import mse
from utils.metric import complex_psnr
from cascadenet.network.model import build_d2_c2, build_d5_c5
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format


def prep_input(im, acc=6):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    #mask = cs.cartesian_mask(im.shape, acc, sample_n=8)##########笛卡儿采样 
    mask = loadmat(join('./data/mask/mask_radial_015_HFinCenter.mat'))['mask_radial_015_HFinCenter']
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    # print(type(k_und),k_und.shape)
    # np.save('k_und.mat',k_und)
    # assert False
    im_gnd_l = to_lasagne_format(im)
    im_und_l = to_lasagne_format(im_und)
    k_und_l = to_lasagne_format(k_und)
    mask_l = to_lasagne_format(mask, mask=True)

    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]


def create_dummy_data():
    
    test_data = loadmat(join(project_root, './data/test_data_32.mat'))['test_data_32']
    test_nx, test_ny, test_nz = test_data.shape
    test = np.transpose(test_data, (2, 0, 1)).reshape((-1, test_nx, test_ny))
    print('###test=',test.shape)
    return  test


def compile_test_fn(network, net_config, args):
    """
    Create validation function
    """
    # Hyper-parameters
    base_lr = float(args.lr[0])
    l2 = float(args.l2[0])

    # Theano variables
    input_var = net_config['input'].input_var
    mask_var = net_config['mask'].input_var
    kspace_var = net_config['kspace_input'].input_var
    target_var = T.tensor4('targets')

    # Objective
    pred = lasagne.layers.get_output(network)
    # complex valued signal has 2 channels, which counts as 1.
    loss_sq = lasagne.objectives.squared_error(target_var, pred).mean() * 2
    if l2:
        l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        print("l2_penalty = ",l2_penalty)
        print("l2 = ",l2)
        loss = loss_sq + l2_penalty * l2

    print(' Compiling ... ')
    t_start = time.time()

    val_fn = theano.function([input_var, mask_var, kspace_var, target_var],
                             [loss, pred],
                             on_unused_input='ignore')
    t_end = time.time()
    print(' ... Done, took %.4f s' % (t_end - t_start))

    return val_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['2'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.001'], help='initial learning rate')
    parser.add_argument('--l2', metavar='float', nargs=1,
                        default=['1e-6'], help='l2 regularisation')
    parser.add_argument('--acceleration_factor', metavar='float', nargs=1,
                        default=['4.0'],
                        help='Acceleration factor for k-space sampling')

    args = parser.parse_args()

    # Project config
    model_name = 'd5_c5'
    acc = float(args.acceleration_factor[0])  # undersampling rate
    batch_size = int(args.batch_size[0])
    Nx, Ny = 256, 256
    # Configure directory info
    project_root = '.'
    save_dir = join(project_root, 'models/%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Specify network
    input_shape = (batch_size, 2, Nx, Ny)
    net_config, net,  = build_d5_c5(input_shape)
    #assert False
    # D5-C5 with pre-trained parameters
    epoch_state = 731
    with np.load('./models/d5_c5/d5_c5_epoch_{0}.npz'.format(epoch_state)) as f:
        param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net, param_values)

    # Compute acceleration rate
    dummy_mask = cs.cartesian_mask((10, Nx, Ny), acc, sample_n=8)
    sample_und_factor = cs.undersampling_rate(dummy_mask)
    print('Undersampling Rate: {:.2f}'.format(sample_und_factor))

    # Compile function
    val_fn = compile_test_fn(net, net_config, args)


    # Create dataset
    test = create_dummy_data()

    vis = []
    base_psnr_list = []
    test_psnr_list = []
    for im in iterate_minibatch(test, batch_size , shuffle=False):
        im_und, k_und, mask, im_gnd = prep_input(im, acc=acc)
        err, pred = val_fn(im_und, mask, k_und, im_gnd)
        for im_i, und_i, pred_i in zip(im,
                                       from_lasagne_format(im_und),
                                       from_lasagne_format(pred)):
            base_psnr_list.append(complex_psnr(im_i, und_i, peak='max'))
            testpsnr=complex_psnr(im_i, pred_i, peak='max')
            test_psnr_list.append(testpsnr)

        for im_num in range(batch_size):
            vis.append((im[im_num],
                        from_lasagne_format(pred)[im_num],
                        from_lasagne_format(im_und)[im_num],
                        from_lasagne_format(mask, mask=True)[im_num]))

    i = 0
    for im_i, pred_i, und_i, mask_i in vis:
        plt.imsave(join(save_dir, 'im{0}_test.png'.format(i)),
                   abs(np.concatenate([ und_i, pred_i,
                                       im_i], 1)),
                   cmap='gray')
        plt.imsave(join(save_dir, 'mask{0}.png'.format(i)), mask_i,
                   cmap='gray')
        i += 1
    savemat(join(save_dir, 'im_epoch_{0}'.format(epoch)),{'im': vis})