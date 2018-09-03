# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:57:07 2018

@author: yamamoto
"""

import argparse

import numpy as np

import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions, triggers
from chainer import iterators, optimizers, serializers

from multiplication_experiment_network import NaluLayer

def LoadData(min_value=1, max_value=100, N = 10000, validation_split=True,
             validation_rate=0.1):
    '''Generate toy data'''
    
    if validation_split==True:        
        N_train = int(N * (1-validation_rate))
     
    X = np.random.randint(min_value, max_value, (N,2)).astype(np.float32)
    X = X / max_value
    
    # Generate Answers
    Y = X[:,0] * X[:,1]
    Y = np.expand_dims(Y, axis=1)
    
    # Concatenate
    data = np.concatenate((X, Y), axis=1).astype(np.float32)
    
    if validation_split==True: 
        #Split train & validation set
        return data[:N_train], data[N_train:]
    else:
        return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print("Loading datas")
    max_value = 200
    train, validation = LoadData(max_value=max_value, N=20000,
                                 validation_split=True)
    
    # Set up a neural network to train.
    print("Building model")
    model = NaluLayer(2, 1)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)

    train_iter = iterators.SerialIterator(train, batch_size=args.batch, shuffle=True)
    test_iter = iterators.SerialIterator(validation, batch_size=args.batch,
                                         repeat=False, shuffle=False)

    if args.model != None:
        print( "loading model from " + args.model)
        serializers.load_npz(args.model, model)

    if args.opt != None:
        print( "loading opt from " + args.opt)
        serializers.load_npz(args.opt, optimizer)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))

    # Snapshot
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    #serializers.load_npz('./results/snapshot_iter_1407', trainer)

    # Decay learning rate
    points = [args.epoch*0.75]
    trainer.extend(extensions.ExponentialShift('alpha', 0.1),
                   trigger=triggers.ManualScheduleTrigger(points,'epoch'))


    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']), trigger=(1, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    #Plot computation graph
    trainer.extend(extensions.dump_graph('main/loss'))

    # Train
    trainer.run()

    # Save results
    modelname = "./results/model"
    print( "saving model to " + modelname )
    serializers.save_npz(modelname, model)

    optimizername = "./results/optimizer"
    print( "saving optimizer to " + optimizername )
    serializers.save_npz(optimizername, optimizer)

    # Estimate model
    model = model = NaluLayer(2, 1, return_prediction=True)
    weight_dir = "./results/model"
    print( "Loading model from " + weight_dir)
    serializers.load_npz(weight_dir, model)
    
    n_test = 10
    test = LoadData(N = n_test, validation_split=False)
    loss, y = model(test)
    y = cuda.to_cpu(y.data)
    #print(test[0], y[0])
    for i in range(n_test):
        print('-' * 10)
        print('Q:  ', round(test[i,0]*max_value),'x', round(test[i,1]*max_value))
        print('A:  ', round(test[i,2]*(max_value**2)))
        print('P:  ', round(y[i,0]*(max_value**2)))
    print('-' * 10)
    
if __name__ == '__main__':
    main()