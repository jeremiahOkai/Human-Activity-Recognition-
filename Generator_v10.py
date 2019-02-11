import numpy as np 
import random
from keras.utils import np_utils
from collections import Counter

class Generator(object):
    def __init__(self,
                 X_train,
                 Y_train,
                 batchsize=64,
                 dropout = 2,
                 dropout_ratio=0.8,
                 noisedrop_ratio=0.0,
                 noisedrop = 0,
                 downsample = True
                 ):
        """
        Arguments
        ---------
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.size_train = X_train.shape[0]
        self.batchsize = batchsize
        self.dropout = dropout
        self.dropout_ratio = dropout_ratio
        self.noisedrop = noisedrop
        self.noisedrop_ratio = noisedrop_ratio
        self.downsample = downsample
        self.on_train_begin()
        
    @classmethod
    def downsampling_classes(cls, input_x, input_y): 
        min_value = (Counter(input_y)).most_common()[-1][1]
        xs = np.empty(shape=(0, input_x.shape[1], input_x.shape[2]))
        ys = np.empty(shape=(0))
        for classname in np.unique(input_y):
            x = np.random.permutation(input_x[input_y == classname])
            x = x[:min_value]
            y = np.empty(min_value)
            y.fill(classname)
            xs = np.append(xs, x, axis=0)
            ys = np.append(ys, y, axis=0)
        ys = np_utils.to_categorical(ys)
        return xs, ys

    def _random_indices(self, ratio):
        """Randomly select a 'ratio' (percentage) of instances from the batch"""
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)
    
    def apply_dropout(self):
        """Drop features"""
        indices = self._random_indices(self.dropout_ratio)          
        for i in indices: 
            idxnoise = np.random.choice(self.inputs[i].shape[1], size=self.dropout, replace=False)
            self.inputs[i][:, idxnoise] = 0
            
    def drop_whole_sensor_measurement(self):
        option = ['a', 'w', 'c', 'aw', 'ac', 'wc', 'awc']
        indices = self._random_indices(self.dropout_ratio)
        for i in indices: 
            idx = random.sample(option, 1)
            if idx[0] == 'c': 
                self.inputs[i][:, 6:] = 0
            elif idx[0] == 'w': 
                self.inputs[i][:, 3:6] = 0
            elif idx[0] == 'a':
                self.inputs[i][:, :3] = 0 
            elif idx[0] == 'aw':
                self.inputs[i][:, :6] = 0 
            elif idx[0] == 'ac':
                self.inputs[i][:, [0,1,2,6,7,8]] = 0 
            elif idx[0] == 'wc':
                self.inputs[i][:, 3:] = 0 
            elif idx[0] == 'awc':
                self.inputs[i] = self.inputs[i]
            
            
            
    def apply_noisedrop(self):
        """Replace features with noise"""
        indices = self._random_indices(self.noisedrop_ratio)
        for i in indices:
            idxnoise = np.random.choice(sequence.shape[1], size=self.noisedrop, replace=False)
            self.inputs[i][:, idxnoise] = np.random.normal(0, 1, self.noisedrop)
      
#     @classmethod
    def on_epoch_begin(self):
        if (self.downsample == True):
            self.Xtrain_balanced, self.ytrain_balanced = self.downsampling_classes(self.X_train, self.Y_train)
        else:
            self.Xtrain_balanced = self.X_train
            self.ytrain_balanced = np_utils.to_categorical(self.Y_train)
        #print(self.Xtrain_balanced.shape)
        random_indices = np.random.permutation(self.Xtrain_balanced_size)
        self.Xtrain_balanced = self.Xtrain_balanced[random_indices]
        self.ytrain_balanced = self.ytrain_balanced[random_indices]
    
    def on_train_begin(self):
        Xtrain_balanced, ytrain_balanced = self.downsampling_classes(self.X_train, self.Y_train)
        self.Xtrain_balanced_size =  Xtrain_balanced.shape[0]
        
    def generate(self): 
        """Generator"""
        while True:
            #print("\n Shuffled \n")
            self.on_epoch_begin()
            #print(" \n", self.Xtrain_balanced[0][0]," \n")
            cuts = [(b, min(b + self.batchsize, self.Xtrain_balanced_size)) for b in range(0, self.Xtrain_balanced_size, self.batchsize)]
            self.count = 0
            for start, end in cuts:
                self.count += 1
                self.inputs = self.Xtrain_balanced[start:end].copy()
                self.targets = self.ytrain_balanced[start:end].copy()
                self.actual_batchsize = self.inputs.shape[0]  # Need this to avoid indices out of bounds
#                 self.apply_dropout()
#                 self.apply_noisedrop()
#                 print('before....', self.inputs[0][0])
#                 self.drop_whole_sensor_measurement()
                self.apply_dropout()
#                 print('after...', self.inputs[0][0])
                yield (self.inputs, self.targets)
            


            
            
            
            
            
            
            
            
            
            
