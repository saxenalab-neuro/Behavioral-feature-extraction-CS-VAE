import numpy as np
import tensorflow as tf
tf.random.set_seed(33)
from sklearn.utils import shuffle
import h5py
import random
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self,hdf5file='h.hdf5',image_name= ["images_36"], label_name= ["labels_36"],dim=(128,128), n_channels=2,n_labs=5,batch_size=256, shuffle=True, **kwargs):
        self.image_name = image_name
        self.label_name = label_name
        self.dim = dim
        self.n_channels = n_channels
        self.n_labs=n_labs
        self.a=h5py.File(hdf5file, 'r')
        self.name = list(self.a.keys())
        self.i1 = self.a[self.name[1]]
        self.trail = list(self.i1.keys())
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
       
    def __get_shuffle(self):
        va = {i: [] for i in range(len(self.image_name))}
        num_sub=len(self.image_name)
        trial_len=len(self.i1[self.trail[0]])
        for i in range(num_sub):
            imn = self.image_name[i]
            lbn = self.image_name[i]
            for j in self.trail:
                data = np.array(self.a[imn][j])
                v = np.var(data, axis=0)
                va[i].append(np.mean(v))

        so = {i: [] for i in range(num_sub)}
        vv = {i: [] for i in range(num_sub)}
        for i in range(num_sub):
            so[i] = np.argsort(np.array(va[i]) * -1)
            vv[i] = np.sort(np.array(va[i]) * -1)

        z = 0
        idx = {i: [] for i in range(int(len(self.trail) / 2)* num_sub * trial_len)}
        for i in range(num_sub):
            seq = so[i]
            for j in seq[:int(len(self.trail) / 2)]:
                for k in range(trial_len):
                    n = self.trail[j]
                    idx[z].append(i)
                    idx[z].append(n)
                    idx[z].append(k)
                    z += 1

        idx2 = shuffle(idx)

        idx = {i: [] for i in range(int(len(self.trail)/2)*  num_sub * trial_len)}
        z = 0
        for k in range(trial_len):
            for i in range(num_sub):
                seq = so[i]
                for j in seq[int(len(self.trail) / 2):]:
                    n = self.trail[j]

                    idx[z].append(i)
                    idx[z].append(n)
                    idx[z].append(k)
                    z += 1
        # len(idx)
        idx3 = shuffle(idx)
        idx = idx2 + idx3
        return idx

    def __len__(self):
        return int(np.floor(len(self.__get_shuffle()) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.__get_shuffle()))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.__get_shuffle()[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return [X,Y],[X,Y]
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, self.n_labs))

        # Generate data
        for i, alll in enumerate(list_IDs_temp):
            # Store sample

            if alll[0] == 0:
                imn = self.image_name[0]
                lbn = self.label_name[0]

            elif alll[0] == 1:
                imn = self.image_name[1]
                lbn = self.label_name[1]
            elif alll[0] == 2:
                imn = self.image_name[2]
                lbn = self.label_name[2]
            else:
                imn = self.image_name[3]
                lbn = self.label_name[3]
            try:
                x = (np.array(self.a[imn][alll[1]][alll[2]]) / 255).T  # read dataset on the fly
                x = np.rollaxis(x, 1, 0)
                y = np.array(self.a[lbn][alll[1]][alll[2]])  # read dataset on the fly
                if np.sum(x) == 0 or np.sum(x) == np.inf or np.sum(x) == -np.inf or np.sum(y) == 0 or np.sum(y) == np.inf or np.sum(y) == -np.inf:
                    x = (np.array(self.a[imn][self.trail[0]][1]) / 255).T  # read dataset on the fly
                    x = np.rollaxis(x, 1, 0)
                    y = np.array(self.a[lbn][self.trail[0]][1])  # read dataset on the fly
            except:
                x = (np.array(self.a[imn][self.trail[0]][1]) / 255).T  # read dataset on the fly
                x = np.rollaxis(x, 1, 0)
                y = np.array(self.a[lbn][self.trail[0]][1])  # read dataset on the fly
                
                
            X[i] = x

            # Store class
            Y[i] =y

        return X,Y

# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
#                  n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')

#             # Store class
#             y[i] = self.labels[ID]

#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
