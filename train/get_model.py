import train.loss as loss
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(33)

import train.model as model
import h5py
from tensorflow.keras import backend as K
# from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, constraints
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Input, Dense, Conv2D, multiply, Lambda, Add, Concatenate, Multiply, Conv2DTranspose, \
    Layer, Reshape, ZeroPadding2D, Flatten, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, LearningRateScheduler, \
    ModelCheckpoint
# set_session(tf.Session(config=config))
from sklearn.utils import shuffle
import numpy as np
import json
gpus = tf.config.list_physical_devices('GPU')


class ALLModel(Model):
    def __init__(self,img_x,img_y,img_z,sdim,udim,ks,flag,theta,alpha,beta,latent_dim,bac):
        super().__init__()
        self.img_x,self.img_y,self.img_z=img_x,img_y,img_z
        self.sdim=sdim
        self.udim=udim
        self.ks=ks
        self.flag=flag
        self.theta=theta
        self.alpha=alpha
        self.beta=beta
        self.latent_dim=latent_dim
        self.bac=bac
    def get_model(self):

        input0 = Input(shape=(self.img_x,self.img_y,self.img_z), name="mice1")
        inputlabel = Input(shape=(self.sdim), name="micelabel1")
        ######image encoding###
        [A2, B, C, z_log_var_A, z_log_var_B, hid]=model.Encoder(latent_dim=self.latent_dim , bac=self.bac,sdim=self.sdim, udim=self.udim)(input0)
        #####latent partation####
        [latent_output,label_output],[csd_loss,sup_label_loss,kl_loss_sup,mi_loss, total_corr_loss, dim_wise_kl_loss]=model.get_latent(sdim=self.sdim, udim=self.udim,inputshape=K.shape(input0)[0], ks=self.ks, flag=self.flag, theta=self.theta,alpha=self.alpha,beta=self.beta)([inputlabel,A2, B, C, z_log_var_A, z_log_var_B])
        #####image decoder####
        out=model.Decoder(hidden_dim=hid.shape[1],img_z=self.img_z)(latent_output)
        [out, _], frame_mse_loss = loss.MSE_UNSUP()([out, input0])

        allmodel = Model(inputs=[input0, inputlabel], outputs=[out,label_output])
        #     allmodel.summary() ###check the momdel structure########
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
        ######add corresponding loss metric#####
        allmodel.add_metric(csd_loss, name="csd_loss",aggregation='mean')
        allmodel.add_metric(sup_label_loss, name="sup_label_loss",aggregation='mean')
        allmodel.add_metric(kl_loss_sup, name="kl_loss_sup",aggregation='mean')
        allmodel.add_metric(mi_loss, name="mi_loss",aggregation='mean')
        allmodel.add_metric(total_corr_loss, name="total_corr_loss",aggregation='mean')
        allmodel.add_metric(dim_wise_kl_loss, name="dim_wise_kl_loss",aggregation='mean')
        allmodel.add_metric(frame_mse_loss, name="frame_mse_loss",aggregation='mean')

        allmodel.compile(optimizer=optimizer,loss='mse')
        
        return allmodel
    