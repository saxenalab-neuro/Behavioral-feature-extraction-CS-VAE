import numpy as np
import tensorflow as tf
tf.random.set_seed(33)
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer
import math
# from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, multiply, Lambda, Add, Concatenate, Multiply, Conv2DTranspose, \
    Layer, Reshape, ZeroPadding2D, Flatten, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
import loss

class Encoder(Layer):
    def __init__(self, latent_dim=9, bac=2,sdim=5, udim=2,name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.sdim=sdim
        self.latent_dim=latent_dim
        self.bac=bac
        self.udim=udim
        
        self.LRELU = LeakyReLU(alpha=0.05)
        self.bachnorm=BatchNormalization()
        self.bachnorm1=BatchNormalization()
        self.bachnorm2=BatchNormalization()
        self.bachnorm3=BatchNormalization()
        self.bachnorm4=BatchNormalization()

        self.encoder1=Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.encoder2=Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.encoder3=Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.encoder4=Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.encoder5=Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.flat=Flatten()
        self.dense1=Dense(latent_dim, kernel_regularizer='l1_l2')
        self.dense2= Dense(sdim + udim, kernel_regularizer='l1_l2')
        self.initializera = tf.keras.initializers.Orthogonal(gain=1.0, seed=42)
        self.initializerb = tf.keras.initializers.Orthogonal(gain=2.0, seed=30)
        self.densec=Dense(bac, use_bias=False, kernel_regularizer='l1_l2')
        self.densea=Dense(sdim, use_bias=False, kernel_initializer=self.initializera, kernel_regularizer='l1_l2')
        self.denseb=Dense(udim, use_bias=False, kernel_initializer=self.initializerb, kernel_regularizer='l1_l2')

    def call(self, inputs):
        x=self.bachnorm(self.LRELU(self.encoder1(inputs)))
        x = self.bachnorm1(self.LRELU(self.encoder2(x)))
        x = self.bachnorm2(self.LRELU(self.encoder3(x)))
        x = self.bachnorm3(self.LRELU(self.encoder4(x)))
        x = self.bachnorm4(self.LRELU(self.encoder5(x)))
        x = self.flat(x)
        z_mu=self.dense1(x)
        z_log_var=self.dense2(x)

        z_log_var_A = z_log_var[:, :self.sdim]
        z_log_var_B = z_log_var[:, self.sdim:]
        A1=self.densea
        B1=self.denseb
        C1=self.densec
        A1.trainable = False
        B1.trainable = False
        # C1.trainable = False

        A2 = A1(z_mu)
        B = B1(z_mu)
        C = C1(z_mu)
        return [A2,B,C,z_log_var_A,z_log_var_B,x]

class get_latent(Layer):

    def __init__(self, sdim=5, udim=2,inputshape=64, ks=15, flag='sr', theta=500,alpha=1000,beta=5,name="get_latent", **kwargs):
        super(get_latent, self).__init__(name=name, **kwargs)
        self.csreg=loss.ITLRegularizer()
        self.denselabel=Dense(sdim, use_bias=True, kernel_constraint=loss.DiagonalWeight(sdim), kernel_initializer="he_normal")
        self.mseloss_label=loss.MSE_SUP()
        self.kl=loss.KLDivergenceLayer()
        self.multi= Multiply()
        self.add=Add()
        self.dkl=loss.De_KL()
        self.cat=Concatenate()
        self.inputshape = inputshape
        self.ks = ks
        self.flag = flag
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.sdim=sdim
        self.udim=udim
    def call(self, inputs):
        inputlabel,A2, B, C, z_log_var_A, z_log_var_B=inputs
        C, dis, loss2 = self.csreg(C, self.ks, self.flag, self.theta)

        D1 = self.denselabel(A2)
        [D, A3], loss3 = self.mseloss_label([D1, inputlabel], self.alpha)

        [A, z_log_var_A2], loss4 = self.kl([A2, z_log_var_A])
        z_sigma_A = Lambda(lambda t: K.exp(.5 * t))(z_log_var_A2)

        eps_A = K.random_normal(shape=(self.inputshape, self.sdim))
        z_eps_A2 = self.multi([z_sigma_A, eps_A])  # Multiply()([z_sigma_A, eps_A])
        z_A = self.add([A, z_eps_A2])

        z_sigma_B = Lambda(lambda t: K.exp(.5 * t))(z_log_var_B)
        eps_B = K.random_normal(shape=(self.inputshape, self.udim))
        z_eps_B = self.multi([z_sigma_B, eps_B])
        z_B = self.add([B, z_eps_B])

        [z_B1, B1, z_log_var_B1], [loss5, loss6, loss7] = self.dkl([z_B, B, z_log_var_B], self.beta)

        alla = Concatenate()([z_A, z_B1, C])

        return [alla,D],[loss2,loss3,loss4,loss5, loss6, loss7]


class Decoder(Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, hidden_dim, img_z,name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder1=Dense(hidden_dim, kernel_initializer="he_normal")
        self.re=Reshape((4, 4, 512))
        self.decoder2=Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.LU=LeakyReLU(alpha=0.05)
        self.bachnorm=BatchNormalization()
        self.bachnorm2=BatchNormalization()
        self.bachnorm3=BatchNormalization()
        self.bachnorm4=BatchNormalization()
        self.decoder3=Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.decoder4=Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.decoder5=Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer="he_normal")
        self.decoder6=Conv2DTranspose(img_z, (5, 5), strides=(2, 2), activation='sigmoid', padding='same', kernel_initializer="he_normal")

    def call(self, inputs):
        x = self.decoder1(inputs)
        x = self.re(x)
        x = self.bachnorm(self.LU(self.decoder2(x)))
        x = self.bachnorm2(self.LU(self.decoder3(x)))
        x = self.bachnorm3(self.LU(self.decoder4(x)))
        x = self.bachnorm4(self.LU(self.decoder5(x)))
        x = self.decoder6(x)
        return x
