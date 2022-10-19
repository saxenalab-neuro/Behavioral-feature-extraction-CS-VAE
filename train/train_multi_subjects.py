import loss
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(33)
import data_generator
import model
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
def train():
    #####read parameter file########
    json_path = 'parameter_init.json'

    with open(json_path, "r") as f:
        param = json.load(f)
    #####selsct set of parameter id####
    if param['grid_search'] == True:
        jobid =os.getenv('SLURM_ARRAY_TASK_ID')
        jobid= int(jobid)
    else:
        jobid = int(0)
    ####parameter initialization######
    background = param['contrained_dim'][jobid]
    udim = param['unsupervised_dim'][jobid]  ###
    sdim = param['label_dim'][jobid]
    bac = background
    latent_dim = background + udim + sdim
    
    alpha = param['alpha'][jobid]  ###
    beta = param['beta'][jobid]  ####
    theta = param['theta'][jobid]  ####
    ks = param['ks'][jobid]  ####
    flag = param['flag'][jobid]  ####
    hdf5_file = param['hdf5_file']
    batch_size=param['batch_size'][0]
    epoch_num=param['epoch'][0]
    image_name=param['image_name']
    label_name=param['label_name']
    img_x,img_y,img_z=param['image_size']
    ######read the hdf5 file and get number of inputs####
    ######please generate the file following the 'create_hdf5.ipynb'####
    a = h5py.File(hdf5_file, 'r')
    name = list(a.keys())
    i1 = a[name[1]]
    length=len(name)*len(list(i1.keys()))*len(a[name[0]][list(i1.keys())[0]])
    ####training flow######
    ####inputs initialization####
    input0 = Input(shape=(img_x,img_y,img_z), name="mice1")
    inputlabel = Input(shape=(sdim), name="micelabel1")
    ######image encoding###
    [A2, B, C, z_log_var_A, z_log_var_B, hid]=model.Encoder(latent_dim=latent_dim , bac=bac,sdim=sdim, udim=udim)(input0)
    #####latent partation####
    [latent_output,label_output],[csd_loss,sup_label_loss,kl_loss_sup,mi_loss, total_corr_loss, dim_wise_kl_loss]=model.get_latent(sdim=sdim, udim=udim,inputshape=K.shape(input0)[0], ks=ks, flag=flag, theta=theta,alpha=alpha,beta=beta)([inputlabel,A2, B, C, z_log_var_A, z_log_var_B])
    #####image decoder####
    out=model.Decoder(hidden_dim=hid.shape[1],img_z=img_z)(latent_output)
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

    allmodel.compile(optimizer=optimizer)
    

    callbacks=[LearningRateScheduler(lambda epoch: 0.001 * 0.85 ** (epoch // 5))]
    term=tf.keras.callbacks.TerminateOnNaN()
    

    history1 =allmodel.fit( data_generator.DataGenerator(hdf5file=hdf5_file,image_name=image_name, label_name=label_name,batch_size=batch_size, if_train=True),validation_data=data_generator.DataGenerator(hdf5file=hdf5_file, image_name=image_name, label_name=label_name,batch_size=batch_size, if_train=False),
                            epochs=epoch_num,validation_steps=np.ceil(length/batch_size-1),
                           verbose=1, steps_per_epoch=np.ceil(length/batch_size-1),
                           callbacks=[term,callbacks])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

    
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
