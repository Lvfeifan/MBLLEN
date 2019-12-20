from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Conv2DTranspose, Reshape, Dropout, concatenate, Concatenate, multiply, add, MaxPooling2D, Lambda, Activation, subtract, Flatten, Dense
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras_contrib.layers.normalization import InstanceNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l2
import imageio
from keras import backend as K
from keras.models import Model
from data_load import Dataloader
from keras.utils import plot_model
from scipy import misc
from glob import glob
import tensorflow as tf
import numpy as np
import scipy
import platform
import keras
import os
import random
import Network
import utls

def my_loss(y_true, y_pred):
    MAE_loss = K.mean(K.abs(y_pred[:,:,:,:3] - y_true))
    SSIM_loss = utls.tf_ssim(tf.expand_dims(y_pred[:, :, :, 0], -1),tf.expand_dims(y_true[:, :, :, 0], -1)) + utls.tf_ssim(
        tf.expand_dims(y_pred[:, :, :, 1], -1), tf.expand_dims(y_true[:, :, :, 1], -1)) + utls.tf_ssim(
        tf.expand_dims(y_pred[:, :, :, 2], -1), tf.expand_dims(y_true[:, :, :, 2], -1))
    VGG_loss = K.mean(K.abs(y_pred[:, :, :, 3:19] - y_pred[:, :, :, 19:35]))

    percent = 0.4
    index = int(256 * 256 * percent - 1)
    gray1 = 0.39 * y_pred[:, :, :, 0] + 0.5 * y_pred[:, :, :, 1] + 0.11 * y_pred[:, :, :, 2]
    gray = tf.reshape(gray1, [-1, 256 * 256])
    gray_sort = tf.nn.top_k(-gray, 256 * 256)[0]
    yu = gray_sort[:, index]
    yu = tf.expand_dims(tf.expand_dims(yu, -1), -1)
    mask = tf.to_float(gray1 <= yu)
    mask1 = tf.expand_dims(mask, -1)
    mask = tf.concat([mask1, mask1, mask1], -1)

    low_fake_clean = tf.multiply(mask, y_pred[:, :, :, :3])
    high_fake_clean = tf.multiply(1 - mask, y_pred[:, :, :, :3])
    low_clean = tf.multiply(mask, y_true[:, :, :, :])
    high_clean = tf.multiply(1 - mask, y_true[:, :, :, :])
    Region_loss = K.mean(K.abs(low_fake_clean - low_clean) * 4 + K.abs(high_fake_clean - high_clean))

    loss = MAE_loss + VGG_loss/3. + 3 - SSIM_loss + Region_loss
    return loss

if not os.path.isdir('./val_images'):
    os.makedirs('./val_images')
if not os.path.isdir('./logs'):
    os.makedirs('./logs')
if not os.path.isdir('./models'):
    os.makedirs('./models')

def f1(x):
    return x[:, :, :, :3]

def f2(x):
    return x[:, :, :, 3:]

def f3(x):
    return tf.reshape(x,[-1, 256, 256, 16])

img_rows = 256
img_cols = 256
img_channels = 3
crop_shape = (img_rows, img_cols, img_channels)
input_shape = (img_rows, img_cols, img_channels*2)
dataset_name = 'dark'
data_loader = Dataloader(dataset_name=dataset_name, crop_shape=(img_rows, img_cols))

# Build the network
mbllen = Network.build_mbllen(crop_shape)
# mbllen.load_weights('./1_dark2_color_identity_param.h5')

Input_MBLLEN = Input(shape=input_shape)
img_A = Lambda(f1)(Input_MBLLEN)
img_B = Lambda(f2)(Input_MBLLEN)

# VGG19 feature, content loss
vgg = Network.build_vgg()
vgg.trainable = False

fake_B = mbllen(img_A)
vgg_fake = Lambda(utls.range_scale)(fake_B)
fake_features = vgg(vgg_fake)
fake_features = Lambda(f3)(fake_features)

img_B_vgg = Lambda(utls.range_scale)(img_B)
imgb_features = vgg(img_B_vgg)
imgb_features = Lambda(f3)(imgb_features)

output_com = concatenate([fake_B, fake_features, imgb_features], axis=3)

opt = Adam(lr=1*1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
combined = Model(inputs=Input_MBLLEN, outputs=output_com)
combined.compile(loss=my_loss,
                 metrics=[utls.bright_mae, utls.bright_mse, utls.bright_psnr, utls.bright_SSIM, utls.bright_AB],
                 optimizer=opt)

# plot_model(mbllen, to_file='./model.png', show_shapes=True)
combined.summary()

def scheduler(epoch):
    lr = K.eval(combined.optimizer.lr)
    print("LR =", lr)
    lr = lr * 0.99
    return lr

num_epoch = 0
class Show_History(keras.callbacks.Callback):
    def on_epoch_end(self, val_loss=None, logs=None):
        # save model
        global num_epoch
        global dataset_name
        global img_rows
        global img_cols
        global mbllen
        num_epoch += 1
        modelname = './models/' + str(num_epoch) + '_' + dataset_name + '_base.h5'
        mbllen.save_weights(modelname)

        # test val data
        path = glob('../dataset/test/*.jpg')
        number = 0
        psnr_ave = 0

        for i in range(len(path)):
            if i>15:
                break

            img_B_path = path[i]
            img_B = utls.imread_color(img_B_path)

            path_mid = os.path.split(img_B_path)
            path_A_1 = path_mid[0] + '_' + dataset_name
            path_A = os.path.join(path_A_1, path_mid[1])
            img_A = utls.imread_color(path_A)

            nw = random.randint(0, img_A.shape[0] - img_rows)
            nh = random.randint(0, img_A.shape[1] - img_cols)

            crop_img_A = img_A[nw:nw + img_rows, nh:nh + img_cols, :]
            crop_img_B = img_B[nw:nw + img_rows, nh:nh + img_cols, :]

            crop_img_A = crop_img_A[np.newaxis, :]
            crop_img_B = crop_img_B[np.newaxis, :]

            fake_B = mbllen.predict(crop_img_A)
            identy_B = mbllen.predict(crop_img_B)

            out_img = np.concatenate([crop_img_A, fake_B, crop_img_B, identy_B], axis=2)
            out_img = out_img[0, :, :, :]

            fake_B = fake_B[0, :, :, :]
            img_B = crop_img_B[0, :, :, :]

            clean_psnr = utls.psnr_cau(fake_B, img_B)
            L_psnr = ("%.4f" % clean_psnr)

            number += 1
            psnr_ave += clean_psnr

            filename = os.path.basename(path[i])
            img_name = './val_images/' + str(num_epoch) + '_' + L_psnr + '_' + filename
            utls.imwrite(img_name, out_img)
        psnr_ave /= number
        print('------------------------------------------------')
        print("[Epoch %d]  [PSNR_AVE :%f]" % (num_epoch,  psnr_ave))
        print('------------------------------------------------')

    def on_batch_end(self, batch, logs={}):
        print(' - LR = ', K.eval(self.model.optimizer.lr))
        fileObject = open("./trainList.txt", 'a')
        fileObject.write("%d %f\n" % (batch, logs['loss']))
        fileObject.close()

show_history = Show_History()
change_lr = LearningRateScheduler(scheduler)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                                         embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
nanstop = keras.callbacks.TerminateOnNaN()
reducelearate = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-10)
earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=3, patience=0, verbose=0, mode='min')

batch_size = 16
step_epoch = 200
combined.fit_generator(
        data_loader.load_data(batch_size),
        steps_per_epoch=step_epoch,
        epochs=200,
        callbacks=[tbCallBack, show_history, change_lr, nanstop, reducelearate])
print('Done!')