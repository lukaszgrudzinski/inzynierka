import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import imageio
from random import *

full_images=np.zeros((28,1080,1920,3))
for i in range(27):
	full_images[i]= imageio.imread('../hd_images/'+str(i)+'.png')
full_images=full_images/255.0
holes=np.zeros((9,200,200,3))
for i in range(9):
	holes[i]= imageio.imread('../hd_images/tears/'+str(i)+'.png')
import scipy.ndimage
holes2=np.zeros((9,400,400,6))
for i in range(9):
	holes2[i]=scipy.ndimage.zoom(holes[i], 2, order=0)
def imageholes(array):
	for i in range(array.shape[0]):                    		        #robienie dziur 
			if randint(0,4)==0:
				continue
			x=randint(0,680)                                         #robienie dziur 
			y=randint(0,1520) 
			hole=holes2[randint(0,8)]
			for a in range(x,x+400):
				for b in range(y,y+400):
					if hole[a-x,b-y,0]==0:
						array[i,a,b]=-1
				 
	return array
def compare(before,withHole,after,start):
	plt.figure(figsize=(13,13))                                     
	for i in range(5):
		plt.subplot(5,3,i*3+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid('False')
		plt.imshow(before[i+start],vmin=-1, vmax=1)
		
		plt.subplot(5,3,i*3+2)
		plt.xticks([])
		plt.yticks([])
		plt.grid('False')
		plt.imshow(withHole[i+start],vmin=-1, vmax=1)
		
		plt.subplot(5,3,i*3+3)
		plt.xticks([])
		plt.yticks([])
		plt.grid('False')
		plt.imshow(after[i+start])
	plt.show()

def makeoldholes(array):
	for i in range(array.shape[0]):                    		        #robienie dziur 
			if randint(0,5)==0:
				continue
			x=randint(100,1820)                                         #robienie dziur 
			y=randint(100,980) 
			r=randint(70,100)                                          #robienie dziur 
			for j in range(1920):                                     #robienie dziur 
				for k in range(1920):                                 #robienie dziur 
					if j>x-r and j<x+r and k>y-r and k<y+r:         #robienie dziur 
						array[i,k,j,0]=-1                 			#robienie dziur 
						array[i,k,j,1]=-1 
						array[i,k,j,2]=-1 
	return array
withHoles=np.copy(full_images)
withHoles=makeoldholes(withHoles)
withHoles2=imageholes(np.copy(full_images))
#compare(full_images,withHoles,withHoles2,0)

from keras.layers import Input, Dense, Conv3D,Conv2D, MaxPooling3D,MaxPooling2D, UpSampling2D,UpSampling3D, Flatten,Reshape,Permute, AveragePooling1D
from keras.models import Model
from keras import backend as K
import keras

input_img = Input(shape=(1080, 1920, 3))  

x = Reshape([1080,1920,3,1])(input_img)
x=MaxPooling3D(pool_size=(4,4,1),strides=None,padding='valid')(x)
x = Conv3D( 32, (3,3,1), activation='relu', padding='same')(x)
x=MaxPooling3D(pool_size=(2,2,1),strides=None,padding='valid')(x)
x = Conv3D( 128, (3,3,1), activation='relu', padding='same')(x)
x = Permute((4,3,2,1))(x)
x = MaxPooling3D(pool_size=(128,1,1),strides=None,padding='valid')(x)
x = Permute((4,3,2,1))(x)
#
x = Conv3D( 32, (3,3,1), activation='relu', padding='same')(x)
x=MaxPooling3D(pool_size=(2,2,1),strides=None,padding='valid')(x)
x = Conv3D( 128, (3,3,1), activation='relu', padding='same')(x)
x = Permute((4,3,2,1))(x)
x = MaxPooling3D(pool_size=(128,1,1),strides=None,padding='valid')(x)
x = Permute((4,3,2,1))(x)
#
x = Conv3D( 32, (3,3,1), activation='relu', padding='same')(x)
x=MaxPooling3D(pool_size=(2,2,1),strides=None,padding='valid')(x)
x = Conv3D( 128, (3,3,1), activation='relu', padding='same')(x)
x = Permute((4,3,2,1))(x)
x = MaxPooling3D(pool_size=(128,1,1),strides=None,padding='valid')(x)
x = Permute((4,3,2,1))(x)

x = Conv3D( 32, (3,3,1), activation='relu', padding='same')(x)
x=MaxPooling3D(pool_size=(2,2,1),strides=None,padding='valid')(x)
x = Conv3D( 128, (3,3,1), activation='relu', padding='same')(x)
x = Permute((4,3,2,1))(x)
x = MaxPooling3D(pool_size=(128,1,1),strides=None,padding='valid')(x)
x = Permute((4,3,2,1))(x)

size_coded=16*30*3
densik= Flatten()(x)
densik = Dense(int(size_coded*1.2), activation=tf.tanh,use_bias=True)(densik)
densik = Dense(int(size_coded*1.2), activation=tf.tanh,use_bias=True)(densik)
densik = Dense(int(size_coded*1.2), activation=tf.tanh,use_bias=True)(densik)
densik = Dense(int(size_coded*1.2), activation=tf.tanh,use_bias=True)(densik)
densik = Dense(size_coded, activation=tf.tanh,use_bias=True)(densik)

x=Reshape([3,1,16,30])(densik)
x=Permute((3,4,1,2))(x)

x=Conv3D(8,[3,3,1],activation='relu', padding='same')(x)

x = UpSampling3D((2, 2,1))(x)
x=Conv3D(8,[3,3,1],activation='relu', padding='same')(x)

x = UpSampling3D((2, 2,1))(x)
x=keras.layers.ZeroPadding3D(padding=(1, 0,0), data_format=None)(x)
x=Conv3D(8,[3,3,1],activation='relu', padding='same')(x)
x = UpSampling3D((2, 2,1))(x)
x=keras.layers.ZeroPadding3D(padding=(1, 0,0), data_format=None)(x)
x=Conv3D(16,[3,3,1],activation='relu', padding='same')(x)
x = UpSampling3D((2, 2,1))(x)
x=Conv3D(1,[3,3,1],activation='relu', padding='same')(x)
x=keras.layers.ZeroPadding3D(padding=(1, 0,0), data_format=None)(x)
x = UpSampling3D((2, 2,1))(x)
#x=keras.layers.ZeroPadding3D(padding=(1, 0,0), data_format=None)(x)
x=Conv3D(1,[3,3,1],activation='relu', padding='same')(x)
x = UpSampling3D((2, 2,1))(x)
x=Conv3D(1,[3,3,1],activation='relu', padding='same')(x)

decoded=Reshape([1080,1920,3])(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile( loss='mse',optimizer=tf.train.AdamOptimizer(),metrics=['accuracy']) 
autoencoder.summary()

for i in range(10000):
	autoencoder.fit(withHoles, full_images,
                epochs=10,
                batch_size=26,
                shuffle=True)
                #validation_data=(train_images, train_images))
	autoencoder.save('model.h5')	
	print("fitting number:",i)
	withHoles=imageholes(np.copy(full_images))
