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
def makeAHole(array,x,y):
	if randint(0,100)>80:
		array[x,y,0]=-1
		array[x,y,1]=-1
		array[x,y,2]=-1
		if x>4:
			makeAHole(array,x-1,y)
		if x<1916:
			makeAHole(array,x+1,y)
		if y>4:
			makeAHole(array,x,y-1)
		if x<1076:
			makeAHole(array,x,y+1)
	return array
def makeholes(array):
	for i in range(array.shape[0]):                    		        
			if randint(0,2)==0:
				continue
			x=randint(100,1820)                                         
			y=randint(100,980)                                          
			array[i]=makeAHole(array[i],y,x)               			
	return array
def makeoldholes(array):
	for i in range(array.shape[0]):                    		        
			if randint(0,5)==0:
				continue
			x=randint(100,980)                                         
			r=randint(70,100)                                          
			for j in range(1080):                                     
				for k in range(1920):                                  
					if j>x-r and j<x+r and k>x-r and k<x+r:         
						array[i,j,k,0]=-1                 			
						array[i,j,k,1]=-1 
						array[i,j,k,2]=-1 
	return array
withHoles=np.copy(full_images)
withHoles=makeoldholes(withHoles)
compare(full_images,withHoles,full_images,0)



from keras.layers import Input, Dense, Conv3D,Conv2D, MaxPooling3D,MaxPooling2D, UpSampling2D,UpSampling3D, Flatten,Reshape,Permute, AveragePooling1D
from keras.models import Model
from keras import backend as K
import keras

input_img = Input(shape=(1080, 1920, 3))  

x = Reshape([1080,1920,3,1])(input_img)
x = Conv3D( 16, (3,3,1), activation='relu', padding='same')(x)
x = Permute((4,3,2,1))(x)
x = MaxPooling3D(pool_size=(16,1,1),strides=None,padding='valid')(x)
x = Permute((4,3,2,1))(x)
#
#x = Conv3D( 16, (3,3,1), activation='relu', padding='same')(x)
#x = Permute((4,3,2,1))(x)
#x = MaxPooling3D(pool_size=(16,1,1),strides=None,padding='valid')(x)
#x = Permute((4,3,2,1))(x)
#
#x = Conv3D( 16, (3,3,1), activation='relu', padding='same')(x)
#x = Permute((4,3,2,1))(x)
#x = MaxPooling3D(pool_size=(16,1,1),strides=None,padding='valid')(x)
#x = Permute((4,3,2,1))(x)
#
#x=MaxPooling3D(pool_size=(6,6,1),strides=None,padding='valid')(x)
# #x = Conv3D( 1, (2,2,1), activation='relu', padding='same')(x)
#size_coded=180*320
#densik= Permute((3,1,2,4))(x)
#densik = Reshape([3,size_coded])(densik)
#densik = Dense(size_coded, activation=tf.tanh,use_bias=True)(densik)
# #densik = Dense(size_coded, activation=tf.tanh,use_bias=True)(densik)
# #densik = Dense(size_coded, activation=tf.tanh,use_bias=True)(densik)
#densik = Dense(size_coded, activation=tf.tanh,use_bias=True)(densik)
#
#x=Reshape([3,1,180,320])(densik)
#x=Permute((3,4,1,2))(x)
#x=Conv3D(8,[3,3,1],activation='relu', padding='same')(x)
#x = UpSampling3D((2, 2,1))(x)
#x=Conv3D(8,[3,3,1],activation='relu', padding='same')(x)
#x = UpSampling3D((2, 2,1))(x)
#x=Conv3D(16,[3,3,1],activation='relu', padding='same')(x)
#x = UpSampling3D((2, 2,1))(x)
#x=Conv3D(1,[3,3,1],activation='relu', padding='same')(x)
#x=Flatten()(x)
#x=Dense(1080*1920*3, activation=tf.tanh,use_bias=True)(x)
#decoded=Reshape([1080,1920,3])(x)
#
#
#

autoencoder = Model(input_img, x)
autoencoder.compile( loss='mse',optimizer=tf.train.AdamOptimizer(),metrics=['accuracy']) 
autoencoder.summary()

#for i in range(10000):
#	autoencoder.fit(makeoldholes(full_images), full_images,
#                epochs=10,
#                batch_size=26,
#                shuffle=True)
#                #validation_data=(train_images, train_images))
#	autoencoder.save('model.h5')	
#	print("fitting number:",i)
#	withHoles=makeoldholes(full_images)
