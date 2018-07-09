#python3
import keras
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Activation,Flatten
from keras.models import Sequential
import numpy
from keras import optimizers
import gym
from random import *
from keras import backend as K
import tensorflow as tf

with K.tf.device('/gpu:2'):
	#config=tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=4,allow_soft_placement=True,device_count={'GPU':1})
	session=tf.Session()
	K.set_session(session)

max_replay_frames=10000
frames=[]
frames_target=[]
env=gym.make("BreakoutDeterministic-v4")

model=Sequential()

model.add(Conv2D(64,3,input_shape=(210,160,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(4))

optimize=optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimize, loss='mse', metrics=['accuracy'])

current_frame=env.reset()
epsilon=0.8
i=0

while i<10000:
	#env.render()
	print(i)
	epsilon=0.8*(10000-i)/10000
	luck=random()
	if luck < epsilon:
		action=int(random()*4//1)
	else:
		lst=[current_frame]
		lst=numpy.array(lst)
		q=model.predict(lst)
		action=numpy.argmax(q[0])
	new_frame,reward,done,info=env.step(action)
	target=numpy.array([0,0,0,0])
	target[action]=reward
	q_new=model.predict(numpy.array([new_frame]))
	target=target+0.99*q_new
	if len(frames)<max_replay_frames:
		frames.append(current_frame)
		frames_target.append(target)
	else :
		frames.pop(0)
		frames_target.pop(0)
		frames.append(current_frame)
		frames_target.append(target)
	
	train_X=[]
	train_Y=[]
	train_X.append(current_frame)
	train_Y.append(target)
	if len(frames)<=99:
		for j in range(len(frames)):
			train_X.append(frames[j])
			train_Y.append(frames_target[j])
	else:
		indices=random.sample(range(0,len(frames)),99)
		for j in indices:
			train_X.append(frames[j])
			train_Y.append(frames_target[j])
	train_X=numpy.array(train_X)
	train_Y=numpy.array(train_Y)
	train_Y=numpy.squeeze(train_Y)
	print("SHAPE OF TRAIN_S",train_X.shape)
	model.fit(train_X,train_Y,epochs=1)
	i=i+1
	current_frame=new_frame
	
model.save_weights("Breakout.h5")
