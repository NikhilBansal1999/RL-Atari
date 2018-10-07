#Python3
import gym
import numpy as np
import random
from keras.layers import Dense,Conv2D,Multiply,Input,Flatten,Lambda
from collections import deque
from keras.optimizers import RMSprop
from keras.models import Model,clone_model
from keras import backend as K
from skimage.color import rgb2gray
from skimage.transform import resize
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

env=gym.make('PongDeterministic-v4')
INPUT_SHAPE=(84,84,4)
steps_done=0

def pre_process_image(img):
    img=np.uint8(resize(rgb2gray(img), (84, 84), mode='constant') * 255)
    return img

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

image_input=Input(shape=(84,84,4))
action_done=Input(shape=(5,))

Q_predictor =Lambda(lambda x: x / 255.0)(image_input)
Q_predictor=Conv2D(16,(8,8),strides=(4,4),activation='relu')(Q_predictor)
Q_predictor=Conv2D(32,(4,4),strides=(2,2),activation='relu')(Q_predictor)
Q_predictor=Flatten()(Q_predictor)
Q_predictor=Dense(256,activation='relu')(Q_predictor)
Q_predictor=Dense(5)(Q_predictor)

final_output=Multiply()([Q_predictor,action_done])

model=Model(inputs=[image_input,action_done],outputs=final_output)
optimize=RMSprop(lr=0.00025,rho=0.95,epsilon=0.01)
model.compile(loss=huber_loss,optimizer=optimize,metrics=['accuracy'])

target_model=clone_model(model)
target_model.set_weights(model.get_weights())
experience=deque(maxlen=200000)
update_target_model=10000

def action_todo(curr_state):
    global target_model,steps_done
    epsilon=0
    if steps_done<1000000:
        epsilon=1-0.9*steps_done/1000000
    else:
        epsilon=0.1

    if random.random()<epsilon:
        return random.randint(0,4)
    else:
        predictions=target_model.predict([curr_state, np.ones(5).reshape(1, 5)])
        predictions=predictions[0]
        return np.argmax(predictions)

num_episodes=0
env.reset()
observation,reward,done,info=env.step(1)
observation=pre_process_image(observation)
observation=np.stack((observation,observation,observation,observation),axis=2)
frame_history=np.reshape(observation,(1,84,84,4))
for i in range(50000):
    life=info['ale.lives']
    if i%10000==0:
        print(i,"done")
    action=action_todo(frame_history)
    observation,reward,done,info=env.step(action)
    observation=pre_process_image(observation)
    observation=np.reshape(observation,(1,84,84,1))
    next_frame=np.concatenate((observation,frame_history[:,:,:,:3]),axis=3)
    dead=False
    if life>info['ale.lives']:
        dead=True
    experience.append((frame_history,next_frame,action,reward,dead))
    frame_history=next_frame

print("Random Play Over")

while num_episodes<100000:
    done = False
    dead = False
    num_episodes=num_episodes+1
    print("Episode",num_episodes,"started")
    if num_episodes%20==0 or num_episodes==1:
        model.save_weights('atari_model_pong.h5')
        fhand=open("episodes.txt","a")
        fhand.write(str(num_episodes)+" episodes completed\n")
        fhand.write(str(steps_done)+" steps_done\n")
        fhand.close()
    env.reset()
    life_left=5
    #skip few frames at the beginning of the episode
    for i in range(random.randint(1,30)):
        observation,reward,done,info=env.step(1)

    observation=pre_process_image(observation)
    observation=np.stack((observation,observation,observation,observation),axis=2)
    frame_history=np.reshape(observation,(1,84,84,4))
    while not done:
        steps_done=steps_done+1
        if steps_done%10000==0:
            target_model.set_weights(model.get_weights())
            fhand=open("episodes.txt","a")
            fhand.write("Target Model weights reset\n")
            fhand.close()

        action=action_todo(frame_history)
        action_asked=action+1
        observation,reward,done,info=env.step(action_asked)
        observation=pre_process_image(observation)
        observation=np.reshape(observation,(1,84,84,1))
        next_frame=np.append(observation,frame_history[:,:,:,:3],axis=3)
        dead=False
        if life_left>info['ale.lives']:
            dead=True
            life_left=info['ale.lives']
        experience.append((frame_history,next_frame,action,reward,dead))

        if not dead:
            frame_history=next_frame

        train_batch=random.sample(experience,32)
        history = np.zeros((32,84,84,4))
        next_history = np.zeros((32,84,84,4))
        target = np.zeros((32,))
        action_did, reward_got, dead_state = [], [], []

        for i,value in enumerate(train_batch):
            history[i]=value[0]
            next_history[i]=value[1]
            action_did.append(value[2])
            reward_got.append(value[3])
            dead_state.append(value[4])

        action_mask=np.ones((32,5))
        Q_next_state=model.predict([next_history,action_mask])
        for i in range(32):
            if dead_state[i]:
                target[i]=-1
            else:
                target[i]=reward_got[i]+0.99*np.amax(Q_next_state[i])

        action_rep=np.eye(5)[np.array(action_did).reshape(-1)]
        target_values=action_rep*target[:,None]

        model.fit([history, action_rep], target_values, epochs=1,batch_size=32, verbose=0)
