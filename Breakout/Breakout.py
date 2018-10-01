#Python3
import gym
import numpy as np
import random
from keras.layers import Dense,Conv2D,Multiply,Input,Flatten
from collections import deque
from keras.optimizers import RMSprop
from keras.models import Model,clone_model

env=gym.make('BreakoutDeterministic-v4')
INPUT_SHAPE=(105,80,4)
steps_done=0

def pre_process_image(img):
    img=np.mean(img,axis=2) # convert to grayscale
    img=img[::2, ::2]
    return img

image_input=Input(shape=(105,80,4))
action_done=Input(shape=(4,))

Q_predictor=Conv2D(16,(8,4),activation='relu')(image_input)
Q_predictor=Conv2D(32,(4,2),activation='relu')(Q_predictor)
Q_predictor=Flatten()(Q_predictor)
Q_predictor=Dense(256,activation='relu')(Q_predictor)
Q_predictor=Dense(4)(Q_predictor)

final_output=Multiply()([Q_predictor,action_done])

model=Model(inputs=[image_input,action_done],outputs=final_output)
optimize=RMSprop(lr=0.00025,rho=0.95,epsilon=0.01)
model.compile(loss='mse',optimizer=optimize,metrics=['accuracy'])

target_model=clone_model(model)
target_model.set_weights(model.get_weights())
experience=deque(maxlen=5000)
update_target_model=10000

def action_todo(curr_state):
    curr_state=curr_state/255
    global target_model,steps_done
    epsilon=0
    if steps_done<10000000:
        epsilon=1-0.9*steps_done/10000000
    else:
        epsilon=0.1

    if random.random()<epsilon:
        return random.randint(0,3)
    else:
        predictions=target_model.predict([curr_state, np.ones(4).reshape(1, 4)])
        predictions=predictions[0]
        return np.argmax(predictions)

num_episodes=0
env.reset()
observation,reward,done,info=env.step(1)
observation=pre_process_image(observation)
observation=np.stack((observation,observation,observation,observation),axis=2)
frame_history=np.reshape(observation,(1,105,80,4))
for i in range(5000):
    life=info['ale.lives']
    action=action_todo(frame_history)
    observation,reward,done,info=env.step(action)
    observation=pre_process_image(observation)
    observation=np.reshape(observation,(1,105,80,1))
    next_frame=np.concatenate((observation,frame_history[:,:,:,:3]),axis=3)
    dead=False
    if life>info['ale.lives']:
        dead=True
    experience.append((frame_history,next_frame,action,reward,dead))
    frame_history=next_frame

print("Random Play Over")

while num_episodes<100000:
    done = False
    num_episodes=num_episodes+1
    if num_episodes%100==0:
        print(num_episodes,"done")
    env.reset()
    life_left=5
    #skip few frames at the beginning of the episode
    for i in range(25):
        observation,reward,done,info=env.step(1)

    observation=pre_process_image(observation)
    observation=np.stack((observation,observation,observation,observation),axis=2)
    frame_history=np.reshape(observation,(1,105,80,4))
    while not done:
        steps_done=steps_done+1
        if steps_done%10000==0:
            target_model.set_weights(model.get_weights())

        action=action_todo(frame_history)
        observation,reward,done,info=env.step(action)
        observation=pre_process_image(observation)
        observation=np.reshape(observation,(1,105,80,1))
        next_frame=np.concatenate((observation,frame_history[:,:,:,:3]),axis=3)
        dead=False
        if life_left>info['ale.lives']:
            dead=True
            life_left=info['ale.lives']
            print("Death at step ",steps_done)
        experience.append((frame_history,next_frame,action,reward,dead))
        frame_history=next_frame

        train_batch=random.sample(experience,32)
        history = np.zeros((32,105,80,4))
        next_history = np.zeros((32,105,80,4))
        target = np.zeros((32,4))
        action_did, reward_got, dead_state = [], [], []

        for i,value in enumerate(train_batch):
            history[i]=value[0]
            next_history=value[1]
            action_did.append(value[2])
            reward_got.append(value[3])
            dead_state.append(value[4])

        action_mask=np.ones((32,4))
        history=history/255
        Q_next_state=model.predict([history,action_mask])

        action_mask=np.zeros((32,4))
        for i in range(32):
            action_mask[i][action_did[i]]=1
            if dead_state:
                target[i][action_did[i]]=-1
            else:
                target[i][action_did[i]]=reward_got[i]+0.99*np.max(Q_next_state[i])

        model.fit([history, action_mask], target, epochs=1,batch_size=32, verbose=0)
