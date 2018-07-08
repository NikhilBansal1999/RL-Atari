import gym
import numpy as np 
from collections import deque
import keras
import random
from keras import backend as K
import tensorflow as tf

with tf.device('/gpu:1'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True, device_count={'GPU':1, 'CPU':1})
    session = tf.Session(config=config)
    K.set_session(session)
    
def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))

def atari_model(n_actions):

    ATARI_SHAPE = (105, 80, 4)

    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv_1 = keras.layers.convolutional.Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)
    conv_2 = keras.layers.convolutional.Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
    conv_flattened = keras.layers.core.Flatten()(conv_2)
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    output = keras.layers.Dense(n_actions)(hidden)
    filtered_output = keras.layers.multiply([output, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model

def q_iteration(env, model, state, iteration, memory):
    
    epsilon = get_epsilon_for_iteration(iteration)

    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, preprocess(state))

    new_state, reward, done, _ = env.step(action)
    memory.append([preprocess(state), action, preprocess(new_state), reward, done])
    
    batch = random.sample(memory, 32)
    fit_batch(model, batch)

    return new_state, done

def get_epsilon_for_iteration(iteration):

    if iteration < 1000000:
        return 1 - (0.9 * iteration)/1000000
    else:
        return 0.1

def choose_best_action(model, state):

    inputs = [[ np.expand_dims(np.stack([state]*4, axis=2), axis=0), np.expand_dims(np.eye(env.action_space.n)[i], axis=0) ] for i in range(env.action_space.n)]
    outputs = [model.predict(inputs[i]) for i in range(len(inputs))]
    mask = [np.eye(env.action_space.n)[i] for i in range(env.action_space.n)]
    outputs = [np.dot(outputs[i], mask[i]) for i in range(len(outputs))]
    return np.argmax(outputs)


def fit_batch(model, batch):

    inputs = [[ np.stack([x[0]]*4, axis=2), np.eye(env.action_space.n)[x[1]] ] for x in batch]
    a_prime = [np.eye(env.action_space.n)[choose_best_action(model, x[2])] for x in batch]
    y = [( x[3] + 0.99 * model.predict([ np.expand_dims(np.stack([ x[2] ]*4, axis=2), axis=0), np.expand_dims(a_prime[i], axis=0) ]) ) for i, x in enumerate(batch)]
    inp1=[]
    for k in inputs:
        inp1.append(k[0])
    inp1=np.array(inp1)
    inp2=[]
    for k in inputs:
        inp2.append(k[1])
    inp2=np.array(inp2)
    y = np.array(y).squeeze()
    model.fit([inp1, inp2], y)


env = gym.make('BreakoutDeterministic-v4')
memory = deque(maxlen=100000)
model = atari_model(env.action_space.n)

state = env.reset()
for i in range(20000):
    action = env.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    memory.append([preprocess(state), action, preprocess(new_state), reward, done])
    if done:
        state = env.reset()
    else:
        state = new_state

iteration = 1
state = env.reset()

while True:
    new_state, done = q_iteration(env, model, state, iteration, memory)
    if iteration % 1000 == 0:
        print("Iteration: ", iteration)
        model.save_weights("RLagent_weights.h5")
    if done:
        state = env.reset()
        iteration += 1
    else:
        state = new_state
        iteration += 1
