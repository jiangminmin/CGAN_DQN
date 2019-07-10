"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import time

from DDPG_env_online import freq_env
from collections import deque
import random
import math

import matplotlib
import matplotlib.pylab as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk


#####################  hyper parameters  ####################

MAX_EPISODES = 10000
MAX_EP_STEPS = 200
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002   # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32
WIDTH = 1601
HEIGHT = 50


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, a_bound,e_greedy_increment=0.001,e_greedy=0.9,):

        self.memory = deque()
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.a_bound = a_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 1], 's')
        self.S_ = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 1], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon_increment = e_greedy_increment

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            action = self.sess.run(self.a, feed_dict={self.S: s[np.newaxis, :]})
        else:
            action = np.random.randint(100500000, 109500001)
        return action
        #return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        #indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])
        action_batch = np.reshape(action_batch, (BATCH_SIZE, 1))
        reward_batch = np.reshape(reward_batch, (BATCH_SIZE, 1))

        self.sess.run(self.atrain, {self.S: state_batch})
        self.sess.run(self.ctrain, {self.S: state_batch, self.a: action_batch, self.R: reward_batch, self.S_: next_state_batch})
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def store_transition(self, s, a, r, s_,done):
        self.memory.append((s, a, r, s_, done))
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.popleft()
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net_1 = tf.layers.conv2d(s, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu, name='l1',
                                     trainable=trainable,)
            net_2 = tf.layers.conv2d(net_1, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu, name='l2',
                                     trainable=trainable)
            net_3 = tf.layers.conv2d(net_2, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu, name='l3',
                                     trainable=trainable)
            flat = tf.layers.flatten(net_3)
            net_4 = tf.layers.dense(flat, 30, activation=tf.nn.relu, name='l4', trainable=trainable)
            a = tf.layers.dense(net_4, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.add(tf.multiply(a, self.a_bound, name='scaled_a'),105000000)

            #net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            #a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            #return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            cnet_1 = tf.layers.conv2d(s, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu, name='cl1',
                                      trainable=trainable,)
            cnet_2 = tf.layers.conv2d(cnet_1, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu, name='cl2',
                                      trainable=trainable)
            cnet_3 = tf.layers.conv2d(cnet_2, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu, name='cl3',
                                      trainable=trainable)
            c_flat = tf.layers.flatten(cnet_3)
            #w1_s = tf.get_variable('w1_s', [2048, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.layers.dense(c_flat,n_l1,activation=tf.nn.relu,name='cnet',trainable=trainable) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
            #w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            #w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            #b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            #net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            #return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################
#tf.reset_default_graph()
env = freq_env()

a_dim = 1
a_bound = 4500000

ddpg = DDPG(a_dim, a_bound)

#var = 10  # control exploration
t1 = time.time()

root = tk.Tk()
root.title("matplotlib in TK")
f = Figure(figsize=(6, 6), dpi=100)
canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

for i in range(MAX_EPISODES):
    s = env.waterfall_reset()
    s_reshape = np.reshape(s, (HEIGHT, WIDTH, 1))
    ep_reward = 0
    done = False
    #for j in range(MAX_EP_STEPS):
    while True:
        # Add exploration noise
        a = ddpg.choose_action(s_reshape)
        #a = np.clip(a, 0, 160)
        #a = np.clip(np.random.normal(a, var), 0, 160)    # add randomness to action selection for exploration
        a = int(round(a))

        s_, r, done = env.step(a,s)
        print("a:",a,"r:",r,"done:",done)
        s_res = np.reshape(s_, (HEIGHT, WIDTH, 1))

        ddpg.store_transition(s_reshape, a, r, s_res,done)

        if ddpg.pointer > MEMORY_CAPACITY:
            #var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        s_reshape = s_res
        ep_reward += r

        if i > MAX_EPISODES*0.95:
            f.clf()
            a = f.add_subplot(111)
            a.imshow(s_[0, :, :, 0], interpolation='nearest', aspect='auto')
            canvas.draw()
        #if j == MAX_EP_STEPS - 1:
        if done:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), ddpg.epsilon, )
            break
print('Running time: ', time.time() - t1)