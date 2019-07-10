from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import os
import itertools
import scipy.misc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from sample_env_offline import freq_env

"""
class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
class gameEnv():
    def __init__(self,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        #plt.imshow(a,interpolation="nearest")
        #plt.show()
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(), 1, 1, 2, None,'hero')
        self.objects.append(hero)
        goal = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal)
        hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)
        goal2 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal2)
        hole2 = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole2)
        goal3 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal4)
        state = self.renderEnv()
        self.state = state
        return state
    def moveChar(self,direction):
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1
        self.objects[0] = hero
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(),1,1,1,1,'goal'))
                else:
                    self.objects.append(gameOb(self.newPosition(),1,1,0,-1,'fire'))
                return other.reward,False
        return 0.0,False
    def renderEnv(self):
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        a = np.stack([b,c,d],axis=2)
        return a
    def step(self,action):
        self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderEnv()
        return state,reward,done
env = gameEnv(size=5)
"""
env=freq_env()

class Qnetwork():
    def __init__(self, h_size):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 8000], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 50, 160, 1])
        self.conv1 = tf.layers.conv2d(inputs=self.imageIn, filters=32, kernel_size=[8, 8], strides=[4, 4],activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=[2, 2],activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=[1, 1],activation=tf.nn.relu)
        #self.conv4 = tf.layers.conv2d(inputs=self.conv3, filters=h_size,kernel_size=[2, 8], strides=[1, 1],activation=tf.nn.relu)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv3, 2, 3)

        self.streamA = tf.layers.flatten(self.streamAC)
        self.streamV = tf.layers.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        #self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        #self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.layers.dense(self.streamA, env.actions)
        self.Value = tf.layers.dense(self.streamV, 1)
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))#mainQN
        self.predict = tf.argmax(self.Qout, 1)#targetQN

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    return np.reshape(states,[8000])
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .9 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 50000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

root = tk.Tk()
root.title("matplotlib in TK")
f = Figure(figsize=(6, 6), dpi=100)
canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        # Reset environment and get first new observation
        s_o = env.waterfall_reset()
        s = processState(s_o)
        d = False
        rAll = 0
        j = 0
        # The Q-Network
        while not d:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 10)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]

            s1_o, r, d = env.step(a,s_o)
            s1 = processState(s1_o)
            """
            f.clf()
            waterfall_figure = f.add_subplot(111)
            waterfall_figure.imshow(s1_o[0,:,:,0])
            canvas.draw()
            """
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ )#* end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,mainQN.actions: trainBatch[:, 1]})

                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s_o = s1_o

            #if d == True:
                #break

        myBuffer.add(episodeBuffer.buffer)
        #jList.append(j)
        rList.append(rAll)
        #print('rAll:',rAll)
        # Periodically save the model.
        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(i,total_steps, np.mean(rList[-10:]), e)

    print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
plt.show()