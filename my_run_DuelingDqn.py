"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

from sample_env_offline import freq_env

import time
from my_dueling_dqn import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

root = tk.Tk()
root.title("matplotlib in TK")
f = Figure(figsize=(6, 6), dpi=100)
canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

env = freq_env()
MEMORY_SIZE = 5000
ACTION_SPACE = 10

sess = tf.Session()
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=160, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=False)

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=160, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())
def load_network():
    checkpoint = tf.train.get_checkpoint_state('')
    if checkpoint and checkpoint.model_checkpoint_path:
        dueling_DQN.saver.restore(dueling_DQN.sess, checkpoint.model_checkpoint_path)
        print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
    else:
        print('Training new network...')

def train(RL):
    acc_r = [0]
    total_steps = 0
    for episode in range(1000):
        observation = env.waterfall_reset()
        r = 0
        step = 0
        while True:
            # if total_steps-MEMORY_SIZE > 9000: env.render()

            action = RL.choose_action(observation)


            observation_, reward, done = env.step(action,observation)
            r += reward
            if episode > 900:
                f.clf()
                waterfall_figure = f.add_subplot(111)
                waterfall_figure.imshow(observation_[0, :, :, 0])
                canvas.draw()

            #reward /= 10      # normalize to a range of (-1, 0)

            #acc_r.append(reward)  # accumulated reward

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()
                #env.render()
            observation = observation_
            step += 1
            total_steps += 1
            if done:
                break
        print(episode,r,step,total_steps,RL.epsilon)
        acc_r.append(r)
    return RL.cost_his, acc_r

c_dueling, r_dueling = train(dueling_DQN)
c_natural, r_natural = train(natural_DQN)

plt.figure(1)
plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()

