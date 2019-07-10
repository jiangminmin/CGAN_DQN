from sample_env_offline import freq_env
from DQN import DeepQNetwork
import time
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

def run_dqn():
    step = 0
    for episode in range(10000):
        # initial observation
        observation = env.waterfall_reset()
        r = 0
        while True:
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action,observation)
            r += reward
            """"""
            f.clf()
            waterfall_figure = f.add_subplot(111)
            waterfall_figure.imshow(observation_[0, :, :, 0])
            canvas.draw()

            #print("-----------------",observation, action, reward, observation_,"----------------------")
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()
                #if (loss < 1e-3): break

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1
        print("{} episode,{} step,{} reward,{} epsilon".format(episode,step,r,RL.epsilon))

    # end of anti_jamming
    print('anti_jamming over')
    
def load_network():
    checkpoint = tf.train.get_checkpoint_state('')
    if checkpoint and checkpoint.model_checkpoint_path:
        RL.saver.restore(RL.sess, checkpoint.model_checkpoint_path)
        print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
    else:
        print('Training new network...')

if __name__ == "__main__":

    # maze game
    env = freq_env()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.00025,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=100,
                      memory_size=5000,
                      # output_graph=True
                      )
    if RL.Load_model:
        load_network()

    root = tk.Tk()
    root.title("matplotlib in TK")
    f = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    run_dqn()
    RL.plot_cost()