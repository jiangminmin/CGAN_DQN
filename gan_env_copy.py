from __future__ import division
import numpy as np
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage import transform,data
import tkinter as tk
from gan_load_model import generator
import tensorflow as tf
import matplotlib.pylab as plt

class freq_env(object):
    def __init__(self, file_name='data_bg_129_130.txt'):
        super(freq_env, self).__init__()
        self.action_space = np.linspace(80,1520,19,dtype=np.int)
        self.n_actions = len(self.action_space)
        self.n_features = 160
        self.action_last = 0
        self.gen = generator()
        self.model = self.gen.create_model()
        self.current_line = -1
        self.jamm_pos = 0
        self.lines = 100
        self.interval = 5
    def waterfall_reset(self,sess):
        sweep = np.zeros(shape=(self.lines, 1601))

        rand_1 = np.random.randint(0, 4)
        sweep[:, 86 + rand_1:105 + rand_1] = 2
        rand_2 = np.random.randint(0, 4)
        sweep[:, 277 + rand_2:297 + rand_2] = 2
        rand_3 = np.random.randint(0, 4)
        sweep[:, 395 + rand_3:408 + rand_3] = 2
        rand_4 = np.random.randint(-2, 5)
        sweep[:, 611 + rand_4:638 + rand_4] = 2
        rand_5 = np.random.randint(0, 2)
        sweep[:, 944 + rand_5:952 + rand_5] = 2
        rand_6 = np.random.randint(0, 1)
        sweep[:, 961 + rand_6:990 + rand_6] = 2
        rand_7 = np.random.randint(0, 2)
        sweep[:, 1000 + rand_7:1009 + rand_7] = 2
        rand_8 = np.random.randint(0, 4)
        sweep[:, 1046 + rand_8:1065 + rand_8] = 2
        rand_9 = np.random.randint(0, 4)
        sweep[:, 1157 + rand_9:1180 + rand_9] = 2
        rand_10 = np.random.randint(0, 2)
        sweep[:, 1195 + rand_10:1205 + rand_10] = 2

        for i in range(self.lines):
            randint = np.random.randint(0, 10)
            sweep[i,self.action_space[(i//self.interval)%self.n_actions]-5-randint:self.action_space[(i//self.interval)%self.n_actions]+6+randint]+=1
            self.current_line = (self.current_line + 1) % self.interval
            self.jamm_pos = self.action_space[(i//self.interval)%self.n_actions]

        l = np.array(sweep[0:self.lines, :])
        l = l * 255 / 3

        waterfall_reset = sess.run(self.model.outputs, feed_dict={self.model.my_inputs: l})
        waterfall_reset = np.reshape(waterfall_reset, (256, 256))
        waterfall_reset = transform.resize(waterfall_reset, (100, 1601), mode='constant')
        #waterfall_reset = waterfall_reset * 24
        #waterfall_reset = (waterfall_reset - np.min(waterfall_reset)) / (np.max(waterfall_reset) - np.min(waterfall_reset)) * 255
        return waterfall_reset,sweep

    def waterfall_next(self,waterfall_reset,waterfall_label,action,sess):
        waterfall_n = waterfall_label.copy()
        for i in range(self.lines):
            if i < self.lines -1:
                waterfall_n[i,:] = waterfall_n[i+1,:]
            else:
                waterfall_n[i,:] = 0
                rand_1 = np.random.randint(0, 4)
                waterfall_n[i, 86+rand_1:105+rand_1] = 2
                rand_2 = np.random.randint(0, 4)
                waterfall_n[i, 277+rand_2:297+rand_2] = 2
                rand_3 = np.random.randint(0, 4)
                waterfall_n[i, 395+rand_3:408+rand_3] = 2
                rand_4 = np.random.randint(-2, 5)
                waterfall_n[i, 611+rand_4:638+rand_4] = 2
                rand_5 = np.random.randint(0, 2)
                waterfall_n[i, 944+rand_5:952+rand_5] = 2
                rand_6 = np.random.randint(0, 1)
                waterfall_n[i, 961+rand_6:990+rand_6] = 2
                rand_7 = np.random.randint(0, 2)
                waterfall_n[i, 1000+rand_7:1009+rand_7] = 2
                rand_8 = np.random.randint(0, 4)
                waterfall_n[i, 1046+rand_8:1065+rand_8] = 2
                rand_9 = np.random.randint(0, 4)
                waterfall_n[i, 1157+rand_9:1180+rand_9] = 2
                rand_10 = np.random.randint(0, 2)
                waterfall_n[i, 1195+rand_10:1205+rand_10] = 2

                randInt = np.random.randint(0,10)
                randInt_1 = np.random.randint(0, 10)
                if self.current_line == self.interval-1:
                    self.jamm_pos = self.jamm_pos % 1520 + 80
                    waterfall_n[i,self.jamm_pos - 5 - randInt:self.jamm_pos + 6 + randInt] += 1
                else:
                    waterfall_n[i,self.jamm_pos - 5 - randInt:self.jamm_pos + 6 + randInt] += 1
                #user
                waterfall_n[i,80*(action+1) - 5 - randInt_1:80*(action+1) + 6 + randInt_1] += 1
        l = np.array(waterfall_n[0:self.lines, :])
        l = l * 255 / 3
        self.current_line = (self.current_line + 1) % self.interval

        #waterfall_next = self.gen.generate_img(l)
        waterfall_next = sess.run(self.model.outputs, feed_dict={self.model.my_inputs: l})
        waterfall_next = np.reshape(waterfall_next, (256, 256))
        waterfall_next = transform.resize(waterfall_next, (100, 1601), mode='constant')
        #waterfall_next = waterfall_next *24
        #waterfall_next = (waterfall_next-np.min(waterfall_next))/(np.max(waterfall_next)-np.min(waterfall_next)) * 255
        #for i in range(1601):
            #print(waterfall_next[0,i])
        #for i in range(19):
            #if sum(waterfall_next[-1,self.action_space[i]-10 : self.action_space[i]+11]) > 1000:
                #print(i)
            #print(sum(waterfall_next[-1,self.action_space[i]-10 : self.action_space[i]+11]))
        waterfall_next[0:-1,:] = waterfall_reset[1:,:]
        return waterfall_next,waterfall_n

    def step(self,waterfall_reset,observation,action,sess):
        s_,s_label = self.waterfall_next(waterfall_reset,observation, action,sess)
        if (action+1)*80 == self.jamm_pos or action in [0,4,7,11,12,14] or (self.action_last +1)*80 == self.jamm_pos:
            reward = -1
            done = True
        elif action == self.action_last:
            reward = 1
            done = False
        else:
            reward = 0.8
            done = False
        #print(self.jamm_pos,action,reward,done)
        self.action_last = action
        return s_,s_label, reward, done
"""
if __name__ == "__main__":
    env=freq_env()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint("debug_train")
    with tf.Session() as sess:
        print("loading model from checkpoint")
        saver.restore(sess, checkpoint)

        waterfall_reset,waterfall_label = env.waterfall_reset(sess)

        #waterfall_next,waterfall_n = env.waterfall_next(waterfall_label,9,sess)

        #plt.imshow(waterfall_next,interpolation='nearest',aspect='auto')
        #plt.show()
        """"""
        # #min_threshold = env.step(1,waterfall_reset)
        # #print(min_threshold)

        root = tk.Tk()
        root.title("matplotlib in TK")
        f = Figure(figsize=(6, 6), dpi=100)
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        for i in range(1000):
            flag = np.random.randint(0,19)
            waterfall_reset,s_label, reward, done = env.step(waterfall_reset,waterfall_label,flag,sess)
            #print(waterfall_reset == s_)
            waterfall_label = s_label
            f.clf()
            a = f.add_subplot(111)
            a.imshow(waterfall_reset[:,:],interpolation='nearest',aspect='auto')
            canvas.draw()
            #time.sleep(2)
"""