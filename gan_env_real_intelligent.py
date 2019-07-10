from __future__ import division
if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"
import usrp_spectrum_sense_intelligent as usrp_spetrum_sense
from user_tx import user
from jammer_tx import jammer
import random
import threading
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

        self.jamm_pos = 80
        self.lines = 100
        self.interval = 5

        self.tb = usrp_spetrum_sense.my_top_block()
        self.tb.start()
        self.user_obj = user()
        self.jammer_obj = jammer()

    def main_user_tx(self):
        self.user_obj.Start(True)
        self.user_obj.wait()

    def main_jammer_tx(self):
        freq_ls = np.linspace(100500000, 109500000, 19)
        self.jammer_obj.Start(True)
        while 1:
            for freq_index in range(len(freq_ls)):
                self.jammer_obj.set_center_freq(freq_ls[freq_index])
                time.sleep(2.5)

    def main_thread(self):
        thread_ls = []
        thread_ls.append(threading.Thread(target=self.main_user_tx))
        thread_ls.append(threading.Thread(target=self.main_jammer_tx))
        for t in thread_ls:
            t.start()

    def waterfall_reset(self,):
        waterfall_reset = np.zeros(shape=(self.lines, 1601))
        waterfall_reset[0:-1,:] = usrp_spetrum_sense.main(self.tb, 99)
        self.user_obj.set_center_freq(100000000 + 6250 * self.action_space[random.randint(0,18)])
        waterfall_reset[-1,:] = usrp_spetrum_sense.main(self.tb, 1)
        self.jamm_pos = int(self.jammer_obj.get_center_freq())
        print(self.jamm_pos)
        return waterfall_reset

    def waterfall_next(self,waterfall_last,action):
        waterfall_next = waterfall_last.copy()
        waterfall_next[0:-1,:] = waterfall_next[1:,:]
        self.user_obj.set_center_freq(100000000 + 6250 * self.action_space[action])
        waterfall_next[-1,:] = usrp_spetrum_sense.main(self.tb, 1)
        self.jamm_pos = (int(self.jammer_obj.get_center_freq()) - 100000000)/6250
        print(self.jamm_pos)
        return waterfall_next

    def step(self,observation,action):
        s_ = self.waterfall_next(observation, action)
        if (action+1)*80 == self.jamm_pos or action in [0,1,4,7,11,12]:
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
        return s_, reward, done
""""""
if __name__ == "__main__":
    env=freq_env()
    env.main_thread()
    waterfall_reset= env.waterfall_reset()
    #waterfall_next = env.waterfall_next(waterfall_reset,9)
    #plt.imshow(waterfall_next,interpolation='nearest',aspect='auto')
    #plt.show()
    """"""
    root = tk.Tk()
    root.title("matplotlib in TK")
    f = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    for i in range(1000):
        flag = np.random.randint(0,19)
        s_, reward, done = env.step(waterfall_reset,flag)
        waterfall_reset = s_
        f.clf()
        a = f.add_subplot(111)
        a.imshow(s_[:,:],interpolation='nearest',aspect='auto')
        canvas.draw()
        #time.sleep(0.5)

