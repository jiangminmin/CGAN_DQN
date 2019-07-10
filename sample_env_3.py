#coding=utf-8
if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")
import numpy as np
import time
import threading
from jammer_tx import jammer
from user_tx import user
import usrp_spectrum_sense as usrp_spetrum_sense
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
import matplotlib.pylab as plt
import re
import random
class freq_env(object):
    def __init__(self):
        super(freq_env,self).__init__()
        self.action_space = [i for i in np.arange(100500000.0,110000000.0,500000)]
        self.n_actions = len(self.action_space)
        self.n_features = 1601
        self.user_obj = user()
        self.action_ls = [100000000.0]
        self.start_freq = 129000000.0
        self.end_freq = 130000000.0
        self.tb = usrp_spetrum_sense.my_top_block()
        self.tb.start()
        self.filename='data_100_110.txt'
        self.start_freq = float(re.findall('\d+', self.filename)[0]) * 1000000.0
        self.end_freq = float(re.findall('\d+', self.filename)[1]) * 1000000.0
        self.freq_line = []
        self.power_ls = []
        self.waterfall_bg = self.waterfall_bg()
        self.mode = 1
    def waterfall_bg(self):
        with open(self.filename) as fr:
            for line in fr.readlines():
                curLine = line.strip().split()
                self.freq_line.append(curLine[2])
                self.power_ls.append(float(curLine[3]))
        self.index_start = [i for i, x in enumerate(self.freq_line) if x == str(self.start_freq)]
        self.index_end = [i for i, x in enumerate(self.freq_line) if x == str(self.end_freq)]
    def waterfall_reset(self):
        self.current_line = -50
        self.jamm_pos = []
        self.freq_pow = []
        rand_int = random.randint(0, self.n_actions-1)
        self.action_last = rand_int
        waterfall_reset = np.zeros(shape=(1, 50, 1601, 1), dtype=float)
        self.line_num = list(waterfall_reset.shape)[1]

        self.set_user_freq(self.action_space[rand_int])
        w_f_reset= usrp_spetrum_sense.main(self.tb)

        for i in range(self.line_num):
            self.current_line += 1
            waterfall_reset[0, i, :, 0] = self.power_ls[(self.index_end[0] + 1) * i:(self.index_end[0] + 1) * i + self.index_end[0]+1]
        for i in range(self.n_actions):
            sum_freq = np.sum(waterfall_reset[0, -1, (2*i+1) * 40:(2*i+3) * 40, 0])
            self.freq_pow.append(sum_freq)
            if sum_freq > 400:
                self.jamm_pos.append(i)
        #print(self.freq_pow)
        #print(self.jamm_pos)
        #plt.subplot(211)
        #for i in range(19):
            #plt.axvline((i+1)*80,color='red')
        #plt.plot(waterfall_reset[0,-1])
        waterfall_reset[0, -1, :, 0] = w_f_reset[0,0,:,0]
        #plt.subplot(212)
        #plt.plot(w_f[0, -1])
        #plt.show()
        return waterfall_reset
        pass
    def main_user_tx(self):
        self.user_obj.Start(True)
        self.user_obj.wait()
    def set_user_freq(self, user_freq):
        self.user_obj.set_center_freq(user_freq)
        #print('user_freq change to:=================================================', self.user_obj.get_center_freq())
    def waterfall_next(self, waterfall_last, action):
        waterfall_n = waterfall_last.copy()
        self.set_user_freq(self.action_space[action])
        w_f= usrp_spetrum_sense.main(self.tb)
        for i in range(self.line_num):
            if i < self.line_num -1:
                waterfall_n[0,i,:,0] = waterfall_n[0,i+1,:,0]
            else:
                waterfall_n[0,i,:,0] = w_f[0,0,:,0]
        return waterfall_n
    def step(self, action, observation):
        s_ = self.waterfall_next(observation, action)
        if action in self.jamm_pos:
            reward = -1
            done = True
        elif action == self.action_last:
            reward = 1
            done = False
        else:
            reward = 0.8
            done = False
        self.action_last = action
        return s_, reward, done
    def spect_sense(self):
        root = tk.Tk()
        root.title("matplotlib in TK")
        f = Figure(figsize=(7, 7), dpi=200)
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        x = 0
        while True:
            x = x + 1 if x < 18 else 0
            self.set_user_freq(self.action_space[x])

            w_f= usrp_spetrum_sense.main(self.tb)

            f.clf()
            waterfall_figure = f.add_subplot(111)
            waterfall_figure.plot(w_f[0, -1])
            canvas.draw()

            #time.sleep(1)

    def main_thread(self):
        # ------------------------thread start----------------------------------------
        thread_ls = []
        thread_ls.append(threading.Thread(target=self.main_user_tx))
        for t in thread_ls:
            t.start()
"""
if __name__ == "__main__":
    env = freq_env()
    env.main_thread()

    #env.spect_sense()
    #print(env.index_start)
    #print(env.index_end)
    waterfall_reset = env.waterfall_reset()
    #plt.imshow(waterfall_reset[0,:,:,0])
    #plt.show()
    #plt.subplot(211)
    #plt.plot(waterfall_reset[0,-1])
    #plt.subplot(212)
    #plt.plot(waterfall_reset[0, -2])
    #plt.show()
    """"""
    root = tk.Tk()
    root.title("matplotlib in TK")
    f = Figure(figsize=(6, 6), dpi=300)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    for i in range(1000):
        flag = np.random.randint(0, 18)
        s_, reward, done = env.step(flag, waterfall_reset)
        #print(waterfall_reset == s_)
        waterfall_reset = s_
        f.clf()
        a = f.add_subplot(111)
        a.imshow(s_[0, :, :, 0])
        #a.plot(s_[0,-1*i])
        canvas.draw()
"""