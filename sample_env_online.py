# coding=utf-8
if __name__ == '__main__':
    import ctypes
    import sys

    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"
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
import copy

import matplotlib.pylab as plt
import re
import random


class freq_env(object):
    def __init__(self):
        super(freq_env, self).__init__()
        self.action_space = [i for i in np.arange(100500000.0, 110000000.0, 500000)]
        self.n_actions = len(self.action_space)
        self.user_obj = user()
        self.jammer_obj = jammer()
        self.tb = usrp_spetrum_sense.my_top_block()
        self.tb.start()
        self.waterfall_bg = self.waterfall_background()
        self.main_thread()
        self.waterfall_lib = self.waterfall_lib()
        self.mode = 1

    def waterfall_background(self):
        self.jamm_pos = []
        self.freq_pow = []
        waterfall_bg = usrp_spetrum_sense.main(self.tb,100)
        self.waterfall_bg_mean = np.mean(waterfall_bg[0], 0).reshape(1, 1, 1601, 1)

        for i in range(self.n_actions):
            sum_freq_1 = np.sum(self.waterfall_bg_mean[0, 0, (i + 1) * 80 - 40:(i + 1) * 80 + 41, 0])
            self.freq_pow.append(sum_freq_1)
            # sum_freq_2 = np.sum(waterfall_bg[0, 0, (2 * i + 1) * 40:(2 * i + 3) * 40, 0])
            # sum_freq_3 = np.sum(waterfall_bg[0, 24, (2 * i + 1) * 40:(2 * i + 3) * 40, 0])
            if sum_freq_1 > 280:
                self.jamm_pos.append(i)
        #print(self.freq_pow)
        #print(self.jamm_pos)
        return waterfall_bg

    def main_jammer_tx(self):
        freq_ls = [i for i in np.arange(100500000.0, 110000000.0, 500000)]
        self.jammer_obj.Start(True)
        while 1:
            for freq_index in range(len(freq_ls)):
                self.jammer_obj.set_center_freq(freq_ls[freq_index])
                # print('jammer_freq:', self.jammer_obj.get_center_freq())
                time.sleep(2.5)

    def waterfall_lib(self):
        waterfall_lib = np.zeros(shape=(self.n_actions, 50, 1601, 1), dtype=float)
        for i in range(self.n_actions):
            self.set_user_freq(self.action_space[i])
            waterfall = usrp_spetrum_sense.main(self.tb,50)
            waterfall_lib[i] = waterfall
        t = threading.Thread(target=self.main_jammer_tx)
        t.start()
        return waterfall_lib

    def main_user_tx(self):
        self.user_obj.Start(True)
        self.user_obj.wait()

    def set_user_freq(self, user_freq):
        self.user_obj.set_center_freq(user_freq)
        # print('user_freq change to:=================================================', self.user_obj.get_center_freq())

    def waterfall_reset(self):
        waterfall_reset = self.waterfall_bg.copy()
        self.line_num = list(waterfall_reset.shape)[1]
        rand_int = random.randint(0, self.n_actions - 1)
        rand_int_2 = random.randint(0, 49)
        pow_ls = []
        for i in range(self.n_actions):
            pow = np.sum(waterfall_reset[0, -1, (i + 1) * 80 - 40:(i + 1) * 80 + 41, 0])
            pow_ls.append(pow)
        min_index = pow_ls.index(min(pow_ls))
        self.action_last = rand_int
        waterfall_reset[0, -1] = self.waterfall_lib[rand_int, rand_int_2].copy()

        return waterfall_reset
        pass

    def waterfall_next(self, waterfall_last, action):
        waterfall_n = waterfall_last.copy()
        sub_pow_ls = []
        jamm_ls = copy.copy(self.jamm_pos)

        self.set_user_freq(self.action_space[action])
        w_f = usrp_spetrum_sense.main(self.tb, 1)
        waterfall_lib_mean = np.mean(self.waterfall_lib[action], 0).reshape(1, 1, 1601, 1)
        sub_waterfall = w_f - waterfall_lib_mean
        for i in range(self.n_actions):
            pow_ = np.sum(sub_waterfall[0, 0, (i + 1) * 80 - 2:(i + 1) * 80 + 3, 0])
            sub_pow_ls.append(pow_)
            if pow_ > 40:
                jamm_ls.append(i)
        for i in range(self.line_num):
            if i < self.line_num - 1:
                waterfall_n[0, i] = waterfall_n[0, i + 1]
            else:
                waterfall_n[0, i] = w_f
        return waterfall_n, jamm_ls

    def step(self, action, observation):
        s_, jamm_ls = self.waterfall_next(observation, action)
        #print(action,jamm_ls)
        if action in jamm_ls:
            reward = -1
            done = True
        elif action == self.action_last:
            reward = 1
            done = False
        else:
            reward = 0.8
            done = False
        self.action_last = action
        #print(done)
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
            w_f = usrp_spetrum_sense.main(self.tb,50)
            f.clf()
            waterfall_figure = f.add_subplot(111)
            waterfall_figure.plot(w_f[0, -1])
            canvas.draw()

    def main_thread(self):
        thread_ls = []
        thread_ls.append(threading.Thread(target=self.main_user_tx))
        for t in thread_ls:
            t.start()

"""
if __name__ == "__main__":
    env = freq_env()
    #waterfall_bg = env.waterfall_background()
    #env.main_thread()
    #waterfall_lib = env.waterfall_lib()
    # env.spect_sense()
    # print(env.index_start)
    # print(env.index_end)
    waterfall_reset = env.waterfall_reset()

    # waterfall_next,a = env.waterfall_next(waterfall_reset,1)
    # plt.imshow(waterfall_reset[0,:,:,0])
    # plt.show()
    # plt.subplot(211)
    # plt.plot(waterfall_reset[0,-1])
    # plt.subplot(212)
    # plt.plot(waterfall_reset[0, -2])
    # plt.show()
    """"""
    root = tk.Tk()
    root.title("matplotlib in TK")
    f = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    for i in range(1000):
        flag = random.randint(0, env.n_actions - 1)
        s_, reward, done = env.step(flag, waterfall_reset)
        # print(waterfall_reset == s_)
        waterfall_reset = s_
        f.clf()
        a = f.add_subplot(111)
        a.imshow(s_[0, :, :, 0], interpolation='nearest', aspect='auto')
        # a.plot(s_[0,-1*i])
        canvas.draw()
"""