# coding=utf-8
"""
if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"
"""
import numpy as np
# import user_tx as user
import time
import threading
# from jammer_tx import jammer
# from user_tx import user
# import usrp_spectrum_sense as usrp_spetrum_sense
import random
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk


class freq_env(object):
    def __init__(self, file_name='data_bg_129_130.txt'):
        super(freq_env, self).__init__()
        # self.action_space = [i for i in np.arange(100000000.0,110000001.0,1000000)]
        self.action_space = range(10)
        self.n_actions = len(self.action_space)
        self.n_features = 160
        # self.jammer_obj = jammer()
        # self.user_obj = user()
        self.action_last = 0
        self.user_freq = [6.94817642133, 11.0710678659, 14.6195817555, 17.3627686085,
                          22.8806895204, 24.2871283707, 24.7753483924, 25.2412643153,
                          25.6009231012, 25.3235259942, 24.7298037759, 23.9569713324,
                          23.1641902552, 19.5581498392, 15.0973001711, 11.5414549903]
        self.jam_freq = [21.4002386604, 26.2925073734, 27.0832711188, 27.103520289,
                         26.8839269717, 26.6711441363, 27.7289883437, 26.680099807,
                         27.0146645565, 27.1471763055, 27.3323130004, 27.330121643,
                         27.0635592862, 26.8182771576, 24.7148691866, 18.9784744044]
        self.filename = file_name
        self.start_freq = float(re.findall('\d+', self.filename)[0]) * 1000000.0
        self.end_freq = float(re.findall('\d+', self.filename)[1]) * 1000000.0
        self.freq_line = []
        self.power_ls = []
        self.current_line = 0
        self.waterfall_bg = self.waterfall_bg()
        self.a = 0
        self.actions = 10

    # ------------------------------generate 160x160 waterfall-------------------------------
    def waterfall_bg(self):
        with open(self.filename) as fr:
            for line in fr.readlines():
                curLine = line.strip().split()
                self.freq_line.append(curLine[2])
                self.power_ls.append(float(curLine[3]))
        self.index_start = [i for i, x in enumerate(self.freq_line) if x == str(self.start_freq)]
        self.index_end = [i for i, x in enumerate(self.freq_line) if x == str(self.end_freq)]

    def waterfall_reset(self):
        self.line_num = 50
        self.current_line = -50
        self.jamm_pos = []
        self.rand_int = random.randint(0, 9)
        self.action_last = self.rand_int
        # print("action_last",self.action_last)
        waterfall_reset = np.zeros(shape=(1, 50, 160, 1), dtype=float)

        for x in range(0, self.line_num):
            for y in range(0, int(self.index_end[0])):
                waterfall_reset[0, x, y] = float(self.power_ls[(int(self.index_end[0]) + 1) * x + y])
        for m in range(0, self.line_num):
            self.current_line += 1
            for j in range(0,10,2):
                for n in range(j * 16, (j + 1) * 16):
                    self.jamm_pos.append(j)
                    waterfall_reset[0, m, n] = waterfall_reset[0, m, n, 0] + self.jam_freq[n - j * 16]
        for f in range(self.rand_int * 16, (self.rand_int + 1) * 16):
            waterfall_reset[0, -1, f] = waterfall_reset[0, -1, f, 0] + self.user_freq[f - self.rand_int * 16]
        return waterfall_reset
        pass

    def waterfall_next(self, waterfall_last, action):
        waterfall_n = waterfall_last.copy()
        if self.current_line == 450:
            self.current_line = -49
        else:
            self.current_line += 1
        # waterfall_next = np.zeros(shape=(1,160,160,1),dtype=float)
        for x in range(0, self.line_num - 1):
            for y in range(0, int(self.index_end[0])):
                waterfall_n[0, x, y] = waterfall_n[0, x + 1, y]
        for m in range(0, int(self.index_end[0])):
            waterfall_n[0, -1, m] = float(self.power_ls[(int(self.index_end[0]) + 1) * (self.line_num - 1 + self.current_line) + m])
        #self.jamm_pos = (self.jamm_pos + 1) % 10
        for j in range(0,10,2):
            for n in range(j * 16, (j + 1) * 16):
                waterfall_n[0, -1, n] = waterfall_n[0, -1, n, 0] + self.jam_freq[n - j * 16]

        #for j in range(len(self.jamm_pos)):
            #waterfall_n[0, -1, self.jamm_pos[j] * 16:(self.jamm_pos[j] + 1) * 16,0] = waterfall_n[0, -1, self.jamm_pos[j] * 16:(self.jamm_pos[j] + 1) * 16, 0] + self.jam_freq[:]

        for f in range(action * 16, (action + 1) * 16):
            waterfall_n[0, -1, f] = waterfall_n[0, -1, f, 0] + self.user_freq[f - action * 16]

        return waterfall_n
        pass

    def step(self, action, observation):
        s_ = self.waterfall_next(observation, action)
        if action in [0,10,2] and self.rand_int in [0,10,2]:
            reward = -1
            done = True
        elif action == self.action_last:
            reward = 1
            done = False
        else:
            reward = 0.7
            done = False
        self.action_last = action
        return s_, reward, done


""""""
if __name__ == "__main__":
    env=freq_env()
    waterfall_reset = env.waterfall_reset()
    #print(waterfall_reset)
    # #min_threshold = env.step(1,waterfall_reset)
    # #print(min_threshold)

    #print(waterfall_reset)
    #plt.imshow(waterfall_reset[0,:,:,0])
    #plt.show()
    root = tk.Tk()
    root.title("matplotlib in TK")
    f = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    a = f.add_subplot(111)
    for i in range(100):
        flag = np.random.randint(0,10)
        s_, reward, done = env.step(flag, waterfall_reset)
        waterfall_reset = s_
        a.imshow(s_[0,:,:,0])
        canvas.draw()
        time.sleep(0.5)

    #print(s_==waterfall_reset)
    #s_ = env.waterfall_next(waterfall_reset,1)
    #plt.imshow(s_[0,:,:,0])
    #plt.show()

	#print(waterfall_reset==s_)
	#plt.plot(waterfall_next[0,1])
	#for i in range(1,11):
		#plt.axvline(16*i,color='r')
	#plt.show()
