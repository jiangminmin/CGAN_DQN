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
import math
import copy

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
        #self.user_freq = [100,100,100,100,
                          #100,100,100,100,
                          #100,100,100,100,
                          #100,100,100,100]
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
        self.mode = 1

    # ------------------------------generate 160x160 waterfall-------------------------------
    def waterfall_bg(self):
        with open(self.filename) as fr:
            for line in fr.readlines():
                curLine = line.strip().split()
                self.freq_line.append(curLine[2])
                self.power_ls.append(float(curLine[3]))
        self.index_start = [i for i, x in enumerate(self.freq_line) if x == str(self.start_freq)]
        self.index_end = [i for i, x in enumerate(self.freq_line) if x == str(self.end_freq)]
#========================================================sweeping=====================
    def waterfall_reset(self):
        self.current_line = -50
        self.jamm_pos = [0]
        rand_int = random.randint(0, 9)
        self.action_last = rand_int
        waterfall_reset = np.zeros(shape=(1, 50, 160, 1), dtype=float)
        self.line_num = list(waterfall_reset.shape)[1]
        for i in range(self.line_num):
            self.current_line += 1
            self.jamm_pos[0] = i % 10
            waterfall_reset[0,i,:,0] = self.power_ls[(self.index_end[0]+1)*i:(self.index_end[0]+1)*i+self.index_end[0]]
            waterfall_reset[0,i,(i % 10) * 16:((i % 10) + 1) * 16,0] += self.jam_freq[:]
            if i == list(waterfall_reset.shape)[1] - 1:
                waterfall_reset[0, i, rand_int * 16:(rand_int + 1) * 16,0] += self.user_freq[:]
        return waterfall_reset
    def waterfall_next(self, waterfall_last, action):
        waterfall_n = waterfall_last.copy()
        if self.current_line == 450:
            self.current_line = -49
        else:
            self.current_line += 1
        self.jamm_pos[0] = (self.jamm_pos[0] + 1) % 10
        for i in range(self.line_num):
            if i < self.line_num -1:
                waterfall_n[0,i,:,0] = waterfall_n[0,i+1,:,0]
            else:
                waterfall_n[0,i,:,0] = self.power_ls[(self.index_end[0]+1)*(self.line_num-1+self.current_line):
                                                     (self.index_end[0]+1)*(self.line_num-1+self.current_line)+self.index_end[0]]
                waterfall_n[0,i,self.jamm_pos[0] * 16:(self.jamm_pos[0] + 1) * 16,0] += self.jam_freq[:]
                waterfall_n[0,i,action * 16:(action + 1) * 16,0] += self.user_freq[:]
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
#=========================================================combo===============================================
    def waterfall_reset_combo(self):
        self.line_num = 50
        self.current_line = -50
        self.jamm_pos = range(10)[0:10:2]
        self.rand_int = random.randint(0, 9)
        self.action_last = copy.copy(self.rand_int)
        waterfall_reset = np.zeros(shape=(1, 50, 160, 1), dtype=float)
        self.line_num = list(waterfall_reset.shape)[1]
        for i in range(self.line_num):
            self.current_line += 1
            waterfall_reset[0, i, :, 0] = self.power_ls[(self.index_end[0] + 1) * i:(self.index_end[0] + 1) * i + self.index_end[0]]
            for a in self.jamm_pos:
                waterfall_reset[0, i, a*16:(a+1)*16, 0] += self.jam_freq[:]
            if i == list(waterfall_reset.shape)[1] - 1:
                waterfall_reset[0, i, self.rand_int * 16:(self.rand_int + 1) * 16, 0] += self.user_freq[:]
        return waterfall_reset
    def waterfall_next_combo(self,waterfall_last,action):
        waterfall_n = waterfall_last.copy()
        if self.current_line == 450:
            self.current_line = -49
        else:
            self.current_line += 1
        for i in range(self.line_num):
            if i < self.line_num -1:
                waterfall_n[0,i,:,0] = waterfall_n[0,i+1,:,0]
            else:
                waterfall_n[0,i,:,0] = self.power_ls[(self.index_end[0]+1)*(self.line_num-1+self.current_line):
                                                     (self.index_end[0]+1)*(self.line_num-1+self.current_line)+self.index_end[0]]
                for a in self.jamm_pos:
                    waterfall_n[0, i, a * 16:(a + 1) * 16, 0] += self.jam_freq[:]
                waterfall_n[0,i,action * 16:(action + 1) * 16,0] += self.user_freq[:]
        return waterfall_n
        pass
    def step_combo(self, action, observation):
        s_ = self.waterfall_next_combo(observation, action)
        if action in range(10)[0:10:2] or self.rand_int in range(10)[0:10:2]:
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
#=============================================dynamic=============================================
    def waterfall_reset_dynamic(self):
        self.line_num = 50
        self.current_line = -50
        self.jamm_pos=[0]
        self.rand_int = random.randint(0, 9)
        self.action_last = copy.copy(self.rand_int)
        waterfall_reset = np.zeros(shape=(1, 50, 160, 1), dtype=float)
        self.line_num = list(waterfall_reset.shape)[1]
        for i in range(self.line_num):
            self.current_line += 1
            if math.floor(i/10)%2 == 0:
                self.jamm_pos[0] = i % 10
                waterfall_reset[0, i, :, 0] = self.power_ls[(self.index_end[0] + 1) * i:(self.index_end[0] + 1) * i + self.index_end[0]]
                waterfall_reset[0, i, (i % 10) * 16:((i % 10) + 1) * 16, 0] += self.jam_freq[:]
            elif math.floor(i/10)%2 == 1:
                self.jamm_pos = range(10)[0:10:2]
                waterfall_reset[0, i, :, 0] = self.power_ls[(self.index_end[0] + 1) * i:(self.index_end[0] + 1) * i + self.index_end[0]]
                for a in self.jamm_pos:
                    waterfall_reset[0, i, a * 16:(a + 1) * 16, 0] += self.jam_freq[:]
                if i == list(waterfall_reset.shape)[1] - 1:
                    waterfall_reset[0, i, self.rand_int * 16:(self.rand_int + 1) * 16, 0] += self.user_freq[:]
        return waterfall_reset
    def waterfall_next_dynamic(self,waterfall_last,action):
        waterfall_n = waterfall_last.copy()
        if self.current_line == 450:
            self.current_line = -49
        else:
            self.current_line += 1
        for i in range(self.line_num):
            if i < self.line_num - 1:
                waterfall_n[0, i, :, 0] = waterfall_n[0, i + 1, :, 0]
            else:
                waterfall_n[0, i, :, 0] = self.power_ls[
                                          (self.index_end[0] + 1) * (self.line_num - 1 + self.current_line):
                                          (self.index_end[0] + 1) * (self.line_num - 1 + self.current_line) +
                                          self.index_end[0]]
                waterfall_n[0, i, action * 16:(action + 1) * 16, 0] += self.user_freq[:]
                if math.floor((self.current_line-1)/10)%2 == 0:
                    self.jamm_pos = range(10)[0:10:2]
                    for a in self.jamm_pos:
                        waterfall_n[0, i, a * 16:(a + 1) * 16, 0] += self.jam_freq[:]
                else:
                    self.jamm_pos=[(self.current_line-1)%10]
                    waterfall_n[0, i, self.jamm_pos[0] * 16:(self.jamm_pos[0] + 1) * 16, 0] += self.jam_freq[:]
        return waterfall_n
        pass
    def step_dynamic(self,action,observation):
        s_ = self.waterfall_next_dynamic(observation, action)
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
#==========================================intelligent==================================
    def waterfall_reset_intelligent(self):
        self.line_num = 50
        self.current_line = -50
        self.jamm_pos = range(6)
        self.rand_int = random.randint(0, 9)
        self.action_last = copy.copy(self.rand_int)
        waterfall_reset = np.zeros(shape=(1, 50, 160, 1), dtype=float)
        self.line_num = list(waterfall_reset.shape)[1]
        for i in range(self.line_num):
            self.current_line += 1
            waterfall_reset[0, i, :, 0] = self.power_ls[(self.index_end[0] + 1) * i:(self.index_end[0] + 1) * i + self.index_end[0]]
            if math.floor(i/10)%2 == 0:
                self.jamm_pos = range(6)
                for a in self.jamm_pos:
                    waterfall_reset[0, i, a * 16:(a + 1) * 16, 0] += self.jam_freq[:]
            else:
                self.jamm_pos=[0,1,6,7,8,9]
                for a in self.jamm_pos:
                    waterfall_reset[0, i, a * 16:(a + 1) * 16, 0] += self.jam_freq[:]
            if i == list(waterfall_reset.shape)[1] - 1:
                waterfall_reset[0, i, self.rand_int * 16:(self.rand_int + 1) * 16, 0] += self.user_freq[:]
        return waterfall_reset
        pass
    def waterfall_next_intelligent(self, waterfall_last, action):
        waterfall_n = waterfall_last.copy()
        if self.current_line == 450:
            self.current_line = -49
        else:
            self.current_line += 1
        for i in range(self.line_num):
            if i < self.line_num - 1:
                waterfall_n[0, i, :, 0] = waterfall_n[0, i + 1, :, 0]
            else:
                waterfall_n[0, i, :, 0] = self.power_ls[
                                          (self.index_end[0] + 1) * (self.line_num - 1 + self.current_line):
                                          (self.index_end[0] + 1) * (self.line_num - 1 + self.current_line) +
                                          self.index_end[0]]
                waterfall_n[0, i, action * 16:(action + 1) * 16, 0] += self.user_freq[:]
                if math.floor((self.current_line - 1) / 10) % 2 == 0:
                    self.jamm_pos = [0,1,6,7,8,9]
                    for a in self.jamm_pos:
                        waterfall_n[0, i, a * 16:(a + 1) * 16, 0] += self.jam_freq[:]
                else:
                    self.jamm_pos = range(6)
                    for a in self.jamm_pos:
                        waterfall_n[0, i, a * 16:(a + 1) * 16, 0] += self.jam_freq[:]
        return waterfall_n
        pass
    def step_intelligent(self,action,observation):
        s_ = self.waterfall_next_intelligent(observation, action)
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
        pass
"""
if __name__ == "__main__":
    env=freq_env()
    waterfall_reset = env.waterfall_reset_intelligent()
    #plt.imshow(waterfall_reset[0,:,:,0])
    #plt.show()
    #print(waterfall_reset)
    # #min_threshold = env.step(1,waterfall_reset)
    # #print(min_threshold)
    """"""
    root = tk.Tk()
    root.title("matplotlib in TK")
    f = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    a = f.add_subplot(111)
    for i in range(1000):
        flag = np.random.randint(0,10)
        s_, reward, done = env.step_intelligent(flag, waterfall_reset)
        #print(waterfall_reset == s_)
        waterfall_reset = s_
        a.imshow(s_[0,:,:,0])
        canvas.draw()
        time.sleep(0.5)
"""
