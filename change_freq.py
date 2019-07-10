#coding=utf-8
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
class freq_env(object):
	def __init__(self):
		super(freq_env,self).__init__()
		self.action_space = [i for i in np.arange(100000000.0,110000001.0,1000000)]
		self.n_actions = len(self.action_space)
		self.n_features = 1601
		self.jammer_obj = jammer()
		self.user_obj = user()
		self.action_ls = [100000000.0]
		self.start_freq = 129000000.0
		self.end_freq = 130000000.0
		self.tb = usrp_spetrum_sense.my_top_block()
		self.tb.start()
		self.wf, self.min_index, self.user_f, self.sum_ls = usrp_spetrum_sense.main(self.tb)
	"""
	def reset(self):
		self.action_ls=[100000000.0]
		self.set_user_freq(100000000.0)
		self.s,self.userfreq = usrp_spetrum_sense.main()
		return self.s
	"""
	def main_jammer_tx(self):
		freq_ls = [i for i in np.arange(120000000.0, 140000001.0, 1000000)]
		self.jammer_obj.Start(True)
		while 1:
			for freq_index in range(len(freq_ls)):
				self.jammer_obj.set_center_freq(freq_ls[freq_index])
				time.sleep(4)
				#print('jammer_freq:', self.jammer_obj.get_center_freq())
		# tb.wait()
	def get_jammer_freq(self):
		return self.jammer_obj.get_center_freq()
	def main_user_tx(self):
		self.user_obj.Start(True)
		self.user_obj.wait()
	def set_user_freq(self,user_freq):
		self.user_obj.set_center_freq(user_freq)
		print('user_freq change to:', self.user_obj.get_center_freq())
	def spect_sense(self):
		while True:
			self.set_user_freq(self.user_f)
			w_f,min_i,userfreq,sumls=usrp_spetrum_sense.main(self.tb)
			x=3
			if self.min_index == 0 or self.min_index == 1 or self.min_index == 2 or self.min_index == 3 or self.min_index == 4:
				if sumls[self.min_index]>x*self.sum_ls[self.min_index] or sumls[self.min_index+2]>x*self.sum_ls[self.min_index+2] or sumls[self.min_index+4]>x*self.sum_ls[self.min_index+4]:
					self.user_f=userfreq
					self.wf = w_f
					self.min_index = min_i
					self.sum_ls = sumls
					self.set_user_freq(self.user_f)
			elif self.min_index == 39 or self.min_index == 38 or self.min_index == 37 or self.min_index == 36 or self.min_index == 35:
				if sumls[self.min_index]>x*self.sum_ls[self.min_index] or sumls[self.min_index-2]>x*self.sum_ls[self.min_index-2] or sumls[self.min_index-4]>x*self.sum_ls[self.min_index-4]:
					self.user_f=userfreq
					self.wf = w_f
					self.min_index = min_i
					self.sum_ls = sumls
					self.set_user_freq(self.user_f)
			else:
				if sumls[self.min_index]>x*self.sum_ls[self.min_index] or sumls[self.min_index-2]>x*self.sum_ls[self.min_index-2] or sumls[self.min_index-4]>x*self.sum_ls[self.min_index-4]or sumls[self.min_index+2]>x*self.sum_ls[self.min_index+2] or sumls[self.min_index+4]>x*self.sum_ls[self.min_index+4]:
					self.user_f=userfreq
					self.wf = w_f
					self.min_index = min_i
					self.sum_ls = sumls
					self.set_user_freq(self.user_f)
			time.sleep(2)
				#timer = threading.Timer(3,self.spect_sense)
				#timer.start()
	#--------------------------------------return s_,reward,done-----------------------
	def step(self,action):
		# ---------------------------------define reward------------------------------
		jammer_freq = self.get_jammer_freq()
		print("jammer_freq---------------------",jammer_freq)
		if self.action_space[action]-jammer_freq >100000.0:
			if self.action_ls[-1] == self.action_space[action]:
				reward = 1
				done = True
			else:
				reward = 0
				done = True
		else:
			reward = -1
			done = False
		self.action_ls.append(self.action_space[action])
		#---------------------------------define s_ ----------------------------------
		self.set_user_freq(self.action_space[action])
		s_ = usrp_spetrum_sense.main()
		return s_,reward,done
	def main_thread(self):
		# ------------------------thread start----------------------------------------
		thread_ls = []
		thread_ls.append(threading.Thread(target=self.main_jammer_tx))
		thread_ls.append(threading.Thread(target=self.main_user_tx))
		for t in thread_ls:
			t.start()
		#self.step(2)
""""""
if __name__ == "__main__":
	env=freq_env()
	env.main_thread()
	env.spect_sense()
	pass