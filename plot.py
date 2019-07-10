import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

freq_line=[]
power=[]
with open('data97_98_97.4.txt') as fr:
	for line in fr.readlines():
		curLine = line.strip().split()
		freq_line.append(curLine[2])
		power.append(float(curLine[3]))
index_start = [i for i,x in enumerate(freq_line) if x == str(97000000.0)]
index_end = [i for i,x in enumerate(freq_line) if x == str(98000000.0)]
print(index_start[0],index_end[0])
print(len(power))
for i in range(200):
	a = power[(int(index_end[0])+1)*i : (int(index_end[0])+1)*(i+1)]
	b=[]
	for x in range(int(len(a)/16)):
		b.append(sum(a[16*x:16*(x+1)]))
	print(np.argmin(b))
"""
#np.random.seed(1)
#tf.set_random_seed(1)
s = tf.placeholder(tf.float32, [None, 2], name='s')

with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [2, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, 4], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 4], initializer=b_initializer, collections=c_names)
                q_eval = tf.matmul(l1, w2) + b2
e_params = tf.get_collection('eval_net_params')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
actions_value = sess.run(q_eval, feed_dict={s:[[0.5,0.5]]})
action = np.argmax(actions_value)
print("hahahaha")
print(sess.run(w1))
print(actions_value)
print(action)
"""
"""
	def dataset(self,fileName,start_freq,end_freq):
		freq_line = []
		with open(fileName) as fr:
		    for line in fr.readlines():
		        curLine = line.strip().split()
		        freq_line.append(curLine[2])
		index_100 = [i for i, x in enumerate(freq_line) if x == str(start_freq)]
		index_110 = [i for i, x in enumerate(freq_line) if x == str(end_freq)]
		self.waterfall = np.zeros(shape=(len(index_100),int(index_100[1]-index_100[0])))
		for i in range(0,len(index_100)):
		    self.waterfall[i] = freq_line[index_100[i]:index_110[i]+1]
		return self.waterfall
"""