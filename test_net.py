import tensorflow as tf
import numpy as np
import random
import matplotlib.pylab as plt
def waterfall_reset():
	start_freq = 97000000.0
	end_freq = 98000000.0
	freq_line = []
	power_ls = []
	user_freq = [6.94817642133, 11.0710678659, 14.6195817555, 17.3627686085,
				 22.8806895204, 24.2871283707, 24.7753483924, 25.2412643153,
				 25.6009231012, 25.3235259942, 24.7298037759, 23.9569713324,
				 23.1641902552, 19.5581498392, 15.0973001711, 11.5414549903]
	with open('data97_98_97.4.txt') as fr:
		for line in fr.readlines():
			curLine = line.strip().split()
			freq_line.append(curLine[2])
			power_ls.append(curLine[3])
	index_end = [i for i, x in enumerate(freq_line) if x == str(end_freq)]
	rand_int = random.randint(0, 9)
	other_1 = random.randint(0, 9)
	other_2 = random.randint(0, 9)
	waterfall_reset = np.zeros(shape=(1, 100, 160, 1), dtype=float)
	for x in range(0, int(index_end[0])-60):
		for y in range(0, int(index_end[0])):
			waterfall_reset[0, x, y] = float(power_ls[(int(index_end[0]) + 1) * (x + 1) + y])
	for m in range(0, int(index_end[0])-60):
		for n in range(rand_int * 16, (rand_int + 1) * 16):
			waterfall_reset[0, m, n] = waterfall_reset[0, m, n, 0] + user_freq[n - rand_int * 16]
	return waterfall_reset
def net():
    s = tf.placeholder(tf.float32, [None,100, 160, 1], name='s')

    with tf.variable_scope('eval_net'):
        # c_names(collections_names) are the collections to store variables
        c_names, n_l1, w_initializer, b_initializer = \
            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
            tf.random_normal_initializer(0.1), tf.constant_initializer(0.1)  # config of layers 0.02 0.0

        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [8, 8, 1, 32], tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                 collections=c_names)
            b1 = tf.get_variable('b1', [32], initializer=tf.constant_initializer(0.1), collections=c_names)
            l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(s, w1, [1, 4, 4, 1], 'SAME'), b1))
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [4, 4, 32, 64], tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                 collections=c_names)
            b2 = tf.get_variable('b2', [64], initializer=tf.constant_initializer(0.1), collections=c_names)
            l2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l1, w2, [1, 2, 2, 1], 'SAME'), b2))
        with tf.variable_scope('l3'):
            w3 = tf.get_variable('w3', [3, 3, 64, 64], tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                 collections=c_names)
            b3 = tf.get_variable('b3', [64], initializer=tf.constant_initializer(0.1), collections=c_names)
            l3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), b3))
        shape = l3.get_shape().as_list()
        l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
        shape_ = l3_flat.get_shape().as_list()
        with tf.variable_scope('l4'):
            w4 = tf.get_variable('w4', [shape_[1], 512], initializer=w_initializer, collections=c_names)
            b4 = tf.get_variable('b4', [512], initializer=b_initializer, collections=c_names)
            l4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l3_flat, w4), b4))
        shape_l4 = l4.get_shape().as_list()
        with tf.variable_scope('l5'):
            w5 = tf.get_variable('w5', [shape_l4[1], 10], initializer=w_initializer, collections=c_names)
            b5 = tf.get_variable('b5', [10], initializer=b_initializer, collections=c_names)
            q_eval = tf.nn.relu(tf.nn.bias_add(tf.matmul(l4, w5), b5))
    tf.add_to_collection('pred_network', q_eval)
    b=np.array([0,1,2,3,4,5,6,7,8,9],dtype=np.float32)
    with tf.variable_scope('loss'):
        # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        loss = tf.reduce_mean(tf.squared_difference(b, q_eval))
    with tf.variable_scope('train'):
        _train_op = tf.train.RMSPropOptimizer(0.005).minimize(loss)
    a = waterfall_reset()
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())
        cost_ls = []
        for i in range(400):
            train,cost=sess.run([_train_op,loss],feed_dict={s:a})
            if i % 10 == 0:
                saver.save(sess, "my-testmodel", global_step=i)
            cost_ls.append(cost)
            print(cost_ls[-1],"--------------------------",i)
            if(cost_ls[-1]<1e-3):break
    plt.plot(np.arange(len(cost_ls)),cost_ls)
    plt.show()
def load_model():
    with tf.Session() as sess:
        a = waterfall_reset()
        saver = tf.train.import_meta_graph('my-testmodel-150.meta')
        saver.restore(sess, tf.train.latest_checkpoint(""))
        graph = tf.get_default_graph()
        y = tf.get_collection('pred_network')[0]
        input_x = graph.get_operation_by_name('s').outputs[0]
        print(sess.run(y, feed_dict={input_x:a}))
if __name__ == "__main__":
    #net()
    load_model()