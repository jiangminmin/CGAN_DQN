import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.00025,
            reward_decay=0.9,
            e_greedy=0.9,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=0.001,
            output_graph=False,
            replace_target_iter=10
    ):
        self.Load_model = False
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = 10#replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max


        # total learning step
        self.learn_step_counter = 0
        self.save_epoch = 1000

        self.flag = 0
        # initialize zero memory [s, a, r, s_]
        #self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory = [([0]*4) for i in range(self.memory_size)]
        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)
        writer = tf.summary.FileWriter("/media/jiangminmin/00030E7C0003A35C/tensorflowgraph", tf.get_default_graph())
        # python /usr/local/lib/python2.7/dist-packages/tensorflow/tensorboard/tensorboard.py --logdir=//media//jiangminmin//00030E7C0003A35C//tensorflowgraph
        writer.close()
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.tmp = tf.placeholder(tf.float32, [20,1601], name='tmp')
        #self.s = tf.placeholder(tf.float32, [None, self.tmp.shape[0], self.tmp.shape[1], 1], name='s')
        self.s = tf.placeholder(tf.float32, [None,50, 160, 1], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0.1), tf.constant_initializer(0.1)  # config of layers 0.02 0.0

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [8,8,1,32], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), collections=c_names)
                b1 = tf.get_variable('b1', [32], initializer=tf.constant_initializer(0.1), collections=c_names)
                l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.s, w1,[1,4,4,1],'SAME'),b1))
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [4,4,32,64], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), collections=c_names)
                b2 = tf.get_variable('b2', [64], initializer=tf.constant_initializer(0.1), collections=c_names)
                l2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l1, w2,[1,2,2,1],'SAME'),b2))
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [3,3,64,64], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), collections=c_names)
                b3 = tf.get_variable('b3', [64], initializer=tf.constant_initializer(0.1), collections=c_names)
                l3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l2, w3,[1,1,1,1],'SAME'),b3))
            shape = l3.get_shape().as_list()
            l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            shape_ = l3_flat.get_shape().as_list()
            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [shape_[1], 512], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [512], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l3_flat,w4),b4))
            shape_l4 = l4.get_shape().as_list()
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [shape_l4[1],10], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [10], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.nn.relu(tf.nn.bias_add(tf.matmul(l4,w5),b5))
        tf.add_to_collection('pred_network', self.q_eval)
        with tf.variable_scope('loss'):
            #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net(q_next) ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 50, 160, 1], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0.1), tf.constant_initializer(0.1)  # config of layers tf.constant_initializer(0.02)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [8,8,1,32], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), collections=c_names)
                b1 = tf.get_variable('b1', [32], initializer=tf.constant_initializer(0.1), collections=c_names)
                l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.s, w1,[1,4,4,1],'SAME'),b1))
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [4,4,32,64], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), collections=c_names)
                b2 = tf.get_variable('b2', [64], initializer=tf.constant_initializer(0.1), collections=c_names)
                l2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l1, w2,[1,2,2,1],'SAME'),b2))
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [3,3,64,64], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), collections=c_names)
                b3 = tf.get_variable('b3', [64], initializer=tf.constant_initializer(0.1), collections=c_names)
                l3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l2, w3,[1,1,1,1],'SAME'),b3))
            shape = l3.get_shape().as_list()
            l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            shape_ = l3_flat.get_shape().as_list()
            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [shape_[1], 512], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [512], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l3_flat,w4),b4))
            shape_l4 = l4.get_shape().as_list()
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [shape_l4[1],10], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [10], initializer=b_initializer, collections=c_names)
                self.q_next = tf.nn.relu(tf.nn.bias_add(tf.matmul(l4,w5),b5))

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        #1x160x160x1
        self.memory[index][0] = s
        self.memory[index][1] = a
        self.memory[index][2] = r
        self.memory[index][3] = s_
        self.memory_counter += 1
        #print(self.memory)

    def choose_action(self, observation):
        if self.Load_model:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        #print('++++++++++++++++++++++++++++++++++++++++++++{}++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'.format(self.learn_step_counter))
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if (self.learn_step_counter+1) % (self.save_epoch+1) == 0:
            self.saver.save(self.sess, "./my-model", global_step=self.learn_step_counter)
            pass
        #if self.flag == self.memory_size - self.batch_size:
            #self.flag = 0
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        #sample_index = np.arange(self.batch_size, dtype=np.int32)
        self.memory_lib_s = np.zeros(shape=(self.batch_size,50,160,1),dtype=float)
        self.memory_lib_s_ = np.zeros(shape=(self.batch_size, 50, 160, 1), dtype=float)
        eval_act_index = np.zeros(shape=(self.batch_size), dtype=int)
        reward = np.zeros(shape=(self.batch_size), dtype=int)
        for i in range(self.batch_size):
            self.memory_lib_s[i] = self.memory[sample_index[i]][0]
            eval_act_index[i] = self.memory[sample_index[i]][1]
            reward[i] = self.memory[sample_index[i]][2]
            self.memory_lib_s_[i] = self.memory[sample_index[i]][3]


        #batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: self.memory_lib_s_,  # fixed params
                self.s: self.memory_lib_s,  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        #print(q_target.shape)
        #print(q_target,"-------------------q_eval")
        #print(q_next,"----------------------q_next")
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        #print(q_target,"-------------------------q_target")
        #print(self.memory_lib_s,eval_act_index,reward,self.memory_lib_s_)
        #print("------------------------------------------------------------------",q_eval==q_target)
        # train eval network
        #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        #print("---------------------------------------------------------",self.memory_lib_s)
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: self.memory_lib_s,
                                                self.q_target: q_target})

        #print(q_target-q_eval)
        self.cost_his.append(self.cost)
        #print(self.cost_his)
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1
        #print(self.learn_step_counter,self.cost_his[-1])
        #self.flag += 1
        #print("-----------------------------hahahhahhahhah-----------------------------------------")
        #return self.cost_his[-1]
    def plot_cost(self):
        import matplotlib.pyplot as plt
        #del self.cost_his[1]
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
"""
if __name__ == "__main__":
    env = freq_env()
    dqn = DeepQNetwork(10, 160,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    #flag = [None,observation.shape[0],observation.shape[1],1]
    actions_value = dqn.sess.run(dqn.q_eval,feed_dict={dqn.s:observation})
    print(actions_value)
"""


