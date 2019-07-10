"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import sample_env_offline

np.random.seed(1)
tf.set_random_seed(1)


class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.00025,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not
        self.Load_model = False

        self.learn_step_counter = 0
        self.save_epoch = 1000

        self.memory = [([0]*4) for i in range(self.memory_size)]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            """
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            """
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [8,8,1,32], tf.float32, initializer=tf.contrib.layers.xavier_initializer(), collections=c_names)
                b1 = tf.get_variable('b1', [32], initializer=tf.constant_initializer(0.1), collections=c_names)
                l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(s, w1,[1,4,4,1],'SAME'),b1))
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

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w5 = tf.get_variable('w5', [shape_l4[1], 1], initializer=w_initializer, collections=c_names)
                    b5 = tf.get_variable('b5', [1], initializer=b_initializer, collections=c_names)
                    self.V = tf.nn.bias_add(tf.matmul(l4, w5), b5)
                    """
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l3, w2) + b2
                    """
                with tf.variable_scope('Advantage'):
                    w5 = tf.get_variable('w5', [shape_l4[1], self.n_actions], initializer=w_initializer, collections=c_names)
                    b5 = tf.get_variable('b5', [self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.nn.bias_add(tf.matmul(l4, w5), b5)
                    """
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l3, w2) + b2
                    """
                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('l5'):
                    w5 = tf.get_variable('w5', [shape_l4[1], 10], initializer=w_initializer, collections=c_names)
                    b5 = tf.get_variable('b5', [10], initializer=b_initializer, collections=c_names)
                    out = tf.nn.relu(tf.nn.bias_add(tf.matmul(l4, w5), b5))
                """
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2
                """
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, 50,160,1], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 50,160,1], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size

        self.memory[index][0] = s
        self.memory[index][1] = a
        self.memory[index][2] = r
        self.memory[index][3] = s_

        self.memory_counter += 1

    def choose_action(self, observation):
        #observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if (self.learn_step_counter+1) % (self.save_epoch+1) == 0:
            self.saver.save(self.sess, "./my-model", global_step=self.learn_step_counter)
            print("successful save model")
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        #batch_memory = self.memory[sample_index, :]

        self.memory_lib_s = np.zeros(shape=(self.batch_size, 50, 160, 1), dtype=float)
        self.memory_lib_s_ = np.zeros(shape=(self.batch_size, 50, 160, 1), dtype=float)
        eval_act_index = np.zeros(shape=(self.batch_size), dtype=int)
        reward = np.zeros(shape=(self.batch_size), dtype=int)
        for i in range(self.batch_size):
            self.memory_lib_s[i] = self.memory[sample_index[i]][0]
            eval_act_index[i] = self.memory[sample_index[i]][1]
            reward[i] = self.memory[sample_index[i]][2]
            self.memory_lib_s_[i] = self.memory[sample_index[i]][3]

        q_next = self.sess.run(self.q_next, feed_dict={self.s_: self.memory_lib_s_}) # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: self.memory_lib_s})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #eval_act_index = batch_memory[:, self.n_features].astype(int)
        #reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        #q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: self.memory_lib_s,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
