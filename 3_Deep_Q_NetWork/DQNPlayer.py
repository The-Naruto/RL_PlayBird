"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
from collections import deque



np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.001,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.001,
            output_graph=False,
    ):
        self.n_actions = len(n_actions)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  # 间隔多少步更新
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment  # 不断缩小随机范围
        self.epsilon = 0.1 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]     2 = 动作+奖励
        self.memory = deque()


        # consist of [target_net, evaluate_net]
        self._build_net()
        # t_params = tf.get_collection('target_net_params')
        # e_params = tf.get_collection('eval_net_params')
        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder("float", [None, 80, 80, 4]) # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.a = tf.placeholder("float", [None, self.n_actions])
        # network weights
        W_conv1 = weight_variable([8, 8, 4, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        W_fc1 = weight_variable([1600, 512])
        b_fc1 = bias_variable([512])

        W_fc2 = weight_variable([512, self.n_actions])
        b_fc2 = bias_variable([self.n_actions])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(self.s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)

        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # out put layer
        self.q_eval = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self.readout_action = tf.reduce_sum(tf.multiply(self.q_eval, self.a), reduction_indices=1)
        self.train_step =  tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.sess.run(tf.initialize_all_variables())


    def store_transition(self, s, a, r, s_, done):
        new_s = rebuild_s(s)
        new_s_ = rebuild_s(s_)

        # store the transition in D
        self.memory.append((new_s, a, r, new_s_, done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def choose_action(self, observation):
        new_s = rebuild_s(observation)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [new_s]})[0]
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self,s , a , r, s_ , done):
        self.store_transition(s , a , r, s_ , done)

        # # check to replace target parameters
        # if self.learn_step_counter % self.replace_target_iter == 0:
        #     self.sess.run(self.replace_target_op)
        #     print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if len(self.memory) > self.memory_size:
            batch_memory = random.sample(self.memory, self.memory_size)
        else:
            batch_memory = random.sample(self.memory, len(self.memory))

        s_batch = [d[0] for d in batch_memory]
        a_batch = [d[1] for d in batch_memory]
        r_batch = [d[2] for d in batch_memory]
        s__batch = [d[3] for d in batch_memory]



        sr = self.q_eval.eval(feed_dict={self.s:s__batch})

        y_batch = []
        for i in range(0,len(batch_memory)):
            if batch_memory[i][4]:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i]+self.gamma * np.max(sr[i]))

        self.q_eval.run(feed_dict = {self.s: s_batch,self.q_target:y_batch,self.a:a_batch })

        self.cost_his.append(self.cost)

        # increasing epsilon
        if self.epsilon<self.epsilon_max:
            self.epsilon = self.epsilon + (self.epsilon_max-self.epsilon)/5000
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def rebuild_s(s):
    s1 = cv2.cvtColor(cv2.resize(s, (80, 80)), cv2.COLOR_BGR2GRAY)
    s2 = cv2.threshold(s1, 1, 255, cv2.THRESH_BINARY)
    s3 = np.stack((s2, s2, s2, s2), axis=2)
    return s3