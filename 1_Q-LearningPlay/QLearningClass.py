import numpy as np
import pandas as pd
import random
import os

os.chdir('1_Q-LearningPlay/mem_and_log/')

Memory = 'mem.csv'
Log = 'log.csv'

class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,save_mem_round=100,train_mode=1):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.train_mode = train_mode
        self.save_mem_round = save_mem_round

        self.loss_his = pd.DataFrame(columns=['round','loss','steps'], dtype=np.float64)
        self.step_loss = 0
        self.steps = 0

        self.round_counts = 0
        self.q_table = pd.DataFrame(columns=[str(a) for a in self.actions], dtype=np.float64)
        if self.train_mode!=0:
            self.__save_get_memorize(True)


    def __save_get_memorize(self,read_flag):
        exits_flag = os.path.exists(Memory)
        # True: read, False: save
        if read_flag:
            if exits_flag:
                self.q_table = pd.read_csv(Memory,index_col=0,header=0)
                # self.q_table.index = [str(a) for a in  self.q_table.index ]
                print('~~~Reading memory!~~~')
        else:
            self.q_table.to_csv(Memory)
            print('~~~Saving memory!~~~')
            self.loss_his.to_csv(Log,mode='a',header=False,index=False)
            self.loss_his = pd.DataFrame(columns=['round','loss','steps'], dtype=np.float64)


    def choose_action(self, observation):
        new_observation = self.check_state_exist(observation)
        # observation = self.states_pre_process(observation)
        action = ''
        action_index = 0
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[new_observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.argmax(state_action)
        else:
            # choose random action
            action = random.randrange(len(self.actions))
        return action

    def learn(self, s, a, r, s_,done):


        new_s_=  self.check_state_exist(s_)
        new_s = self.check_state_exist(s)
        # print(type(self.q_table.columns[0]))
        a = str(a)
        # print(type(a))
        q_predict = self.q_table.loc[new_s, a]
        if done != True:
            q_target = r + self.gamma * self.q_table.loc[new_s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
            self.round_counts += 1
            self.loss_his = self.loss_his.append(pd.Series(
                    [self.round_counts,round(self.step_loss/self.steps,3),self.steps],
                    index=self.loss_his.columns,
                    name=self.round_counts,
                )
            )
            print('当前为第{0}回合，消耗了{1}步,平均损失为:{2}'.format(self.round_counts,self.steps,round(self.step_loss/self.steps,3)))
            self.step_loss = 0
            self.steps = 0
            if self.round_counts % self.save_mem_round == 0 and self.round_counts != 0:
                self.__save_get_memorize(False)


        loss =abs(q_target - q_predict)
        self.step_loss += loss
        self.steps += 1

        self.q_table.loc[new_s, a] += self.lr * loss  # update



    # 将状态规整成一个个区间来降低状态空间 = 512/5*288/10
    def states_pre_process(self,s):
        x = 20
        y = 10
        # print(s)
        news = (round(int(s[0])/x),round(int(s[1])/y))
        ns = str(news)
        return ns

    def check_state_exist(self, state):
        news = self.states_pre_process(state)

        if news not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=news,
                )
            )
        return news