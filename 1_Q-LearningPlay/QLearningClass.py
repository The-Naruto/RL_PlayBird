import numpy as np
import pandas as pd
import random



class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,save_mem_steps=10000,train_mode=1):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.train_mode = train_mode
        self.save_mem_steps = save_mem_steps

        self.step_counts = 0

        self.train_mode = 1
        if self.train_mode==1:
            self.q_table =pd.read_csv('13000QLearning')
            print('~~~记忆读取成功!~~~')
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        new_observation = self.check_state_exist(observation)
        # observation = self.states_pre_process(observation)
        action =  np.zeros([2])
        action_index = 0
        # action selection
        if np.random.uniform() < self.epsilon:

            # choose best action
            state_action = self.q_table.loc[new_observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action_index = np.argmax(state_action)
            action[action_index] = 1

        else:
            # choose random action
            action_index = random.randrange(len(self.actions))
            action[action_index] = 1
        return action

    def learn(self, s, a, r, s_,done):
        self.step_counts += 1

        new_s_=  self.check_state_exist(s_)
        new_s = self.check_state_exist(s)
        q_predict = self.q_table.loc[new_s, a]
        if done != True:
            q_target = r + self.gamma * self.q_table.loc[new_s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[new_s, a] += self.lr * (q_target - q_predict)  # update

        if self.step_counts%self.save_mem_steps == 0:
            self.q_table.to_csv(str(self.step_counts)+"QLearning")

    # 将状态规整成一个个区间来降低状态空间 = 512/5*288/10
    def states_pre_process(self,s):
        x = 80
        y = 50
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