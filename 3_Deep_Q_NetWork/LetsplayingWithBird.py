# import sys
# import os
# o_path = os.getcwd() # 返回当前工作目录
# sys.path.append(o_path) # 添加自己指定的搜索路径

from My_FlappyBird.FlappyBirdClass import FlappyBird
from DQNPlayer import DQN

TrainMode = 1   # 训练模式类别,  0:完全重新开始，1：带着上次回忆开始，2: 仅执行,不训练



def update():
    for episode in range(1000):
        # initial observation
        observation = env.reset_n()

        # RL choose action based on observation
        action = RL.choose_action(observation)
        while True:


            # RL take action and get next observation and reward
            _ , observation_, reward, done = env.frame_step(action)

            action_ = RL.choose_action(observation)

            # RL learn from this transition
            RL.learn(observation, action,reward, observation_,done)

            # swap observation
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    # env.destroy()

if __name__ == "__main__":
    env = FlappyBird()
    RL = DQN(n_actions=env.actions)

    update()