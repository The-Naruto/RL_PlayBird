# import sys
# import os
# o_path = os.getcwd() # 返回当前工作目录
# sys.path.append(o_path) # 添加自己指定的搜索路径

from My_FlappyBird.FlappyBirdClass import FlappyBird
from QLearningClass import QLearning

TrainMode = 1   # 训练模式类别,  0:完全重新开始，1：带着上次回忆开始，2: 仅执行,不训练



def update():
    for episode in range(5000):
        # initial observation
        observation = env.reset()

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, _ , reward, done = env.frame_step(action)

            # RL learn from this transition
            RL.learn(observation, action, reward, observation_,done)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    # env.destroy()

if __name__ == "__main__":
    env = FlappyBird()
    RL = QLearning(actions=env.actions,train_mode=TrainMode)

    update()