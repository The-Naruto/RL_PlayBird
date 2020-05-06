## 写在前面的话
- 游戏部分大量参考了github项目: [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
- 算法部分大量参考了github项目: [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)  
————非常感谢以上两人无私的奉献！

## 项目介绍：  
 本项目尝试训练各种主流**Reinforcement Learning算法**来玩**FlappBird**这个小游戏，计划包括如下算法:
 - Q-Learning
 - Sarsa
 - Sarsa-lambda
 - DQN: Deep-Q-Network
 - Policy-Gradient-softmax
 - A2C：Advantage-Actor-Critic
 - DDPG:Deep-Deterministic-Policy-Gradient
 - A3C：Asynchronous-Advantage-Actor-Critic   
 每个算法下都会存储我训练的''记忆' （可能是Q表也可能是 NN的参数）。
 
 
## 目录结构说明：
- My_FlappyBird是游戏的主目录，其中的test为我自己对pygame的测试文件。  
- 每一个算法单独使用一个单独的文件夹，如：1_Q-LearningPlay, 表示使用Q-Learning算法进行游戏,
- 其中的XXClass.py为实现算法的类,另一个LetsplayingWithBird.py表示运行和开始执行本算法来玩这个游戏。


---------------------------------
## Written in front
- The game section heavily references the github project: [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
- The algorithm section heavily references the github project: [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)  
-----Thank you very much for the selfless dedication of the above two!

## Project Introduction：
This project attempts to train some **Reinforcement Learning algorithms** to 
play **FlappBird** this small game, 
the plan includes the following algorithms:
 - Q-Learning
 - Sarsa
 - Sarsa-lambda
 - DQN: Deep-Q-Network
 - Policy-Gradient-softmax
 - A2C：Advantage-Actor-Critic
 - DDPG:Deep-Deterministic-Policy-Gradient
 - A3C：Asynchronous-Advantage-Actor-Critic   
Under each algorithm, the "memory" of my training will be stored (may be Q table or NN parameters).


## Directory structure description:
- My_FlappyBird is the main directory of the game, where the test is my own test file for pygame.  
- Each algorithm uses a separate folder, such as: 1_Q-LearningPlay, 
which means that the Q-Learning algorithm is used to play the game.  
- Among them, XXClass.py is the class that implements the algorithm, 
and the other LetsplayingWithBird.py means running and starting to execute this algorithm to play this game.














