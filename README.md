# Deep Reinforcement Learning
## Project: Atari Pong Agent Using OpenAI Gym

## Project Overview
In this project, you’ll implement a Neural Network for Deep Reinforcement Learning and see it learn more and more as it finally becomes good enough to beat the computer in Atari 2600 game Pong! You can play around with other such Atari games at the [OpenAI Gym](https://gym.openai.com/).

[![Andrej Karpathy Pong AI using Policy Gradients](https://img.youtube.com/vi/YOW8m2YGtRg/0.jpg)](https://www.youtube.com/watch?v=YOW8m2YGtRg)

By executing this project, you’ll be able to do the following:

+ Write a Neural Network from scratch.
+ Implement a Policy Gradient with Deep Reinforcement Learning.
+ Build an AI for Pong that can beat the computer in less than 250 lines of Python.
+ Use OpenAI Gym.

## Sources
Basically, the code and the idea are all based on Dr. Andrej Karpathy’s [blog](http://karpathy.github.io/2016/05/31/rl/) post on Deep Reinforcement Learning. The CNN code is written in [keras](https://github.com/fchollet/keras). The code in ```atari_pong_agent.py``` is intended to be a simpler version of ```pong.py```, which was written by Dr. Karpathy.

## Prerequisites and Background Reading
You’ll need to know the following:

- Basic Python
- Neural Network design and backpropogation
- Calculus and Linear Algebra

Read the [blog post](http://karpathy.github.io/2016/05/31/rl/) on which all of this project is based, if you want a deeper dive into the project.

## Software and Libraries
This project uses the following software and Python libraries:
- [Python](https://www.python.org/download/releases)
- [NumPy](http://www.numpy.org/)
- [Keras](https://github.com/fchollet/keras)
- [Gym](https://gym.openai.com/)

## Project Setup

1. Follow the instructions for installing [OpenAI Gym](https://gym.openai.com/docs/). This requires installing several more involved dependencies, including ```cmake``` and a recent ```pip``` version.

2. Run ```pip install -e .[atari]```

3. Clone the repository and navigate to the downloaded folder.
  ```
  git clone https://github.com/adityasaxena26/Atari-Pong-Agent-using-OpenAI-Gym.git
  cd Atari-Pong-Agent-using-OpenAI-Gym
  ```
4. Run ```python atari_pong_agent.py```
