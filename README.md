[![Build Status](https://travis-ci.com/djbyrne/RL_Zoo.svg?branch=master)](https://travis-ci.com/djbyrne/RL_Zoo)

# RL Zoo (In Progress)
The goal of this repo is to provide simple and clean examples of the major Reinforcement Learning algorithms within a range 
of environments. 

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name rl_dev python=3.6
	source activate rl_dev
	```
	- __Windows__: 
	```bash
	conda create --name rl_dev python=3.6 
	activate rl_dev

2. Install gym

    Instructions are followed from [here](https://medium.com/@SeoJaeDuk/archive-post-how-to-install-open-ai-gym-on-windows-1f5208c16179)
   ```bash
   conda install git
   conda install -c conda-forge ffmpeg
   # Note: For video recording.
   conda update --all
   conda clean -a
   pip install git+https://github.com/Kojoley/atari-py.git
   pip install gym[all]
   ```
   
3. PyTorch
   
   - __Linux__: 
	```bash
	conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
	```
	- __Windows__: 
	```bash
	conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
	```
	
	- __Mac__: 
	```bash
	conda install pytorch torchvision -c pytorch
    # MacOS Binaries dont support CUDA, install from source if CUDA is needed
	```
	
	- Tensorboard for PyTorch
	```bash
	pip install tensorboard-pytorch
	```
4. Extra Dependencies

    ```bash
	  pip install opencv-python
	```
	
## To Do:

### Algorithms

- [X] DQN
- [X] Double DQN
- [X] Dueling DQN
- [X] DQN PER
- [X] Distributional DQN (C51)
- [X] Noisy DQN
- [X] Rainbow
- [X] REINFORCE
- [X] Vanilla Policy Gradient
- [X] A2C
- [X] A3C
- [ ] PPO
- [X] DDPG
- [ ] D4PG
- [ ] GAIL
- [ ] I2A

### Library

**Environment Wrappers**

- Allow for multiple envrionement types:
	- [X] basic gym
	- [X] box2d
	- [X] atari
	- [X] unity 
	
- Ideally the library will automatically detect what environment you are using and assign the correct type of wrapper/network to use it

**Refactor parameters file**

- Include the entire architecture of the network inside the param file denoting the number of layers and nodes in a network

**Save Experiment Weights**

- Save the model weighs and params for a particular experiment

**Evaluation Script**

- [ ] After training evaluate the performance of an agent over the course of 100 episodes or against a proven baseline such as OpenAI  Baselines

- [ ] Add plots on baseline environment for all agents

- [X] Save model weights

 ## References

 - Max Lapan: https://github.com/Shmuma/ptan

