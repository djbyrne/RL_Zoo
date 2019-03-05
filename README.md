# RL_Zoo
Library with clean, simple and modular implementations of the main RL algorithms

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
   

