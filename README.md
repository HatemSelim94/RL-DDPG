### Environment
A double jointed arm and a subspace represnts the goal. The objective is to maintain the end effector in contact with the goal as long as possible. A reward of 0.1 is given each time step both are in contact. The environment is offered in two versions namley one agent and 20 agents. The observation vector size is 33 per agent and the action space dimension is 4 per agent.
The problem is considered to be solved when the mean of rewards over 100 consecutive episodes is larger than 30.
 
#### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

##### Train the agent
* Run Continuous_Control.ipynb or run Continuous_Control.py (Do not forget to activate conda environment first)

* Initializing the agent:
    ```python
    from ddpg import Agent
    agent = Agent(state_size, action_size,hd1_units=400, hd2_units=300 ,random_seed = 0, buffer_size = int(2e5), batch_size = 256, tau = 0.0005, actorLr =1e-3, criticLr = 1e-3, weight_decay = 0, update_every = 20, gamma = 0.99)
    ```
* Train the DDPG agent:
    ```python
    from ddpg_train import ddph_train
    ddpg_train(agent, env, brain_name, n_episodes=2000, max_t=300000, print_every=100)
     ```
###### Future work
* Try the following:
    * DDPG with 20 Agents
    * A2C   
    * A3C
