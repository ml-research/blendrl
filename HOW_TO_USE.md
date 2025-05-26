# Implementing a New Game in Blendrl

This guide provides a structured approach to implementing a new game in blendrl using OCAtari and HackAtari. It covers environment setup, file structure, logic implementation, training, and evaluation.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Development Workflow](#development-workflow)
3. [Game Implementation](#game-implementation)
4. [Training and Evaluation](#training-and-evaluation)
5. [External References](#external-references)

---

## Environment Setup

Before implementing a game, set up the development environment. This guide compiles all setup instructions into a single document, but may differ from the actual setup in the future.

### Create a Conda environment (recommended) [[Conda](https://anaconda.org/anaconda/conda)]:

```
conda create -n blendrl python=3.11.10
conda activate blendrl
```


### Install blendrl, NSFR, NUDGE, Neumann [[blendrl](https://github.com/ml-research/blendrl/blob/main/INSTALLATION.md)]

```
git clone https://github.com/ml-research/blendrl.git
cd blendrl
pip install -r requirements.txt
```

```
cd nsfr
pip install -e .
cd ../nudge
pip install -e .
cd ..
cd neumann
pip install -e .
cd ..
```

### Install OC_Atari [[OC_Atari](https://github.com/k4ntz/OC_Atari/blob/master/README.md)]

>Installation from pip package.

```
pip install "gymnasium[atari]"
AutoROM --accept-license
pip install ocatari
```

>If that doesn't work:

```
pip install "gymnasium[atari, accept-rom-license]"
pip install gymnasium==0.29.1
pip install ale_py
pip install ocatari
```

>Installation from source.

```
pip install "gymnasium[atari, accept-rom-license]"
git clone https://github.com/k4ntz/OC_Atari/blob/master/README.md
cd OC_Atari
pip install -e .
```

### Install HackAtari [[HackAtari](https://github.com/k4ntz/HackAtari/blob/master/README.md)]

>Installation from pip package.

```
pip install hackatari
```

>Installation from source.

```
git clone https://github.com/k4ntz/HackAtari
cd HackAtari
pip install -e .
```

### Install PyG and its dependencies [[PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)]

>Choosing CUDA is recommended, as this will speed up training later. When choosing CUDA make sure to choose the correct CUDA version installed on your system.

```
conda install pyg -c pyg

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install tensorboard
```
>If installing without CUDA instead follow the following instructions.
```
conda install pyg -c pyg

pip install torch-geometric
pip install torch-sparse
pip install torch-scatter

pip install tensorboard

```

### Verify installation

You should now be able to run the training script.
> env_name: the name of the environment  
num_envs: the number of parallel game environments  
num_steps: the number of steps to run in each environment per policy rollout  
gamma: the discount factor gamma  
total_timesteps: total timesteps of the experiments  
save_steps: the number of steps to save models  
more parameters can be found in blendrl/train_blendrl.py

```
python train_blenderl.py --env-name <name> --joint-training --num-steps <int> --num-envs <int> --gamma 0.99　
```

## Development Workflow

### File structure

>The following diagram shows the file structure of the blendrl project.

```
/blendrl
├── in
│   └── envs
│       └── <game_name>                                  # Game-specific folder (e.g., riverraid, seaquest)
│           ├── logic
│           │   └── default                              # Default logic configuration for the game
│           │       ├── blender_clauses.txt              # Blending policy: logic vs. neural decision
│           │       ├── clauses.txt                      # Logic rules: FOL-based policy actions
│           │       ├── consts.txt                       # Object definitions
│           │       ├── neural_preds.txt                 # Neural predicates
│           │       └── preds.txt                        # Possible agent actions
│           ├── blenderl_reward.py                       # Custom reward function logic
│           ├── env.py                                   # Game environment class (inherits from NudgeEnv)
│           ├── env_vectorized.py                        # Handles parallelized environments for training
│           ├── mlp.py                                   # Neural MLP architecture (PyTorch-based)
│           └── valuation.py                             # Implements neural predicates from neural_preds.txt to evaluate game states
├── out
│   └── runs
│       └── <run_name>                                   # Game-specific folder (e.g., riverraid, seaquest)
│           ├── checkpoints
│           │   ├── step_#num                            # Training checkpoints
│           │   ├── data.pkl                             
│           │   └── training_log.pkl
│           └── config.yaml.py                           # blendrl training configuration
├── train_blenderl.py                                    # Training entry point script
├── play_gui.py                                          # Playing entry point script
├── README.md                                            # Project guide
└── requirements.txt                                     # Dependencies for the whole project
```

### Workflow

>Example workflow to integrate a new game with the blendrl project.

```
┌──────────────────┐      ┌────────────────┐      ┌──────────┐      ┌─────────────────┐      
│blendrl/ OC_Atari/│      │Inspiration from│      │Game rules│      │ocatari/ram/__.py│      
│HackAtari paper   │      │existing envs   │      │          │      │                 │      
└──────────────────┘      └───────┬────────┘      └────────┬─┘      └────────┬────────┘      
                                  │                        │                 │               
       ┌──────────────────────────┼───────────────────┐    ├─────────────────┤               
       │                          │                   │    │                 │               
       │                  ┌───────▼─────┐         ┌───▼────▼───┐             │               
       │                  │Create env.py│         │Create logic◄────────────x│x─────────────┐
       │                  │             │         │files       │             │              │
       │                  └───────┬─────┘         └──────┬─────┘             │              │
       │                          │                      │                   │              │
       │                          │                      │                   │              │
       │                          │                      │                   │              │
┌──────▼──────┐           ┌───────▼─────────┐     ┌──────▼─────┐    ┌────────▼─────────┐    │
│Create mlp.py│           │Create           │     │Create      │    │Create            ◄────┤
│             │           │env_vectorized.py│     │valuation.py│    │blenderl_reward.py│    │
└──────┬──────┘           └───────┬─────────┘     └──────┬─────┘    └────────┬─────────┘    │
       │                          │                      │                   │              │
       └──────────────────────────┴──────────────────────┼───────────────────┘              │
                                                         │                                  │
                                                  ┌──────▼──────┐                           │
                                                  │Train blendrl│                           │
                                                  │agent        │                           │
                                                  └──────┬──────┘                           │
                                                         │                                  │
                                                         │                                  │
                                                         │                                  │
                          ┌────────────┐          ┌──────▼─────────┐                        │
                          │Final result◄──────────┤Evaluate results├────────────────────────┘
                          │            │          │                │                         
                          └────────────┘          └────────────────┘                                
```

### Logic dependencies

>Relation and dependencies between logic files.

```
         ┌───────────────────┐      ┌───────────┐
         │blender_clauses.txt│      │clauses.txt│
         │                   │      │           │
         └────────┬──────────┘      └──┬──┬─────┘
                  │                    │  │      
                  ├────────────────────┘  │      
                  │                       │      
         ┌────────▼───────┐         ┌─────▼───┐  
         │neural_preds.txt│         │preds.txt│  
         │                │         │         │  
         └────────┬───────┘         └─────────┘  
                  │                              
      ┌───────────┴──────────┐                   
      │                      │                   
┌─────▼────┐           ┌─────▼──────┐            
│consts.txt│           │valuation.py│            
│          │           │            │            
└──────────┘           └────────────┘                    
```

## Game Implementation

### env.py

>The game environment, which inherits from NudgeEnv.

```python
import ...

def make_env(env):
	return env

class NudgeEnv(NudgeBaseEnv):
	name = ...

	def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None):
		"""
        	Constructor for the VectorizedNudgeEnv class.

        	Args:
			mode (str): Mode of the environment. Possible values are "train" and "eval".
			n_envs (int): Number of environments.
			render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
			render_oc_overlay (bool): Whether to render the overlay of OC.
			seed (int): Seed for the environment.
        	"""
	
	def reset(self):
		"""
        	Reset the environment.

        	Returns:
			logic_states (torch.Tensor): Logic states.
			neural_states (torch.Tensor): Neural states.
		"""
	
		return logic_state.unsqueeze(0), neural_state
	
	def step(self, action, is_mapped: bool = False):
		"""
    		Perform a step in the environment.

        	Args:
        		actions (torch.Tensor): Actions to be performed in the environment.
        		is_mapped (bool): Whether the actions are already mapped.
        	Returns:
			Tuple: Tuple containing:
			- torch.Tensor: Observations.
			- list: Rewards.
			- list: Truncations.
			- list: Dones.
			- list: Infos.
        	"""
	
		return (logic_state, neural_state), reward, done, truncations, infos
	
	def extract_logic_state(self, input_state):
		"""
		Extracts the logic state from the input state.
		
        	Args:
    			input_state (list): List of objects in the environment.
		Returns:
        		torch.Tensor: Logic state.
        	"""
	
		return state
		
	def extract_neural_state(self, raw_input_state):
		"""
        	Extracts the neural state from the raw input state.
        	
    		Args:
			raw_input_state (torch.Tensor): Raw input state.
        	Returns:
        		torch.Tensor: Neural state.
    		"""
	
		return torch.tensor(raw_input_state, dtype=torch.float32).unsqueeze(0)
	
	def close(self):
		"""
        	Close the environment.
        	"""
```

### env_vectorized.py

>Vectorized game environment for parallel training.

```python
import ...

def make_env(env):
	return env

class NudgeEnv(NudgeBaseEnv):
	name = ...

	def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None):
		"""
        	Constructor for the VectorizedNudgeEnv class for all environments.

        	Args:
			mode (str): Mode of the environment. Possible values are "train" and "eval".
			n_envs (int): Number of environments.
			render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
			render_oc_overlay (bool): Whether to render the overlay of OC.
			seed (int): Seed for the environment.
        	"""
	
	def reset(self):
		"""
        	Reset all environments.

        	Returns:
			logic_states (torch.Tensor): Logic states.
			neural_states (torch.Tensor): Neural states.
		"""
	
		return torch.stack(logic_states), torch.stack(neural_states)
	
	def step(self, action, is_mapped: bool = False):
		"""
    		Perform a step in all environments.

        	Args:
        		actions (torch.Tensor): Actions to be performed in the environment.
        		is_mapped (bool): Whether the actions are already mapped.
        	Returns:
			Tuple: Tuple containing:
			- torch.Tensor: Observations.
			- list: Rewards.
			- list: Truncations.
			- list: Dones.
			- list: Infos.
        	"""
	
		return (
		        (torch.stack(logic_states), torch.stack(neural_states)),
		        rewards,
		        truncations,
		        dones,
		        infos,
	    	)
	
	def extract_logic_state(self, input_state):
		"""
		Extracts the logic state from the input state.
		
        	Args:
    			input_state (list): List of objects in the environment.
		Returns:
        		torch.Tensor: Logic state.
        	"""
        
		return state
		
	def extract_neural_state(self, raw_input_state):
		"""
        	Extracts the neural state from the raw input state.
        	
    		Args:
			raw_input_state (torch.Tensor): Raw input state.
        	Returns:
        		torch.Tensor: Neural state.
    		"""
	
		return raw_input_state.to(dtype=torch.float32)
	
	def close(self):
		"""
        	Close all environments.
        	"""
```

### blendrl_reward.py

>Reward function for training. Assign reward score based on game states, e.g. position, distance, specific objects.

```python
def reward_function(self) -> float:
	return reward
```

### mlp.py

>Define Multi-Layer Perceptron Neural Network 

```python
class MLP(torch.nn.Module):
	"""
	Multi-Layer Perceptron (MLP) for processing object-centric game states in blendrl.
	
	This neural network expects a fixed-size input representing multiple game objects 
	(entities), each described by a set of features. The MLP is used to produce either 
	action logits (raw, unnormalized output), probabilities, or other value estimates depending on its configuration.
	
	Args:
		in_channels (int): Number of channels of the input.
		hidden_channels (List[int]): List of hidden layer dimensions.
		norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer applied after the linear layer. If None, no normalization is used. Default is None.
		activation_layer (Callable[..., torch.nn.Module], optional): Activation function applied after normalization (or linear layer if no normalization). Default is torch.nn.ReLU.
		inplace (bool, optional): If True, performs operations in-place for applicable layers. Default is None, which uses the default of the activation or dropout layer.
		bias (bool): Whether to include a bias term in linear layers. Default is True.
		dropout (float): Dropout probability applied after activation. Default is 0.0.

	
	Attributes:
		encoding_base_features (int): Number of features per object.
		encoding_max_entities (int): Maximum number of entities considered.
		num_in_features (int): Total number of input features (e.g., 4 features per 24 entities = 96).
		mlp (torch.nn.Sequential): The sequential stack of linear layers and activations.
	"""
	
	def __init__(self, device, has_softmax=False, has_sigmoid=False, out_size=6, as_dict=False, logic=False):
		"""
		Initializes the MLP model.
		
		Constructs a multi-layer perceptron with optional softmax or sigmoid output. The input is a flattened
		representation of object-centric state features.
		
		Args:
		    device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
		    has_softmax (bool, optional): If True, applies softmax activation to the output layer. Default is False.
		    has_sigmoid (bool, optional): If True, applies sigmoid activation to the output layer. Default is False.
		    out_size (int, optional): Number of output neurons. Default is 6.
		    as_dict (bool, optional): Placeholder for enabling dictionary-based outputs. Default is False.
		    logic (bool, optional): Placeholder to indicate logic-specific model use. Default is False.
		"""
        
	def forward(self, state):
		"""
		Performs a forward pass of the MLP.
		
		Args:
		    state (torch.Tensor): Input tensor of shape (batch_size, num_objects, num_features).
		
		Returns:
		    torch.Tensor: Output tensor of shape (batch_size, out_size) representing logits (raw, unnormalized output) or scores.
		"""
	
	return y 
```

### consts.txt

> Game object constant definition, as found in ocatari/ram/<game_name.py>.

```txt
image:img
<object_name>:obj1
<object_name>:obj2,obj3,obj4
...
```

### preds.txt

> Actions that the agent can take.

```txt
<action_name>:1:image
...
```

### neural_preds.txt

> Game states that will be evaluated by valuation.py. Ensure that object counts match the arg count of the respective method. _list denotes a comma separated list (no spaces) of n repetitions.

```txt
<state_name>:arg_count(n):<object_name>_list
...
```

### clauses.txt

> Clauses in FOL (first order logic) that map actions to game states. The first part of the rule specifies the action we want the agent to do if the second part of the rule is evaluated as likely true. These rules return a distribution of weights, which for each action represent the likelihood that the conditions for this action are true. \<p> represents a letter or name to identify an object, e.g. P for player. _list denotes a comma separated list (no spaces) of n repetitions.

```txt
<action_name>(<p>):-<state_name>(<p>_list)_list.
...
```

### blender_clauses.txt

> Clauses that specify when we want the blending module to switch between the reactive neural or strategic logic policy. The rules here consist first of the desired policy and then the condition for that policy. These again return a weight for both the neural and logic part, which are multiplied onto the weights of the weights returned by the neural and logic modules to get a final weight, of which the highest is chosen as the next action to take. _list denotes a comma separated list (no spaces) of n repetitions.

```txt
neural_agent(X):-<state_name>(<p>_list)_list.
logic_agent(X):-<state_name>(<p>_list)_list.
...
```

### valuation.py

>Implement each neural predicate as a valuation function.

```python
import ...

def neural_pred(<object_name>: th.Tensor, ...) -> th.Tensor:
	"""
	Evaluates a spatial or relational condition between game entities.
	
	Each entity is represented as a tensor containing features (e.g., presence probability, x-position, y-position).
	The function returns a probability score indicating how likely a given predicate (e.g., proximity, alignment, position)
	holds between the input objects for each sample in the batch.
	
	Args:
	    <object_name> (torch.Tensor): Tensor representation of the first object (e.g., player), of shape (batch_size, features).
	    ...: Additional torch.Tensor arguments can be included depending on the predicate's arity.
	
	Returns:
	    torch.Tensor: A 1D tensor of shape (batch_size,) with values in [0, 1], representing the probability
	    that the predicate holds for each batch instance.
	"""

	return nsfr.utils.common.bools_to_probs(bool)
	
```


## Training and Evaluation

### Training script

> Command-Line Argument Schema for `train_blenderl.py`

```bash
python train_blenderl.py [--arg1 value1] [--arg2 value2] ...
```

> General experiment settings

| Argument               | Type    | Default                      | Description |
|------------------------|---------|------------------------------|-------------|
| `--exp_name`           | str     | `<script filename>`          | Name of the experiment |
| `--seed`               | int     | `0`                          | Random seed for reproducibility |
| `--torch_deterministic`| bool    | `True`                       | If set, disables CuDNN nondeterminism |
| `--cuda`               | bool    | `True`                       | Enable CUDA if available |
| `--track`              | bool    | `False`                      | Use Weights & Biases tracking |
| `--wandb_project_name` | str     | `"blendeRL"`                 | W&B project name |
| `--wandb_entity`       | str     | `None`                       | W&B team/entity |
| `--capture_video`      | bool    | `False`                      | Record videos during training |

> Environment & Training Configuration

| Argument               | Type    | Default         | Description |
|------------------------|---------|------------------|-------------|
| `--env_id`             | str     | `"Seaquest-v4"`  | Gym environment ID |
| `--env_name`           | str     | `"seaquest"`     | Game shorthand name |
| `--total_timesteps`    | int     | `60000000`       | Total number of training steps |
| `--num_envs`           | int     | `20`             | Number of parallel environments |
| `--num_steps`          | int     | `128`            | Steps per environment per rollout |
| `--anneal_lr`          | bool    | `True`           | Enable learning rate annealing |
| `--gamma`              | float   | `0.99`           | Discount factor |
| `--gae_lambda`         | float   | `0.95`           | GAE lambda parameter |
| `--num_minibatches`    | int     | `4`              | Number of PPO minibatches |
| `--update_epochs`      | int     | `10`             | Number of PPO epochs per update |
| `--norm_adv`           | bool    | `True`           | Normalize advantages |
| `--clip_coef`          | float   | `0.1`            | PPO clip coefficient |
| `--clip_vloss`         | bool    | `True`           | Clip value loss |
| `--ent_coef`           | float   | `0.01`           | Entropy loss coefficient |
| `--vf_coef`            | float   | `0.5`            | Value function loss coefficient |
| `--max_grad_norm`      | float   | `0.5`            | Gradient norm clipping |
| `--target_kl`          | float   | `None`           | Target KL divergence |

> Runtime-Computed (Do Not Set Manually)

| Argument               | Type    | Description |
|------------------------|---------|-------------|
| `batch_size`           | int     | Computed at runtime |
| `minibatch_size`       | int     | Computed at runtime |
| `num_iterations`       | int     | Computed at runtime |

> Blendrl-Specific Arguments

| Argument                   | Type    | Default       | Description |
|----------------------------|---------|---------------|-------------|
| `--algorithm`              | str     | `"blender"`   | Algorithm name |
| `--blender_mode`           | str     | `"logic"`     | Mode: `logic` or `neural` |
| `--blend_function`         | str     | `"softmax"`   | `softmax` or `gumbel_softmax` |
| `--actor_mode`             | str     | `"hybrid"`    | Actor architecture |
| `--rules`                  | str     | `"default"`   | Ruleset used |
| `--save_steps`             | int     | `5000000`     | Steps interval for saving models |
| `--pretrained`             | bool    | `False`       | Use pretrained neural agent |
| `--joint_training`         | bool    | `False`       | Jointly train logic + neural + blender |
| `--learning_rate`          | float   | `2.5e-4`      | Neural learning rate |
| `--logic_learning_rate`    | float   | `2.5e-4`      | Logic module learning rate |
| `--blender_learning_rate`  | float   | `2.5e-4`      | Blender module learning rate |
| `--blend_ent_coef`         | float   | `0.01`        | Entropy coefficient for blending |
| `--recover`                | bool    | `False`       | Recover training from last checkpoint |
| `--reasoner`               | str     | `"nsfr"`      | `nsfr` or `neumann` |

### Playing Script

> Command-Line Argument Schema for `play_gui.py`

```bash
python play_gui.py [--arg1 value1] [--arg2 value2] ...
```

> Arguments

| Argument       | Type  | Default | Description |
|----------------|-------|---------|-------------|
| `--env-name`   | str   | `"seaquest"` | Name of the environment to play |
| `--agent-path` | str   | `"out/runs`"out/runs/..._True_20"` | Path to the trained agent |
| `--fps`        | int   | `5`     | Frames per second for rendering |
| `--seed`       | int   | `0`     | Random seed for reproducibility |


### Evaluation Script

> Command-Line Argument Schema for `evaluate.py`

```bash
python evaluate.py [--arg1 value1] [--arg2 value2] ...
```

> Arguments

| Argument       | Type  | Default | Description |
|----------------|-------|---------|-------------|
| `--env-name`   | str   | `"seaquest"` | Name of the environment to evaluate |
| `--agent-path` | str   | `"out/runs/`"out/runs/..._True_20` | Path to the trained agent |
| `--fps`        | int   | `5`     | Frames per second for rendering |
| `--episodes`   | int   | `2`     | Number of episodes to evaluate |
| `--model`      | str   | `"blendrl"` | Type of model to evaluate: `"blendrl"` or `"neuralppo"` |
| `--device`     | str   | `"cuda:0"` | Device to run the model on (e.g., `"cuda:0"` or `"cpu"`) |

## External References

### External References

| Package/Project | Description                                       | Link                                                                                      |
|------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------|
| Conda            | Python environment manager                        | [Conda](https://anaconda.org/anaconda/conda)                                              |
| blendrl          | Logic-neural agent framework                 | [blendrl](https://github.com/ml-research/blendrl/blob/main/INSTALLATION.md)              |
| NSFR             | Neural-symbolic forward reasoning engine          | [NSFR](https://github.com/ml-research/blendrl/tree/main/nsfr) (part of blendrl repo)     |
| NUDGE            | Logic learning framework                          | [NUDGE](https://github.com/ml-research/blendrl/tree/main/nudge) (part of blendrl repo)   |
| Neumann          | Memory efficient reasoning module                  | [Neumann](https://github.com/ml-research/blendrl/tree/main/neumann) (part of blendrl repo)    |
| OC_Atari         | Object-centric Atari environment                  | [OC_Atari](https://github.com/k4ntz/OC_Atari/blob/master/README.md)                      |
| HackAtari        | RAM-exposing Atari wrapper for object detection   | [HackAtari](https://github.com/k4ntz/HackAtari/blob/master/README.md)                    |
| PyG              | PyTorch Geometric: Graph learning framework       | [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)        |
