# Reliability assessment of off-policy deep reinforcement learning: a benchmark for aerodynamics  

This codebase was utilized to generate the results presented in the paper titled *"Reliability assessment of off-policy deep reinforcement learning: a benchmark for aerodynamics."* The repository includes implementations of three reinforcement learning algorithms—DDPG, TD3, and SAC—along with the necessary setup to reproduce and analyze the benchmark results. For detailed information on the experiments, methodology, and findings, please refer to the associated paper.
  
This project examines three existing reinforcement learning algorithms which store collected samples in a replay buffer: **DDPG**, **TD3**, and **SAC**. These are evaluated and compared on a fluid mechanics benchmark which consists in **controlling an airfoil to reach a target**. The problem is solved with two different levels of data collection complexity: either a **low-cost low-order model** or with a high-fidelity **Computational Fluid Dynamics** (CFD) approach.
  
In practice, two different control tasks are performed. First, both the starting and target points are kept in a fixed position during both the learning and testing of the policy, whereas in the second task, the target may be anywhere in a given domain. The code allows to evaluate the three DRL algorithms on both tasks, when solving the physics with either a low-order or a high-fidelity model, and with various DRL hyperparameters, reward formulations, and environment parameters controlling the dynamics.  
  
In order to facilitate the reproducibility of our results without requiring an in-depth understanding of the code, each case study is stored in a separate repository containing all the necessary code and setup to execute the case directly. The code for the following tasks can be found in the respective repositories:  
- [First task with fixed target and low-order model](https://github.com/SuReLI/aerobench_fixed_target_low_order)
- [First task with fixed target and CFD model](https://github.com/SuReLI/aerobench_fixed_target_star)
- [Second task with variable target and low-order model](https://github.com/SuReLI/aerobench_variable_target_low_order)  
- [Second task with variable target and CFD model](https://github.com/SuReLI/aerobench_variable_target_star) **<-- You are here**.
  
  
### Available algorithms  
- **DDPG** : Deep Deterministic Policy Gradient presented in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).  
- **TD3** : Twin Delayed Deep Deterministic policy gradient [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)  
- **SAC** : Soft Actor-Critic presented in [Soft Actor-Critic:Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)  

 
## Installation  
**Note:** Running this repository requires the use of the STAR-CCM+ solver and the corresponding license. The tutorial below assumes STAR-CCM+ is already installed, you have access to a license, and you have basic knowledge on how to use it.

Ensure you have Python 3.7 or older versions installed, and then install the necessary Python libraries:  
  
```bash  
pip install torch torchvision imageio gym matplotlib PyYAML numpy  
```  
Clone the repository:  
```bash  
git clone https://github.com/SuReLI/aerobench_variable_target_star
```  

## Usage  
  
### Training phase  
Navigate to the directory to run the desired task:
```bash  
cd aerobench_variable_target_star
```  
To initiate training, you need to run **two commands in the order** described below.
1. **First**, launch the Reinforcement Learning algorithm with the following command:
```bash
python3 -u train <agent> --appli='starccm'  
```  
Replace `<agent>` with one of the following values: DDPG, TD3, or SAC.
  
Optional parameters for loading pre-existing models and memory replay buffers and continue the training from there are available:  
- `--load`: Load a pre-existing model.  
- `--loadrm`: Load a pre-existing memory buffer.
as well as for setting the number of episodes for each evaluation:
- `-n` or `--nb_tests`: Set the number of evaluation episodes.
  
Example:  
```bash  
python3 -u train SAC --appli='starccm' -n 4 --load='results/SAC/STARCCMexternalfiles_2023-12-13_16-46-40' --loadrm='results/SAC/STARCCMexternalfiles_2023-12-13_16-46-40'
```  
This command trains the specified reinforcement learning agent (SAC in this case) on the 'starccm' application with the option to load a pre-existing model (soft_actor.pth, critic_target and critic.pth) from the folder `results/SAC/STARCCMexternalfiles_2023-12-13_16-46-40/models/` and a pre-existing memory buffer (replay_memory.yaml) from the folder `results/SAC/STARCCMexternalfiles_2023-12-13_16-46-40/`. Besides, evaluation is performed each time on 4 episodes.

2. **Second**, launch the CFD solver:
```bash
cd cfd/starccm
starccm -batch macro_externalfiles.java flatplate_coarse.sim
``` 

Alternatively, if you are on a supercomputer, you can launch the training using two slurlm files, one for the Reinforcement Learning algorithm and an other one for the CFD solver. Examples of each slurlm files, called respectively **submit_example** and **submit_example_star**, are provided in the repository.


  
### Testing phase
Navigate to the directory root:
```bash 
cd aerobench_variable_target_star
``` 
Initiating testing is similar to what you had to do for training, you also need to run **two commands in the order** described below.
1. **First**, launch the Reinforcement Learning algorithm with the following command:
```bash
python3 -u test <agent> --appli='starccm'
``` 
Just like in the training phase, replace `<agent>` with one of the following values: DDPG, TD3, or SAC.

Optional parameters for testing are available:

-   `-n` or `--nb_tests`: Set the number of test episodes.
-   `-f` or `--folder`: Specify the path to a specific result folder to test. If not provided, the default folder tested is the most recent one with a format similar to `STARCCMexternalfiles_2023-12-13_16-46-40` inside the `/results/<agent>/` directory. 
Note: the model tested is the one contained in the `/models/` subdirectory of the specified result folder.

Example:
```bash
python3 -u test SAC --appli='starccm' -n 10 -f='results/SAC/first_trial'
``` 
This command tests the pre-trained model stored in the folder `results/SAC/first_trial/models/`, on the 'starccm' application, running 10 test episodes.

2. **Second**, launch the CFD solver:
```bash
cd cfd/starccm
starccm -batch macro_externalfiles.java flatplate_coarse.sim
``` 

Alternatively, if you are on a supercomputer, you can launch the testing using two slurlm files, one for the Reinforcement Learning algorithm and an other one for the CFD solver. Examples of each slurlm files, called respectively **submit_example** and **submit_example_star**, are provided in the repository.
 

## Outputs

After running the training or testing phases, the code generates various outputs and results. Below is an overview of the key directories and files you can expect:

### Training Outputs:

For each training, results are stored in a directory of the form `results/<agent>/STARCCMexternalfiles_date/`, where date is the date at which the training started. The folder contains the following outputs:
- **training plot (`train_output.png`):** a visual representation of the training (return, specific trajectories and location of point B)
- **model checkpoints (`models/*.pth`)**,
- **memory buffer (`replay_memory.yaml`)**
- **additional variable files (`variables/*.csv`):** contain CSV files with the values of various variables during the training episodes.
- **configuration File (`config.yaml`):** a copy of the configuration file used for the specific training run.

### Testing Outputs:

For each testing, results are stored in a sub-directory of the results directory tested :`results/<agent>/STARCCMexternalfiles_date/test`. The `test` folder contains the following outputs:

- **testing plot (`test_output.png`):** a visual representation of the testing (return, specific trajectories and location of point B)
- **additional variable files (`variables/*.csv`):** contain CSV files with the values of various variables during the testing episodes.


  
## Run customized cases

To customize the case, one can adjust the values of various parameters related to the Reinforcement Learning algorithm in the **config.yaml** file. Regarding CFD parameters, some can be changed directly in the **config.yaml**, but most of them also need to be adapted in the solver input file **flatplate_coarse.sim** or the application code **CFDcommunication.py** or the macro allowing communications between the python code and the solver **macro_externalfiles.java**. Note that this application was developed for testing purposes and is far from being optimized, both in terms of performance and user-friendliness.
 
 
## Acknowledgments  
  
The reinforcement learning algorithms implemented in this project have been adapted from the [Pytorch-RL-Agents](https://github.com/SuReLI/Pytorch-RL-Agents) repository.  
  
## Contact  
For any questions or comments, feel free to contact Sandrine Berger at [sand.qva@gmail.com](mailto:sand.qva@gmail.com).
