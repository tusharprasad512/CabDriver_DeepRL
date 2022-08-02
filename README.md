# CabDriver_RLProject
The goal of the project is to build an RL-based algorithm which can help cab drivers maximize their profits by improving their decision-making process on the field.

## Goals
1. Create the environment: You are given the ‘Env.py’ file with the basic code structure. This is the "environment class" - each method (function) of the class has a specific purpose. Please read the comments around each method carefully to understand what it is designed to do. Using this framework is not compulsory, you can create your own framework and functions as well.

2. Build an agent that learns to pick the best request using DQN. You can choose the hyperparameters (epsilon (decay rate), learning-rate, discount factor etc.) of your choice.

 3. Training depends purely on the epsilon-function you choose. If the ? decays fast, it won’t let your model explore much and the Q-values will converge early but to suboptimal values. If ? decays slowly, your model will converge slowly. We recommend that you try converging the Q-values in 4-6 hrs.  We’ve created a sample ?-decay function at the end of the Agent file (Jupyter notebook) that will converge your Q-values in ~5 hrs. Try building a similar one for your Q-network.

 4. In the Agent file, we’ve provided the code skeleton. Using this structure is not necessary though.
Convergence- You need to converge your results. The Q-values may be suboptimal since the agent won't be able to explore much in 5-6 hours of simulation. But it is important that your Q-values converge. There are two ways to check the convergence of the DQN model:

 5. Sample a few state-action pairs and plot their Q-values along episodes

 6. Check whether the total rewards earned per episode are showing stability
