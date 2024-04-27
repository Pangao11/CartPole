# CartPole-v0 with Policy Gradient

## Overview
This repository contains an implementation of a policy gradient method to solve the CartPole-v1 task from OpenAI's Gym. The project demonstrates the application of reinforcement learning techniques to develop an agent capable of balancing a pole on a cart.

## Implementation Details
- **Environment**: The `CartPole-v0` environment from OpenAI Gym challenges an agent to balance a pole attached by an un-actuated joint to a cart that moves along a frictionless track.
- **Algorithm**: We utilize a policy gradient method, specifically a variant of the REINFORCE algorithm, to train the agent. The model updates policies directly based on the outcomes of each episode, optimizing the expected reward.
- **Network Architecture**: The policy network consists of a neural network with one hidden layer containing 128 neurons with ReLU activation, and a softmax output layer to generate action probabilities.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.7 or newer
- Gym
- NumPy
- PyTorch

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Pangao11/CartPole.git
2. Running the Program
Execute the training script to start training the agent:
  ···bash
  python CartPole.py
