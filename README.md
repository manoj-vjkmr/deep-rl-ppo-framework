# Deep Reinforcement Learning with Recurrent Proximal Policy Optimization (PPO)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement-Learning-green.svg)
![PPO](https://img.shields.io/badge/Algorithm-PPO-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## Overview

This project implements a **Deep Reinforcement Learning framework** based on **Proximal Policy Optimization (PPO)** for training an autonomous agent to play *Super Mario Bros*. Instead of relying on existing RL libraries such as Stable-Baselines or RLlib, the core reinforcement learning pipeline—including rollout storage, Generalized Advantage Estimation (GAE), PPO optimization, recurrent policy learning, and parallel environment execution—has been implemented in PyTorch.

The project focuses on building a modular and extensible reinforcement learning system capable of learning directly from visual observations while maintaining temporal memory through a recurrent neural network.

---

## Highlights

* Custom implementation of **Proximal Policy Optimization (PPO)**
* Recurrent **Actor-Critic** architecture using **GRU**
* Parallel experience collection using multiprocessing
* Generalized Advantage Estimation (GAE)
* Reward shaping for improved learning efficiency
* Frame preprocessing and stochastic frame skipping
* TensorBoard logging for training visualization
* Modular and extensible codebase

---

## Motivation

Training agents directly from high-dimensional visual observations remains a challenging problem in reinforcement learning. This project explores how modern policy-gradient methods can learn effective control policies through:

* Visual feature extraction using Convolutional Neural Networks
* Temporal memory using recurrent neural networks
* Stable policy optimization via PPO
* Efficient experience collection using parallel environments

Although Super Mario Bros serves as the benchmark environment, the architecture is designed as a general reinforcement learning framework applicable to other visual control tasks.

---

## System Architecture

```
                 +----------------------+
                 | Parallel Mario Envs  |
                 |   (32 Processes)     |
                 +----------+-----------+
                            |
                     Observations
                            |
                            ▼
                  Experience Storage
                            |
                Generalized Advantage
                     Estimation (GAE)
                            |
                            ▼
              Recurrent Actor-Critic Policy
             ┌─────────────────────────────┐
             │ CNN Feature Extractor       │
             │ Previous Action Encoder     │
             │ Feature Fusion              │
             │ GRU Memory Module           │
             │ Actor Head                 │
             │ Critic Head                │
             └─────────────────────────────┘
                            |
                     PPO Optimization
                            |
                      Updated Policy
```

---

## Policy Network

The policy network follows a recurrent actor-critic design.

### Feature Extraction

* Four convolutional layers process grayscale game frames.
* Orthogonal weight initialization improves optimization stability.

### Temporal Memory

A **Gated Recurrent Unit (GRU)** maintains hidden state across timesteps, allowing the policy to reason over temporal dependencies instead of making decisions from a single frame.

### Previous Action Encoding

Rather than relying solely on image observations, the previous four actions are one-hot encoded and embedded before being fused with visual features. This provides additional temporal context to the policy.

### Actor-Critic Heads

The shared representation branches into:

* **Actor:** predicts a probability distribution over actions.
* **Critic:** estimates the state value function.

---

## Reinforcement Learning Pipeline

### Environment

The environment includes several preprocessing wrappers:

* Image resizing
* Grayscale conversion
* Stochastic frame skipping
* Reward shaping
* Action space simplification

### Experience Collection

Multiple game environments run simultaneously using Python multiprocessing, significantly increasing sample throughput compared to sequential rollouts.

### Rollout Storage

A custom experience buffer stores:

* observations
* actions
* rewards
* value estimates
* log probabilities
* recurrent hidden states
* masks for episode termination

### Advantage Estimation

Advantages are computed using **Generalized Advantage Estimation (GAE)** to reduce variance while preserving learning stability.

### PPO Optimization

The policy is updated using the clipped PPO objective with:

* clipped surrogate loss
* value loss
* entropy regularization
* gradient clipping
* learning rate scheduling

---

## Hyperparameters

| Parameter                 | Value  |
| ------------------------- | ------ |
| Parallel Environments     | 32     |
| Rollout Length            | 128    |
| Hidden Dimension          | 512    |
| GRU Hidden Size           | 512    |
| Previous Action Embedding | 256    |
| Discount Factor (γ)       | 0.99   |
| GAE λ                     | 0.95   |
| PPO Epochs                | 4      |
| PPO Minibatches           | 16     |
| PPO Clip ε                | 0.2    |
| Learning Rate             | 5.5e-4 |

---

## Project Structure

```
.
├── train.py                 # Training pipeline
├── run.py                   # Inference script
├── policy.py                # Recurrent Actor-Critic network
├── agent.py                 # PPO optimization
├── experience.py            # Rollout storage & GAE
├── environment.py           # Environment wrappers
├── arguments.py             # Hyperparameters
├── models/                  # Saved checkpoints
└── README.md
```

---

## Training

Train the PPO agent

```bash
python train.py
```

Training statistics including reward progression, policy loss, value loss, and entropy are logged using TensorBoard.

Launch TensorBoard

```bash
tensorboard --logdir=runs
```

---

## Inference

Run a trained policy

```bash
python run.py --world 1 --stage 1
```

---

## Technical Contributions

This implementation includes:

* Custom PPO optimization pipeline
* Recurrent policy learning using GRU
* Generalized Advantage Estimation
* Parallel environment execution
* Modular rollout storage
* Reward shaping
* TensorBoard experiment logging
* Learning rate scheduling
* Gradient clipping
* Orthogonal network initialization

---

## Skills Demonstrated

* Deep Reinforcement Learning
* Policy Gradient Methods
* Proximal Policy Optimization (PPO)
* Recurrent Neural Networks
* PyTorch
* Parallel Computing
* Computer Vision
* Reinforcement Learning Environment Design
* Software Engineering for Machine Learning
* Experiment Tracking

---

## Acknowledgements

The overall reinforcement learning approach was inspired by the following medium article:

**Playing Super Mario with PPO Reinforcement Learning Agent** by Dario Prawira Teh.
https://medium.com/@darioprawarateh/playing-super-mario-with-ppo-reinforcement-learning-agent-e9f61fd04b32

