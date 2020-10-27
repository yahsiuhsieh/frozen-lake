# Frozen Lake

Value Iteration, Policy Iteration and Q learning on Frozen lake gym env

<p align="center">
  <img width="400" height="350" src="https://github.com/arthur960304/frozen-lake/blob/main/images/frozen-lake.png">
</p>

The goal of this game is to **go from the starting state (S) to the goal state (G)** by walking only on frozen tiles (F) and avoid holes (H). However, the ice is slippery, so you won't always move in the direction you intend **(stochastic environment)**.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Built With

* Python 3.6.10

* gym >= 0.15.4

* numpy >= 1.16.2

* matplotlib >= 3.1.1

## Code Organization
```
.
├── src                     # Python scripts
│   ├── value_iteration.py  # VI algorithm
│   ├── policy_iteration.py # PI algorithm
│   ├── q_learning.py       # Q learning algorithm
│   └── utils.py            # Utility sets
├── images                  # Results
└── README.md
```

## Tests

There are 3 methods you can try, namely policy iteration, value iteration, and Q learning, with corresponding file name.

ex. if you want to try policy iteration, just do
```
python policy_iteration.py
```

## Results

The resulting image would show the average success rate versus the number of episode. 

The example below shows the average success rate of value iteration algorithm over 50 episodes.

<p align="center">
  <img width="500" height="300" src="https://github.com/arthur960304/frozen-lake/blob/main/images/VI.png">
</p>

## Authors

* **Arthur Hsieh** - *Initial work* - [arthur960304](https://github.com/arthur960304)
