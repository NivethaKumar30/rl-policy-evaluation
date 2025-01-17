NAME : NIVETHA .K

REG NO: 212222230102

# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

## POLICY EVALUATION FUNCTION

States
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.
Actions
The agent can take two actions:

*R: Move right. *L: Move left.

Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.


Graphical Representation

![image](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/72af9e1d-10e3-4611-bc9a-ffcbf1e58c99)

POLICY EVALUATION FUNCTION

Formula

![image](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/85aad39a-e90d-496c-9588-b510ae1c3482)

PROGRAM

```

pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

import warnings ; warnings.filterwarnings('ignore')
import gym, gym_walk
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
# Reference https://github.com/mimoralea/gym-walk
```

```
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```

```

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```

```

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
```

```

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
```

Slippery Walk Five MDP:
```

# SLIPPERY WALK

env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)
P
init_state
# First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)
# Find the probability of success and the mean return of the first policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))
# Create your own policy


pi_2 = lambda s: {
    0:RIGHT, 1:LEFT, 2:RIGHT, 3:RIGHT, 4:RIGHT, 5:LEFT, 6:RIGHT
}[s]

print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)

# Find the probability of success and the mean return of you your policy

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))

# Compare your policy with the first policy


# The implementation of first code has resulted in success rate of 3% while the second policy has resulted in improving the result of reaching the goal. The success rate for second policy is 88%.

```

Policy Evaluation:

```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy

V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Comparing policies based on value function

# The state value function of the second policy V2 is greater than that of the first policy V1, so we conclude that the second policy is the best policy.

V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```
## OUTPUT:

Policy 1:

![Screenshot 2024-03-19 144925](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/93465399-e0e3-4acd-9cc2-81a407ccc323)

![Screenshot 2024-03-19 145106](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/53a65aa1-2a94-46d2-ad9a-4d322b2dded3)


Policy 2:

![Screenshot 2024-03-19 170250](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/a6010c01-205d-4b13-9839-28f014ed9b91)

![Screenshot 2024-03-19 170405](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/dcfe4a23-13d3-4f9f-bcd0-1d4d46bc5327)


Comparison:

![image](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/fa0b0a63-4d10-4c9a-aad6-1e8b8fb3110c)

Conclusion:

![Screenshot 2024-03-19 170131](https://github.com/NivethaKumar30/rl-policy-evaluation/assets/119559844/81186d9a-3185-45db-a429-208c24eb0b33)


## RESULT:

Thus, a Python program is developed to evaluate the given policy.
