from dataclasses import dataclass
from typing import List

# A class to represent an observation.
@dataclass
class Observation:
    reward : float
    drifts : List[float]

# A class to represent an arm.
@dataclass
class Arm:
    id : int
    mean_reward : float  # true expected reward
    mean_drifts : List[float]  # true expected drifts
    obs_rewards : List[float]
    obs_drifts : List[List[float]]
    empirical_reward : float  # empirical mean reward
    empirical_drifts : List[float]  # empirical mean drifts

    def update(self, obs : Observation):
        """
        Update the observed and empirical reward / drifts using the new
        observation.
        """
        reward, drifts = obs.reward, obs.drifts
        num_resources = len(self.mean_drifts)
        num = len(self.obs_rewards)
        self.obs_rewards.append(reward)
        self.obs_drifts.append(drifts)
        self.empirical_reward = (num * self.empirical_reward + reward) / (num + 1)
        self.empirical_drifts = [
            (num * self.empirical_drifts[j] + drifts[j]) / (num + 1) for j in range(num_resources)
        ]

# A class to represent an instance of the problem.
@dataclass
class Instance:
    T : int
    B : float
    arms : List[Arm]
    num_resources : int

def parse_datafile(filename : str) -> Instance:
    """
    Create an instance from the given datafile.  The format of the datafile is:
    T
    B
    n
    m
    mean reward of arm 0
    mean drift of (arm 0, resource 1), ..., mean drift of (arm 0, resource m)
    mean reward of arm 1
    mean drift of (arm 1, resource 1), ..., mean drift of (arm 1, resource m)
    ...
    mean reward of arm n
    mean drift of (arm n, resource 1), ..., mean drift of (arm n, resource m)
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    T = int(lines[0])
    B = float(lines[1])
    num_arms = int(lines[2])
    num_resources = int(lines[3])
    arms = []
    idx = 4
    for _ in range(num_arms):
        mean_reward = float(lines[idx].strip())
        idx += 1
        mean_drifts = [float(x) for x in lines[idx].strip().split(',')]
        idx += 1
        arms.append(Arm(
            id = len(arms),
            mean_reward = mean_reward,
            mean_drifts = mean_drifts,
            obs_rewards = [],
            obs_drifts = [],
            empirical_reward = 0,
            empirical_drifts = [0] * len(mean_drifts)
        ))

    return Instance(
        T = T,
        B = B,
        arms = arms,
        num_resources = num_resources
    )