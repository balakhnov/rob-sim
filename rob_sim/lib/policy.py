import numpy as np
import torch

class NNPolicy:
    def __init__(self, str) -> None:
        self.policy = torch.load(str)

    def get_action(self, obs):
        obs = torch.from_numpy(obs[np.newaxis,:])
        action = self.policy(obs,True)
        act = action.detach().numpy()[0]
        return act
