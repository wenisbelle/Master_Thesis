import torch
from torch import nn

from actor import Actor


class RandomPolicy(nn.Module):
    """Uniform over the valid actions (the cells of the window inside the map, and the base)."""

    def forward(self, per_agent_obs_td):
        valid_indices = per_agent_obs_td["valid_actions"].bool().nonzero(as_tuple=True)[0]
        pick = torch.randint(0, valid_indices.numel(), (1,))
        return valid_indices[pick].to(torch.int64).reshape(1)


class DQNPolicy(nn.Module):
    """
    Epsilon-greedy over the Q values of the actor. With eps = 0 it is the greedy
    policy the tests run (main_test.py --policy nn).
    """

    def __init__(self, eps_init: float = 1.0, **actor_kwargs):
        super().__init__()
        self.actor = Actor(**actor_kwargs)
        self.register_buffer("eps", torch.tensor(eps_init, dtype=torch.float32))

    def forward(self, per_agent_obs_td):
        valid_mask = per_agent_obs_td["valid_actions"].bool()    # (action_dim,)

        if torch.rand(()).item() < self.eps.item():
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            pick = torch.randint(0, valid_indices.numel(), (1,))

            return valid_indices[pick].to(torch.int64).reshape(1)

        with torch.no_grad():
            q_values = self.actor(per_agent_obs_td)
        return q_values.argmax(dim=-1, keepdim=True)

    def update_epsilon(self, new_eps):
        self.eps.fill_(new_eps)


def load_greedy_policy(checkpoint_path: str, device: str = "cpu", **actor_kwargs) -> DQNPolicy:
    """
    The actor of a checkpoint written by train_logs.TrainingLogger, as a greedy policy (eps = 0).
    The network is built with the arguments saved in the checkpoint, actor_kwargs are only
    used for a checkpoint without them.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy = DQNPolicy(eps_init=0.0, **(checkpoint.get("actor_kwargs") or actor_kwargs))
    policy.actor.load_state_dict(checkpoint["actor_state_dict"])
    policy.eval()
    return policy
