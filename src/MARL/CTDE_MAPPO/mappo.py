"""
mappo.py — MAPPO components: worker-side policy, centralized critic,
and the on-policy rollout buffer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict import TensorDict

from actor import Actor


class MAPPOPolicy(nn.Module):
    def __init__(self, **actor_kwargs):
        super().__init__()
        self.actor = Actor(**actor_kwargs)

    def forward(self, per_agent_obs_td) -> TensorDict:
        logits = self.actor(per_agent_obs_td)          
        dist = Categorical(logits=logits)
        action = dist.sample()                         
        log_prob = dist.log_prob(action)            
        return TensorDict({
            "action":   action.reshape(1).to(torch.int64),
            "log_prob": log_prob.reshape(1),
        }, batch_size=[])


class CentralizedCritic(nn.Module):
    """
    V(global_state, agent_id). Consumes the `global_state` subtree:
        full_map               (B, H, W)   in [-5, 5]
        global_map_uncertainty (B, 1, 2)
        all_positions          (B, N, 2)
        all_active             (B, N) bool
    plus a one-hot of agent_idx so each agent gets its own value estimate.
    """
    def __init__(self, max_num_agents: int, action_dim: int = 11, map_h: int = 50, map_w: int = 50,
                 map_feature_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.N = max_num_agents
        self.map_h, self.map_w = map_h, map_w
        self.action_dim = action_dim

        self.map_stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.action_dim, self.action_dim)),
        )
        self.map_proj = nn.Linear(64 * self.action_dim * self.action_dim, map_feature_dim)

        vec_in = 2 + self.N * 2 + self.N + self.N   # unc (mean, std) + positions + active + one-hot
        self.head = nn.Sequential(
            nn.Linear(map_feature_dim + vec_in, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_td: TensorDict, agent_idx: torch.Tensor) -> torch.Tensor:
        fm  = global_td["full_map"].float()
        if fm.dim() == 3:
            fm = fm.unsqueeze(1)                       # (B, 1, H, W)
        unc = global_td["global_map_uncertainty"].float().flatten(1)   # (B, 2)
        pos = global_td["all_positions"].float().flatten(1)            # (B, N*2)
        act = global_td["all_active"].float()                          # (B, N)
        one_hot = F.one_hot(agent_idx.squeeze(-1).long(), self.N).float()

        m = F.relu(self.map_proj(self.map_stem(fm).flatten(1)))
        x = torch.cat([m, unc, pos, act, one_hot], dim=-1)
        return self.head(x)                            # (B, 1)


class RolloutBuffer:
    """
    On-policy buffer: same interface as your ReplayBuffer for
    WorkersOrchestrator.collect (load_transitions / __len__),
    but data is stacked and DISCARDED after each PPO update.
    """
    def __init__(self):
        self._parts = []

    def load_transitions(self, td: TensorDict):
        if td is not None and td.batch_size[0] > 0:
            self._parts.append(td)

    def get_all(self) -> TensorDict | None:
        if not self._parts:
            return None
        return torch.cat(self._parts, dim=0)

    def clear(self):
        self._parts = []

    def __len__(self):
        return sum(p.batch_size[0] for p in self._parts)