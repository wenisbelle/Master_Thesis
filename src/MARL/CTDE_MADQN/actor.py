"""
actor.py — Per-agent Q network for the cooperative drone-mapping MARL setup.

 INTEGRATION

The orchestrator / worker calls the policy as:

        policy(per_agent_obs_td) -> action_tensor of shape (1,)

See `simulation_orchestrator.AsyncMARLOrchestrator._handle_flags` and the
`policy_callable` defined inside `worker._worker_loop`. The policy (policy.py)
calls `Actor.forward`, which returns the Q value of every action.


 OBSERVATION FIELDS CONSUMED (built by protocol.observe, see env/gradysim_environment/protocol.py)

For a single agent (the orchestrator slices these out per-agent):

  - large_map_patch  : float32, (3, L, L)  the whole map, averaged in blocks (L = 50 for the 100x100 map)
  - small_map_patch  : float32, (3, A, A)  the action window around the drone (A = 21)
                       channels: uncertainty (normalized), distance to the drone, energy margin
  - agent_state      : float32, (15,)      own position, battery, base, and the swarm estimate
                                           (mean battery of the others, how many charging, Nc)
  - encounter_state  : float32, (9,)       the other drone of an encounter, zeros otherwise
  - valid_actions    : bool,    (A*A + 1,)

For BATCHED inputs (during training), each tensor has an extra leading dim.
We detect this automatically below.

 ACTION SPACE

Discrete, A*A + 1: the cells of the A x A window centred on the drone (index
row * A + col, the same layout as small_map_patch) and the base (index A*A).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


# Sub-modules: one encoder per modality, then a fusion

class DualMapEncoder(nn.Module):
    """
    Two-resolution map encoder.
        large : (B, in_channels, N, N)  — the whole map, downsampled
        small : (B, in_channels, M, M)  — the action window, full resolution
        The small should have the same size as the action window.
    The large branch is convolved and downsampled to M×M, concatenated on the
    channel axis with the small branch, jointly convolved, flattened, projected.

    Change from the version without energy: the joint head ends with a stride 2
    convolution. The action window grew from 11 to 21 cells, and without it the
    flattened features (128 * 21 * 21) would make the projection 7M parameters.
    """
    def __init__(self, in_channels: int = 3, large_size: int = 50,
                 small_size: int = 21, feature_dim: int = 128):
        super().__init__()
        self.large_size = large_size
        self.small_size = small_size

        # large
        self.large_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((small_size, small_size)),     # -> (B, 64, M, M)
        )
        # small
        self.small_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )                                                       # -> (B, 64, M, M)
        # joint head: concat on channels, convolve at M×M, then downsample
        self.joint_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),         # -> (B, 64, ceil(M/2), ceil(M/2))
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(self._infer_flat_dim(in_channels), feature_dim)

    @torch.no_grad()
    def _infer_flat_dim(self, in_channels: int) -> int:
        large = torch.zeros(1, in_channels, self.large_size, self.large_size)
        small = torch.zeros(1, in_channels, self.small_size, self.small_size)
        return self._fuse(large, small).flatten(1).shape[1]

    def _fuse(self, large, small):
        l = self.large_stem(large)        # (B, 64, M, M)
        s = self.small_stem(small)        # (B, 64, M, M)
        x = torch.cat([l, s], dim=1)      # (B, 128, M, M)
        return self.joint_conv(x)         # (B, 64, M/2, M/2)

    def forward(self, large, small):
        x = self._fuse(large, small).flatten(1)
        return F.relu(self.proj(x))               # (B, feature_dim)


class VectorEncoder(nn.Module):
    """
    MLP that encodes the non-spatial fields (`agent_state`, `encounter_state`).

    We concatenate everything into a single vector
    """

    def __init__(self, in_dim: int, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class Actor(nn.Module):
    """
    Maps a per-agent observation TensorDict to the Q value of every action.

    Batched vs unbatched
    --------------------
    During rollouts the orchestrator slices a single agent's obs (no batch
    dim). During training you'll batch many transitions together (batch dim
    present). `_features` detects which is which by inspecting `agent_state`,
    so callers don't have to worry about it.
    """

    def __init__(
        self,
        *,
        action_dim: int = 21 * 21 + 1,  # size of the action space: the window and the base
        map_channels: int = 3,          # uncertainty, distance, energy margin
        large_map_size: int = 50,
        small_map_size: int = 21,
        agent_state_dim: int = 15,
        encounter_state_dim: int = 9,
        map_feature_dim: int = 128,
        vector_feature_dim: int = 64,
        hidden_dim: int = 256,
        # Keys inside per_agent_obs_td — change here if you rename them in the env.
        large_map_key: str = "large_map_patch",
        small_map_key: str = "small_map_patch",
        agent_state_key: str = "agent_state",
        encounter_state_key: str = "encounter_state",
        valid_actions_key: str = "valid_actions",
    ):
        super().__init__()
        ##### Saved in the checkpoints, so a test rebuilds the same network (policy.load_greedy_policy) #####
        self.kwargs = dict(action_dim=action_dim, map_channels=map_channels, large_map_size=large_map_size,
                           small_map_size=small_map_size, agent_state_dim=agent_state_dim,
                           encounter_state_dim=encounter_state_dim, map_feature_dim=map_feature_dim,
                           vector_feature_dim=vector_feature_dim, hidden_dim=hidden_dim)
        self.action_dim = action_dim

        self.large_map_key = large_map_key
        self.small_map_key = small_map_key
        self.agent_state_key = agent_state_key
        self.encounter_state_key = encounter_state_key
        self.valid_actions_key = valid_actions_key

        # Vector input dim = agent_state + encounter_state.
        # If you later add or remove obs fields, update the dims (they come from the
        # protocol, AGENT_STATE_DIM and ENCOUNTER_STATE_DIM) and the concatenation in `_features`.
        self.vector_in_dim = agent_state_dim + encounter_state_dim

        ##### Inputs
        self.map_encoder = DualMapEncoder(
            in_channels=map_channels,
            large_size=large_map_size,
            small_size=small_map_size,
            feature_dim=map_feature_dim,
        )
        self.vec_encoder = VectorEncoder(
            in_dim=self.vector_in_dim, feature_dim=vector_feature_dim
        )

        # Fusion merges the CNN and the vector features. Simple MLP
        fused_dim = map_feature_dim + vector_feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.q_head = nn.Linear(hidden_dim, action_dim)

    # Internal: extract tensors from the TensorDict
    # run encoders, and return the fused features.

    def _features(self, obs_td: TensorDict) -> Tuple[torch.Tensor, bool]:
        """
        Returns:
            features:    (B, hidden_dim) fused embedding
            is_unbatched: True if caller passed a single sample, so we know
                          to squeeze the leading dim out of the output later.
        """
        agent_state = obs_td[self.agent_state_key].float()          # unbatched (S,) | batched (B, S)
        encounter_state = obs_td[self.encounter_state_key].float()  # unbatched (E,) | batched (B, E)
        large_map = obs_td[self.large_map_key].float()              # unbatched (C, L, L) | batched (B, C, L, L)
        small_map = obs_td[self.small_map_key].float()              # unbatched (C, A, A) | batched (B, C, A, A)

        # Detect batching from `agent_state` (1-D = single sample, 2-D = batched).
        is_unbatched = agent_state.dim() == 1
        if is_unbatched:
            agent_state = agent_state.unsqueeze(0)
            encounter_state = encounter_state.unsqueeze(0)
            large_map = large_map.unsqueeze(0)
            small_map = small_map.unsqueeze(0)

        # Encode each modality and concatenate
        map_feat = self.map_encoder(large_map, small_map)               # (B, map_feature_dim)
        vec_in = torch.cat([agent_state, encounter_state], dim=-1)      # (B, vector_in_dim)
        vec_feat = self.vec_encoder(vec_in)                             # (B, vector_feature_dim)

        fused = torch.cat([map_feat, vec_feat], dim=-1)
        return self.trunk(fused), is_unbatched     # (B, hidden_dim), bool

    def forward(
        self,
        obs_td: TensorDict
        ) -> torch.Tensor:
        """
        Run the network and return the Q values, -inf on the invalid actions.
        Returns:
            Shape (action_dim,) when the input was unbatched, or (B, action_dim) when batched.
        """
        h, is_unbatched = self._features(obs_td)
        q_values = self.q_head(h)                  # (B, action_dim)

        try:
            valid = obs_td[self.valid_actions_key]
            if is_unbatched and valid.dim() == 1:
                valid = valid.unsqueeze(0)
            q_values = q_values.masked_fill(~valid.bool(), float("-inf"))
        except KeyError:
            pass

        return q_values.squeeze(0) if is_unbatched else q_values
