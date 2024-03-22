from typing import Callable, Tuple
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch as th
from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class ActorCritic(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 184), nn.ReLU(),
            nn.Linear(184, 128), nn.ReLU(),
            nn.Linear(128, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 184), nn.ReLU(),
            nn.Linear(184, 128), nn.ReLU(),
            nn.Linear(128, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # Two additional linear layers (probably without ReLU) are also attached to the last layer specified here
        return self.forward_actor(features), self.forward_critic(features)  # features: (64, 262)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class Policy(MaskableActorCriticPolicy):
# class Policy(PPOPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False  # Don't change this or the whole thing falls apart!!
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # self.mlp_extractor = ActorCritic(self.features_dim)
        self.mlp_extractor = ActorCritic(256+6)


# policy_kwargs = dict(
#     features_extractor_class=FeatureExtractor,
#     features_extractor_kwargs=dict(features_dim=128),
# )
# model = PPO(Policy, "CartPole-v1", verbose=1)
# model.learn(5000)
