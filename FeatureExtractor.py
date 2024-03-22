import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.width = 8
        self.height = 8
        self.n_input_channels = 3  # observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.n_input_channels, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            x = observation_space.sample()[None]
            env = x[:, :self.n_input_channels * 64].reshape(1,
                                                            self.n_input_channels,
                                                            self.height, self.width)
            n_flatten = self.cnn(
                th.as_tensor(env).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        env = observations[:, :self.n_input_channels * 64].reshape(batch_size,
                                                                   self.n_input_channels,
                                                                   self.height,
                                                                   self.width)  # double-check the reshape here
        env_repr = self.cnn(env)
        env_repr = self.linear(env_repr)  # env_repr.shape: [64, 576]
        parameters = observations[:, self.n_input_channels * 64:]
        features = th.concatenate([env_repr, parameters], dim=1)
        return features

# policy_kwargs = dict(
#     features_extractor_class=FeatureExtractor,
#     features_extractor_kwargs=dict(features_dim=256),
# )
# model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
# model.learn(1000)
