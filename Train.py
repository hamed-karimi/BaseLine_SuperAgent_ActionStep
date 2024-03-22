import os.path
import shutil

import torch
# from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
# from sbx import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv
from FeatureExtractor import FeatureExtractor
from CallBack import CallBack
from stable_baselines3.common.callbacks import CheckpointCallback
from Environment import Environment
from ActorCritic import Policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_checker import check_env
from Environment import valid_actions


class Train:
    def __init__(self, utils):
        self.params = utils.params
        self.root_dir = self.params.STORE_DIR
        self.policy_net_size = self.params.POLICY_NET_SIZE
        self.value_net_size = self.params.VALUE_NET_SIZE
        self.episode_num = self.params.EPISODE_NUM
        self.batch_size = self.params.BATCH_SIZE
        self.step_num = self.params.EPISODE_STEPS
        self.device = self.params.DEVICE
        self.res_folder, self.res_name = utils.make_res_folder(root_dir=self.root_dir)
        self.log_dir = os.path.join(self.res_folder, 'log')
        self.tensorboard_call_back = CallBack(res_dir=self.res_folder, log_freq=self.params.PRINT_REWARD_FREQ, )

    def train_policy(self):
        print('start')
        vec_env = make_vec_env(Environment, n_envs=self.params.ENVIRONMENT_NUM,
                               env_kwargs=dict(params=self.params,
                                               few_many_objects=['few', 'many']),
                               vec_env_cls=SubprocVecEnv
                               )
        # vec_env = Environment(self.params, few_many_objects=['few', 'many'])
        # vec_env = ActionMasker(env=vec_env, action_mask_fn=valid_actions)
        # check_env(vec_env)
        # SubprocVecEnv
        # vec_env = VecMonitor(vec_env)
        print('after vec_env')
        # vec_env = VecMonitor(venv=vec_env, filename=self.log_dir)
        # "Tried to reset an environment before done. If you want to allow early resets, "
        # "wrap your env with Monitor(env, path, allow_early_resets=True)"

        checkpoint_callback = CheckpointCallback(
            save_freq=self.params.CHECKPOINT_SAVE_FREQUENCY,
            save_path=self.res_folder,
            name_prefix="PPO",
            save_replay_buffer=False,
            save_vecnormalize=False
        )
        print('after checkpoint')
        policy_kwargs = dict(
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        print('after policy_kwargs')
        model = MaskablePPO(policy=Policy,
                            env=vec_env,
                            policy_kwargs=policy_kwargs,
                            learning_rate=self.params.INIT_LEARNING_RATE,
                            # batch_size=self.batch_size,
                            gamma=self.params.GAMMA,
                            verbose=0,
                            n_steps=self.step_num,
                            # n_epochs=1,
                            tensorboard_log='./runs',
                            device=self.device)
        print('after model')
        if self.params.PRE_TRAINED_MODEL_VERSION != "":
            model.set_parameters('./{0}/model.zip'.format(self.params.PRE_TRAINED_MODEL_VERSION))

        print('before learning')
        model.learn(self.episode_num,
                    log_interval=100,
                    # callback=[self.tensorboard_call_back, checkpoint_callback],
                    tb_log_name=self.res_name
                    )

        model.save(os.path.join(self.res_folder, 'model'))
        # shutil.copytree(self.res_folder, os.path.join('./', self.res_name))
