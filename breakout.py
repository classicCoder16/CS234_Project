import gym
from utils.preprocess import greyscale, process_state
from utils.wrappers import PreproWrapper, MaxAndSkipEnv, ClippedRewardsWrapper, NoopResetEnv, EpisodicLifeEnv

from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN
from utils.dqn_wrappers import wrap_dqn
from configs.breakout import config

"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = wrap_dqn(env)
#     env = EpisodicLifeEnv(env)
#     env = NoopResetEnv(env)
# #    env = MaxAndSkipEnv(env, skip=config.skip_frame)
#     env = PreproWrapper(env, prepro=process_state, shape=(84, 84, 1), 
#                         overwrite_render=config.overwrite_render)
#    env = ClippedRewardsWrapper(env)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
