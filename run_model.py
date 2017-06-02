import cv2
import gym
from utils.preprocess import greyscale, process_state
from utils.wrappers import PreproWrapper, MaxAndSkipEnv, ClippedRewardsWrapper, NoopResetEnv, EpisodicLifeEnv

from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN
from utils.dqn_wrappers import wrap_dqn
import tensorflow as tf
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(description='Trains a DQN model')
    parser.add_argument('-g', '--gpu', default=0, help="GPU number you would like to use")
    parser.add_argument('-m', '--model', default="Pong", help="Which model you would like to run")
    parser.add_argument('-v', '--version', default="-v0", help="Which model you would like to run, default is version 0")
    parser.add_argument('-e', '--experiment', default="trial1", help="Name of output folder")
    parser.add_argument('-r', '--restore', default=None, help="Whether you want to load a saved model")
    parser.add_argument('-rp', '--restorepath', default=None, help="Full result directory you want to restore from")
    parser.add_argument('-n', '--nsteps', default=None, help="How many training iterations to run for")
    parser.add_argument('-rc', '--record', default=None, help="Whether to record the results")
    parser.add_argument('-u', '--update_freq', default=None, help="Update frequency for target network")
    parser.add_argument('-nt', '--num_tuned', default=None, help="Number of the final fully connected layers to tune if restoring")
    parser.add_argument('-fe', '--feat_extract', default=None, help="Whether to do just feature extraction when restoring weights")
    parser.add_argument('-lwf', '--learning_without_forgetting', default=None, help="Whether to do learning without forgetting")
    parser.add_argument('-lwf_loss', '--lwf_loss', default='ce', help="What loss to use for learning without forgetting")
    parser.add_argument('-lwf_weight', '--lwf_weight', default='ce', help="What weight to use for learning without forgetting loss")
    args = parser.parse_args()
    return args


def modify_config(args):
    from configs.general import config
    print args
    config.env_name = args.model + args.version

    config.feat_extract = args.feat_extract

    if args.experiment:
        config.output_path = 'results/' + args.experiment + '/'
        config.model_output = config.output_path + "model.weights/"
        config.log_path     = config.output_path + "log.txt"
        config.plot_output  = config.output_path + "scores.png"
        config.record_path  = config.output_path + "monitor/"

    if args.restore is not None:
        config.restore = bool(args.restore)

    if args.restorepath is not None:
        config.restore_path = 'results/' + args.restorepath + '/model.weights'

    if args.nsteps is not None:
        config.nsteps_train = int(args.nsteps)
        config.lr_nsteps = (config.nsteps_train)/2

    if args.record is not None:
        config.record = bool(args.record)

    if args.update_freq is not None:
        config.target_update_freq = int(args.update_freq)

    if args.num_tuned is not None:
        config.num_tuned = int(args.num_tuned) 

    if args.lwf is not None:
        config.lwf = args.lwf

    if args.lwf_loss is not None:
        config.lwf_loss = args.lwf_loss

    if args.lwf_weight is not None:
        config.lwf_weight = args.lwf_weight

    return config

def print_config(config):
    print 'Current config:\n'
    variables = zip(vars(config).keys(), vars(config).values())
    for var, val in sorted(variables):
        print var + ' = ' + str(val)


if __name__ == '__main__':
    args = parse_args()
    my_config = modify_config(args)
    print_config(my_config)
    with tf.device('/gpu:' + str(args.gpu)):
        # make env
        env = gym.make(my_config.env_name)
        env = wrap_dqn(env)

        # exploration strategy
        exp_schedule = LinearExploration(env, my_config.eps_begin, 
                my_config.eps_end, my_config.eps_nsteps)

        # learning rate schedule
        lr_schedule  = LinearSchedule(my_config.lr_begin, my_config.lr_end,
                my_config.lr_nsteps)

        # train model
        model = NatureQN(env, my_config)
        model.run(exp_schedule, lr_schedule)
