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
    parser = argparse.ArgumentParser(description='Evaluates a DQN model')
    parser.add_argument('-g', '--gpu', default=0, help="GPU number you would like to use")
    parser.add_argument('-m', '--model', default="Pong", help="Which game you would like to test")
    parser.add_argument('-v', '--version', default="-v0", help="Which version of the game you would like to test, default is version 0")
    parser.add_argument('-rp', '--restorepath', default=None, help="Full result directory you want to restore from")
    parser.add_argument('-rc', '--record', default=None, help="Whether to record the results")
    parser.add_argument('-nt', '--num_tuned', default=None, help="Number of the final fully connected layers to tune if restoring")
    parser.add_argument('-fe', '--feat_extract', default=None, help="Whether to do just feature extraction when restoring weights")
    parser.add_argument('-lwf', '--lwf', default=None, help="Whether to do learning without forgetting")
    parser.add_argument('-lwf_loss', '--lwf_loss', default='ce', help="What loss to use for learning without forgetting")
    parser.add_argument('-lwf_weight', '--lwf_weight', default=0.5, help="What weight to use for learning without forgetting loss")
    args = parser.parse_args()
    return args


def modify_config(args):
    from configs.general import config
    print args
    config.env_name = args.model + args.version

    config.feat_extract = args.feat_extract

    if args.restorepath:
        config.restore_path = 'results/' + args.restorepath + '/model.weights'
        config.output_path = 'test_results/' + args.restorepath + '/'
        config.model_output = config.output_path + "model.weights/"
        config.log_path     = config.output_path + "log.txt"
        config.plot_output  = config.output_path + "scores.png"
        config.record_path  = config.output_path + "monitor/"

    if args.record is not None:
        config.record = bool(args.record)

    if args.num_tuned is not None:
        config.num_tuned = int(args.num_tuned) 

    config.lwf = args.lwf

    config.lwf_loss = args.lwf_loss

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
        model = NatureQN(env, my_config)
        model.initialize_eval()
        model.evaluate()
        if my_config.record:
            model.record()
