import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q6_test import config


class MyDQN(Linear):
    """
    Going beyond - implement your own Deep Q Network to find the perfect
    balance between depth, complexity, number of parameters, etc.
    You can change the way the q-values are computed, the exploration
    strategy, or the learning rate schedule. You can also create your own
    wrapper of environment and transform your input to something that you
    think we'll help to solve the task. Ideally, your network would run faster
    than DeepMind's and achieve similar performance!

    You can also change the optimizer (by overriding the functions defined
    in TFLinear), or even change the sampling strategy from the replay buffer.

    If you prefer not to build on the current architecture, you're welcome to
    write your own code.

    You may also try more recent approaches, like double Q learning
    (see https://arxiv.org/pdf/1509.06461.pdf) or dueling networks 
    (see https://arxiv.org/abs/1511.06581), but this would be for extra
    extra bonus points.
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = tf.contrib.layers.conv2d(state, num_outputs=32, kernel_size=8, stride=4, padding='SAME')
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=64, kernel_size=4, stride=2, padding='SAME')
            conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=64, kernel_size=3, stride=1, padding='SAME')
            flattened_state = tf.contrib.layers.flatten(conv3)
            h1 = tf.contrib.layers.fully_connected(flattened_state, num_outputs=512)
            out = tf.contrib.layers.fully_connected(h1, num_outputs=num_actions, activation_fn=None)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out

    def add_loss_op(self, q, target_q, q2):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
              - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############
        best_q_actions = tf.argmax(q2, axis=1)
        best_mask = tf.one_hot(best_q_actions, depth=num_actions)
        best_target_q_vals = tf.reduce_sum(target_q*best_mask, axis=1)

        not_done_mask = 1.0 - tf.cast(self.done_mask, tf.float32)
        q_samps = self.r + self.config.gamma*not_done_mask*best_target_q_vals

        q_sa = tf.reduce_sum(q*tf.one_hot(self.a, depth=num_actions), axis=1)
        self.loss = tf.reduce_mean((q_samps - q_sa)*(q_samps - q_sa))

        ##############################################################
        ######################## END YOUR CODE #######################


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)
