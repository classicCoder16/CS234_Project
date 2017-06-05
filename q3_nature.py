import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, noise=None, reuse=False):
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
        print num_actions
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        with tf.variable_scope(scope, reuse=reuse):
            print 'n in here is', noise
            if self.config.lwf and self.config.noise and scope == 'q':
                conv1 = tf.contrib.layers.conv2d(noise, num_outputs=32, kernel_size=8, stride=4, padding='SAME')
            else:
                conv1 = tf.contrib.layers.conv2d(state, num_outputs=32, kernel_size=8, stride=4, padding='SAME')
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=64, kernel_size=4, stride=2, padding='SAME')
            conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=64, kernel_size=3, stride=1, padding='SAME')
            flattened_state = tf.contrib.layers.flatten(conv3)
            h1 = tf.contrib.layers.fully_connected(flattened_state, num_outputs=512)


            if scope == 'q' and self.config.lwf:
                self.out_old = tf.contrib.layers.fully_connected(h1, num_outputs=self.config.num_old_actions, activation_fn=None)
            else:
                out = tf.contrib.layers.fully_connected(h1, num_outputs=num_actions, activation_fn=None)

        # Just for evaluation purposes, we need to create a separate head for the transfer learning
        # models
        # if scope == 'q' and self.config.fine_tune and self.config.test_time:
        #     with tf.variable_scope('old_' + scope):
        #         if self.config.num_tuned == 2:
        #             h1_old = tf.contrib.layers.fully_connected(flattened_state, num_outputs=512)
        #             self.old_out_ft = tf.contrib.layers.fully_connected(h1_old, num_outputs=self.config.num_old_actions, activation_fn=None)
        #         else:
        #             self.old_out_ft = tf.contrib.layers.fully_connected(h1, num_outputs=self.config.num_old_actions, activation_fn=None)

        # If we are constructing the Q network during learning without forgetting,
        # we have the same architecture up to the last fully connected layers
        if scope == 'q' and self.config.lwf:
            with tf.variable_scope('new_' + scope):

                # Common architecture for both tasks
                conv1_new = tf.contrib.layers.conv2d(state, num_outputs=32, kernel_size=8, stride=4, padding='SAME')
                conv2_new = tf.contrib.layers.conv2d(conv1_new, num_outputs=64, kernel_size=4, stride=2, padding='SAME')
                conv3_new = tf.contrib.layers.conv2d(conv2_new, num_outputs=64, kernel_size=3, stride=1, padding='SAME')
                flattened_state_new = tf.contrib.layers.flatten(conv3_new)

                # If we make the last two fully connected layers different for a new action
                if self.config.num_tuned == 2:
                    # Old task branch of new network
                    if not self.config.noise:
                        h1_old = tf.contrib.layers.fully_connected(flattened_state_new, num_outputs=512, scope='old_fc')
                        self.out_old_pred = tf.contrib.layers.fully_connected(h1_old, num_outputs=self.config.num_old_actions, activation_fn=None, scope='old_fc_1')

                    # New task branch of new network
                    h1_new = tf.contrib.layers.fully_connected(flattened_state_new, num_outputs=512)
                    out = tf.contrib.layers.fully_connected(h1_new, num_outputs=num_actions, activation_fn=None)

                # Else we handle the case where the tasks diverge only at the last layer
                else:
                    h1_common = tf.contrib.layers.fully_connected(flattened_state_new, num_outputs=512)
                    if not self.config.noise:
                        self.out_old_pred = tf.contrib.layers.fully_connected(h1_common, num_outputs=self.config.num_old_actions, activation_fn=None, scope='old_fc_1')
                    out = tf.contrib.layers.fully_connected(h1_common, num_outputs=num_actions, activation_fn=None)
            
            if self.config.noise:
                with tf.variable_scope('new_' + scope, reuse=True):
                    conv1_old = tf.contrib.layers.conv2d(noise, num_outputs=32, kernel_size=8, stride=4, padding='SAME')
                    conv2_old = tf.contrib.layers.conv2d(conv1_old, num_outputs=64, kernel_size=4, stride=2, padding='SAME')
                    conv3_old = tf.contrib.layers.conv2d(conv2_old, num_outputs=64, kernel_size=3, stride=1, padding='SAME')
                    flattened_state_old = tf.contrib.layers.flatten(conv3_old)
                    if self.config.num_tuned == 1:
                        h1_old = tf.contrib.layers.fully_connected(flattened_state_old, num_outputs=512)

                with tf.variable_scope('new_' + scope):
                    if self.config.num_tuned == 2:
                        h1_old = tf.contrib.layers.fully_connected(flattened_state_old, num_outputs=512, scope='old_fc') 
                        self.out_old_pred = tf.contrib.layers.fully_connected(h1_old, num_outputs=self.config.num_old_actions, activation_fn=None, scope='old_fc_1')
                        print 'h1_old', h1_old
                    else:
                        self.out_old_pred = tf.contrib.layers.fully_connected(h1_old, num_outputs=self.config.num_old_actions, activation_fn=None, scope='old_fc_1') 
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
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
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
