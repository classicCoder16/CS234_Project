import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule

from configs.q2_linear import config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################

        img_height, img_width, nchannels = state_shape
        print 'State shape is', state_shape
        self.s = tf.placeholder(tf.uint8, shape=(None, img_height, img_width, nchannels*self.config.state_history))
        if self.config.noise:
            print 'Adding noise placeholder'
            self.n = tf.placeholder(tf.uint8, shape=(None, img_height, img_width, nchannels*self.config.state_history))
        self.a = tf.placeholder(tf.int32, shape=(None))
        self.r = tf.placeholder(tf.float32, shape=(None))
        self.sp = tf.placeholder(tf.uint8, shape=(None, img_height, img_width, nchannels*self.config.state_history))
        self.done_mask = tf.placeholder(tf.bool, shape=(None))
        self.lr = tf.placeholder(tf.float32)

        ##############################################################
        ######################## END YOUR CODE #######################


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
        # out = state

        ##############################################################
        """
        TODO: implement a fully connected with no hidden layer (linear
            approximation) using tensorflow. In other words, if your state s
            has a flattened shape of n, and you have m actions, the result of 
            your computation sould be equal to
                W s where W is a matrix of shape m x n

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 
        
        flattened_state = tf.contrib.layers.flatten(state, scope)
        out = tf.contrib.layers.fully_connected(flattened_state, num_outputs=num_actions,\
                                                activation_fn=None, scope=scope, reuse=reuse)

        ##############################################################
        ######################## END YOUR CODE #######################

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: add an operator self.update_target_op that assigns variables
            from target_q_scope with the values of the corresponding var 
            in q_scope

        HINT: you may find the following functions useful:
            - tf.get_collection
            - tf.assign
            - tf.group

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        
        # Get collection of q_scope variables
        print 'Adding the update target op'
        q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)

        # The target q network only has the new weights attached, not the old output head
        if self.config.lwf:
            q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='new_' + q_scope)
            q_params = [param for param in q_params if 'old' not in param.name]
        
        # Get collection of target_network variables
        target_q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        # Run the assign operation on each corresponding pair
        assign_ops = []
        for param, target_param in zip(q_params, target_q_params):
            assign_ops.append(tf.assign(target_param, param))

        # Group updates together
        self.update_target_op = tf.group(*assign_ops)

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
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

        max_target_vals = self.config.gamma*tf.reduce_max(target_q, axis=1)
        not_done_mask = 1.0 - tf.cast(self.done_mask, tf.float32)
        q_samps = self.r + not_done_mask*max_target_vals
        q_sa = tf.reduce_sum(q*tf.one_hot(self.a, depth=num_actions), axis=1)
        self.loss = tf.reduce_mean((q_samps - q_sa)*(q_samps - q_sa))

        # If we're doing learning without forgetting
        if self.config.lwf:

            # The DQN component of the loss
            self.dqn_loss = self.loss

            # Try the Knowledge Distillation loss
            if self.config.lwf_loss == 'ce':

                # Convert to prob. distr.
                old_probs = tf.nn.softmax(self.out_old)
                curr_probs = tf.nn.softmax(self.out_old_pred)

                # Scale the probs by taking square root and normalizing
                scaled_old_probs = tf.sqrt(old_probs)
                scaled_old_probs /= tf.reduce_sum(scaled_old_probs, axis=1, keep_dims=True)

                scaled_curr_probs = tf.sqrt(curr_probs)
                scaled_curr_probs /= tf.reduce_sum(scaled_curr_probs, axis=1, keep_dims=True)

                self.lwf_loss = tf.reduce_mean(-tf.reduce_sum(scaled_old_probs * tf.log(scaled_curr_probs), reduction_indices=[1]))

            # Try the L-Inf loss instead
            elif self.config.lwf_loss == 'l-inf':
                diff = (self.out_old - self.out_old_pred)
                batch_losses = tf.reduce_max(tf.abs(diff), axis=1)
                self.lwf_loss = tf.reduce_mean(batch_losses)


            # Try the L-Inf loss instead
            elif self.config.lwf_loss == 'l1':
                diff = (self.out_old - self.out_old_pred)
                batch_losses = tf.reduce_sum(tf.abs(diff), axis=1)
                self.lwf_loss = tf.reduce_mean(batch_losses)

            # Default to L2 loss
            else: 
		print 'Here'
                diff = (self.out_old - self.out_old_pred)
                batch_losses = 0.5*tf.reduce_sum(diff**2, axis=1)
                self.lwf_loss = tf.reduce_mean(batch_losses)

                if self.config.num_penal > 1:
                    diff_2 = (self.out_old_inter - self.out_old_pred_inter)
                    batch_losses_2 = tf.reduce_sum(diff_2**2, axis=1)
                    self.lwf_loss += tf.reduce_mean(batch_losses_2)

                if self.config.num_penal > 2:
                    diff_3 = self.conv3_pred - self.out_old_conv3
                    batch_losses_3 = tf.reduce_mean(diff_3**2, axis=1)
                    self.lwf_loss += tf.reduce_mean(batch_losses_3)

            # Reassign the overall loss to the weighted sum of the two losses
            self.loss = self.dqn_loss + self.config.lwf_weight*self.lwf_loss

            # Used for adversarial update
            print 'N is', self.n
	    self.noise_grad = tf.gradients(self.lwf_loss, [self.the_noise])[0]
	    self.noise_grad = tf.Print(self.noise_grad, [self.noise_grad])

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functinos useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############

        # Define the optimizier
        adam_opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # Get the appropriate list of params
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        if self.config.feat_extract:
            if self.config.num_tuned == 2:
                params = tf.contrib.framework.get_variables('q/fully_connected/') + tf.contrib.framework.get_variables('q/fully_connected_1/')
            else:
                params = tf.contrib.framework.get_variables('q/fully_connected_1/')

        if self.config.lwf:
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='new_' + scope)

        # print 'Params being updated are now:', params

        # Get a list of grad, var pairs 
        grads_and_vars = adam_opt.compute_gradients(self.loss, var_list=params)

        # Split list into two lists
        grs, vrs = zip(*grads_and_vars)
        print 'Vrs is', vrs
        # If we're doing clipping, clip the gradients
        # Note: I believe it makes more sense to clip by global norm
        # than clippling each individually, since otherwise we more
        # drastically change the relative scale of parameters. :-)
        if self.config.grad_clip:
            grs, _ = tf.clip_by_global_norm(grs, self.config.clip_val)

        # Recombine grad and vars
        grads_and_vars = zip(grs, vrs)

        # Get the global norm
        self.grad_norm = tf.global_norm(grs)

        # Apply the gradients
        self.train_op = adam_opt.apply_gradients(grads_and_vars)
        
        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
