from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint,TestLogger

from gym.utils.play import play
from random import choice
import matplotlib.pyplot as plt
import argparse

#We downsize the atari frame to 84 x 84 and feed the model 4 frames at a time for
#a sense of direction and speed.
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

#Standard Atari processing
class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

def plot_cumulative_rewards(data):
    plt.plot([i for i in range(len(data))],data)
    plt.xlabel("Step t")
    plt.ylabel("Rewards per episode")
    plt.show()

def build_model(INPUT_SHAPE,WINDOW_LENGTH):
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    if K.common.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.common.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model 

class ConstantAnnealedPolicy(LinearAnnealedPolicy):
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps,constant):
        super().__init__(inner_policy, attr, value_max, value_min, value_test, nb_steps)
        self.constant = constant

    def get_current_value(self):
        """
        returns constan epsilon value
        """
        if self.agent.training:
            value = self.constant
        else:
            value = self.value_test
        return value

class LogAnnealedPolicy(LinearAnnealedPolicy):
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        super().__init__(inner_policy, attr, value_max, value_min, value_test, nb_steps)

    def get_current_value(self):
        """Return current annealing value
        # Returns Value to use in annealing
        """
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.step) + b)
        else:
            value = self.value_test
        return value

class ExponentialAnnealedPolicy(LinearAnnealedPolicy):
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps,decay=0.993):
        super().__init__(inner_policy, attr, value_max, value_min, value_test, nb_steps)
        self.decay = decay 

    def get_current_value(self):
        """Return current annealing value
        # Returns Value to use in annealing
        """
        if self.agent.training:
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, self.decay*(self.agent.step)*a +b)
        else:
            value = self.value_test
        return value


def train_model(model,epsilon=0.9,policy = None,file = None):
    # Finally, we configure and compile our agent.
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 100 000 steps or we 
    # select the constant epsilon policy . The annealing epsilon is done so that 
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    if policy is None:
        policy = ConstantAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                    nb_steps=10000,constant=epsilon)

    if file is None and policy is None:
        return 
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                processor=processor, nb_steps_warmup=10000, gamma=.99, target_model_update=10000,
                train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    weights_filename = 'dqn_{}_weights.h5f'.format("pacman")
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    # log_filename = 'dqn_{}_exp_log.json'.format("pacman")
    # log_filename = 'dqn_{}_constant_long_log.json'.format("pacman")
    # log_filename = 'dqn_{}_cons_log.json'.format("pacman"+str(epsilon)[-1]+"_long")
    log_filename= file
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=2000000, log_interval=100000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    return dqn


def test_model(env):
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    processor = AtariProcessor()

    policy = policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1250000)

    callback = TestLogger()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                processor=processor, enable_double_dqn=False, enable_dueling_network=False, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                train_interval=4, delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    weights_filename = 'dqn_{}_weights.h5f'.format("pacman")
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=1, visualize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='MsPacmanDeterministic-v4')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    np.random.seed(231)
    env.seed(123)
    nb_actions = env.action_space.n
    print("NUMBER OF ACTIONS: " + str(nb_actions))
    
    epsilon_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    filename = 'dqn_{}_constant_long_log.json'.format("pacman")
    for epsilon in epsilon_list:
        model = build_model(INPUT_SHAPE,WINDOW_LENGTH)
        dqn = train_model(model,epsilon=epsilon,file=filename.format(str(epsilon)[2]))



    filename = 'dqn_{}_linear_long_log.json'.format("pacman")
    linear = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                            nb_steps=100000)
    model = build_model(INPUT_SHAPE,WINDOW_LENGTH)
    dqn = train_model(model,file=filename,policy= linear)


    filename = 'dqn_{}_exponential_long_log.json'.format("pacman")
    exponential = ExponentialAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                nb_steps=10000,decay=0.99993)
    model = build_model(INPUT_SHAPE,WINDOW_LENGTH)
    dqn = train_model(model,file=filename,policy = exponential)

    test_model(env)