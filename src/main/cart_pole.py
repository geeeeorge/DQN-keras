import gym
import numpy as np
from src.agents.memory import RandomMemory
from src.agents.policy import EpsGreedyQPolicy
import tensorflow as tf
from src.agents.model import SimpleNeuralNet
from src.agents import DQNAgent
import tqdm
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
actions = np.arange(nb_actions)
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.999, min_eps=.01)
memory = RandomMemory(limit=50000)
ini_observation = env.reset()
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()
model = SimpleNeuralNet(
    input_shape=[len(ini_observation)],
    nb_output=len(actions)
).model()
target_model = SimpleNeuralNet(
    input_shape=[len(ini_observation)],
    nb_output=len(actions)
).model()

agent = DQNAgent(actions=actions,
                 memory=memory,
                 update_interval=200,
                 train_interval=1,
                 batch_size=32,
                 observation=ini_observation,
                 model=model,
                 target_model=target_model,
                 policy=policy,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 is_ddqn=False)

step_history = []
nb_episodes = 10
with tqdm.trange(nb_episodes) as t:
    for episode in t:
        # agent.reset()
        observation = env.reset()
        agent.observe(observation)
        done = False
        step = 0
        episode_reward_history = []
        # train
        while not done:
            action = agent.get_action()
            observation, reward, done, info = env.step(action)
            step += 1
            episode_reward_history.append(reward)
            agent.observe(observation, reward, done)

            if done:
                t.set_description('Episode {}: {} steps'.format(episode, step))
                t.set_postfix(episode_reward=np.sum(episode_reward_history))
                # if episode > 3:
                agent.train()
                if episode % 5 == 0:
                    agent.update_target_hard()
                step_history.append(step)
                break

        # if last step is bigger than 195, stop the game.
        if all(np.array(step_history[-10:]) >= (env.spec.max_episode_steps - 5)):
            print('Problem is solved in {} episodes.'.format(episode))
            break

        agent.training = True

env.close()
x = np.arange(len(step_history))
plt.ylabel('step')
plt.xlabel('episode')
plt.plot(x, step_history)
plt.savefig('../../images/cart_pole_result.png')
