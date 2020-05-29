
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import random

# Libs
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Own modules
from AIgle_Project.Settings.SETTINGS import SETTINGS
from AIgle_Project.src.Tools.Progress_bar_tool import Progress_bar
from AIgle_Project.src.Navigation.Tools.ML_tools import ML_tools
from AIgle_Project.src.Navigation.Tools.RL_tools import RL_tools

from AIgle_Project.src.Navigation.Agents.Image_DQL_agent import Image_DQL_agent
from AIgle_Project.src.Navigation.Agents.Vector_DDQL_agent import Vector_DDQL_agent
from AIgle_Project.src.Navigation.Agents.Vector_DDPG_agent import Vector_DDPG_agent

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class RL_navigation:
    def __init__(self, client):
        # --> Set graphics card for Deep Learning
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        # --> Initialise settings
        settings = SETTINGS()
        settings.agent_settings.gen_agent_settings()
        settings.rl_behavior_settings.gen_dql_settings()
        # settings.rl_behavior_settings.gen_ddpg_settings()

        # --> Initialise tools
        ml_tools = ML_tools()
        rl_tools = RL_tools()

        # --> Seed numpy
        random.seed(10)

        # ---- Create agent
        agent = Vector_DDQL_agent(client, "1",
                                  memory_type=settings.rl_behavior_settings.memory_type,
                                  memory_ref=settings.rl_behavior_settings.memory_ref,
                                  model_ref=settings.rl_behavior_settings.model_ref)

        # agent = DDPG_agent(client, "1",
        #                    memory_type=settings.rl_behavior_settings.memory_type,
        #                    memory_ref=settings.rl_behavior_settings.memory_ref,
        #                    actor_ref=settings.rl_behavior_settings.actor_ref,
        #                    critic_ref=settings.rl_behavior_settings.critic_ref)

        # ---- Create trackers
        # --> Episode rewards
        ep_rewards = []

        episode_bar = Progress_bar(max_step=settings.rl_behavior_settings.episodes,
                                   overwrite_setting=False,
                                   label="Episodes")

        # --> Iterate over episodes
        for episode in range(1, settings.rl_behavior_settings.episodes + 1):
            print("\n=================== Episode", episode)
            episode_bar.update_progress()

            # TODO: Fix tensorboard
            # agent.tensorboard.step = episode

            # ---- Reset
            # --> Reset episode reward and step number
            episode_reward = 0

            # --> Reset agent/environment
            agent.reset()

            # --> Get initial state
            current_state = agent.observation

            # --> Reset flag and start iterating until episode ends
            done = False

            # ---- Compute new episode parameters
            # TODO: Connect episode parameters
            learning_rate, discount, epsilon = rl_tools.get_episode_parameters(episode, settings)

            step_bar = Progress_bar(max_step=settings.agent_settings.max_step,
                                    # overwrite_setting=False,
                                    label="Steps")
            while not done:
                # --> Get a random value
                if random.randint(0, 100) > settings.rl_behavior_settings.epsilon:
                # if np.random.randint(0, 100) > 0:

                    # --> Get best action from main model
                    action = np.argmax(agent.get_qs())

                else:
                    # Get random action
                    action = np.random.randint(0, len(agent.action_lst))

                # --> Perform step using action
                new_state, reward, done = agent.step(action)

                # --> Count reward
                episode_reward += reward

                # TODO: Setup render environment
                # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                #     env.render()

                # --> Add memory to replay memory
                agent.remember(current_state, action, reward, new_state, done)

                # --> Update update replay memory and train models
                client.simPause(True)
                agent.train()
                client.simPause(False)

                # --> Set current state as new state
                current_state = new_state

                if not done:
                    step_bar.update_progress()

            ep_rewards.append(episode_reward)

            print("\n--> Episode complete")

            # TODO: add checkpoint rate in settings
            if episode % 10 == 0:
                plt.plot(ep_rewards)
                plt.grid()
                plt.show()

            # if episode % 100 == 0:
            # --> Record networks
            # agent.actor_model.save_checkpoint(str(episode))
            # agent.critic_model.save_checkpoint(str(episode))

            # --> Record replay memory
            # agent.memory.save_replay_memory(str(episode))

            # print("\n", episode_reward)
            # Append episode reward to a list and log stats (every given number of episodes)
            # ep_rewards.append(episode_reward)
            # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #     agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
            #                                    epsilon=epsilon)
            #
            #     # Save model, but only when min reward is greater or equal a set value
            #     if min_reward >= MIN_REWARD:
            #         agent.model.save(
            #             f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
