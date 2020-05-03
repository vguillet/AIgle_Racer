
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys
import time

# Libs
import airsim
import cv2
import numpy as np
from tqdm import tqdm

# Own modules
from AIgle_Project.Settings.SETTINGS import SETTINGS
from AIgle_Project.src.Navigation.Tools.ML_tools import ML_tools
from AIgle_Project.src.Navigation.Tools.RL_tools import RL_tools
from AIgle_Project.src.Navigation.DQL.DQL_agent import DQL_agent
from AIgle_Project.src.Navigation.Models.DQL_models import DQL_models

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DQL_image_based_navigation:
    def __init__(self, client):
        # --> Initialise settings
        settings = SETTINGS()
        settings.rl_behavior_settings.gen_dql_settings()
        
        # --> Initialise tools
        ml_tools = ML_tools()
        rl_tools = RL_tools()
        
        # ----- Create agent
        agent = DQL_agent(client, "")

        # ----- Create models
        model_io_shape = (agent.rl_state.shape, len(agent.action_lst))

        print(model_io_shape)

        # --> Create main model
        model = DQL_models().model_1(model_io_shape[0],
                                     model_io_shape[1])

        # --> Create target network
        target_model = DQL_models().model_1(model_io_shape[0],
                                            model_io_shape[1])
        # --. Set target network weights equal to main model weights
        target_model.set_weights(model.get_weights())
        
        # ----- Create trackers
        # --> Used to count when to update target network with main network's weights
        target_update_counter = 0
        
        # --> Episode rewards
        ep_rewards = []

        # --> Iterate over episodes
        # for episode in tqdm(range(1, settings.rl_behavior_settings.episodes + 1), ascii=True, unit='episodes'):
        for episode in range(1, settings.rl_behavior_settings.episodes + 1):
            print("================> EPISODE", episode)
            # --> Update tensorboard step every episode
            # TODO: Fix tensorboard
            # agent.tensorboard.step = episode
    
            # ----- Reset
            # --> Reset episode reward and step number
            episode_reward = 0
            step = 1
    
            # --> Reset agent/environment
            agent.reset()
            
            # --> Get initial state
            current_state = agent.rl_state
            
            # --> Reset flag and start iterating until episode ends
            done = False
            
            # ----- Compute new episode parameters
            learning_rate, discount, epsilon = rl_tools.get_episode_parameters(episode, settings)

            while not done:
                # --> Get a random value
                if np.random.random() > settings.rl_behavior_settings.epsilon:
                    # --> Get best action from main model
                    action = np.argmax(agent.get_qs())
                else:
                    # Get random action
                    action = np.random.randint(0, len(agent.action_lst))

                new_state, reward, done = agent.step(action)

                # --> Transform new continuous state to new discrete state and count reward
                episode_reward += reward
                
                # TODO: Setup render environment
                # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                #     env.render()
    
                # Every step we update replay memory and train main network
                agent.remember(current_state, action, reward, new_state, done)
                target_update_counter = agent.train(done, target_update_counter)
    
                current_state = new_state
                step += 1

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
