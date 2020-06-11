
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import random
import datetime

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
from AIgle_Project.src.Navigation.Tools.RL_results import RL_results

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

        # --> Seed numpy
        random.seed(10)

        # ---- Create agent
        # settings.rl_behavior_settings.gen_ddql_settings()
        # agent = Vector_DDQL_agent(client, "1")

        settings.rl_behavior_settings.gen_ddpg_settings()
        agent = Vector_DDPG_agent(client, "1")

        # --> Initialise tools
        ml_tools = ML_tools()
        rl_tools = RL_tools()
        results = RL_results(settings)

        # ---- Create trackers
        # --> epoque batches
        current_ep_batch_reward = []
        batch_counter = 0

        total_batch_steps = 0
        batch_random_steps = 0

        # --> Create epoque progress bar
        epoque_bar = Progress_bar(max_step=settings.rl_behavior_settings.epoques,
                                  overwrite_setting=False,
                                  label="Epoques")

        client.simFlushPersistentMarkers()

        # ======================== PROCESS ==============================================
        if settings.rl_behavior_settings.run_mode == 0:
            # --> Iterate over epoques
            for epoque in range(1, settings.rl_behavior_settings.epoques + 1):
                epoque_bar.update_progress()

                print("\n\n================================================================================ Epoque", epoque)
                batch_counter += 1

                # ---- Reset
                # --> Reset epoque trackers
                epoque_reward = 0

                total_epoque_steps = 0
                epoque_optimal_steps = 0
                epoque_random_steps = 0

                # --> Reset agent/environment
                agent.reset(settings.agent_settings.random_starting_point,
                            settings.agent_settings.random_flip_track)

                # --> Get initial state
                current_state = agent.observation

                print("\n--> Starting epoque")
                print("- Starting state:", current_state)

                # --> Reset flag and start iterating until epoque ends
                done = False

                # ---- Compute new epoque parameters
                tau, discount, epsilon = rl_tools.get_epoque_parameters(epoque, settings)

                print("- epoque tau:", tau)
                print("- epoque discount:", discount)
                print("- epoque epsilon:", epsilon)

                # --> Record epoque parameters
                results.epoque_tau.append(tau)
                results.epoque_discount.append(discount)
                results.epoque_epsilon.append(epsilon)

                # --> Create step progress bar
                # print("\n")
                # step_bar = Progress_bar(max_step=settings.agent_settings.max_step,
                #                         # overwrite_setting=False,
                #                         bar_size=10,
                #                         label="Steps")
                while not done:

                    total_epoque_steps += 1
                    total_batch_steps += 1

                    # --> Get a random value
                    if random.uniform(0, 100) > epsilon:
                        # --> Get best action from main model
                        action = np.argmax(agent.get_qs())

                        epoque_optimal_steps += 1

                    else:
                        # --> Get random action
                        action = np.random.randint(0, len(agent.action_lst))

                        epoque_random_steps += 1
                        batch_random_steps += 1

                    # --> Perform step using action
                    new_state, reward, done = agent.step(action)

                    # --> Count reward
                    epoque_reward += reward

                    # TODO: Setup render environment
                    # if SHOW_PREVIEW and not epoque % AGGREGATE_STATS_EVERY:
                    #     env.render()

                    # --> Add memory to replay memory
                    agent.remember(current_state, action, reward, new_state, done)

                    # --> Update update replay memory and train models
                    client.simPause(True)
                    agent.train(discount, tau)
                    client.simPause(False)

                    # --> Set current state as new state
                    current_state = new_state

                    # --> Clean up trace marks
                    if settings.rl_behavior_settings.show_tracelines == "step":
                        client.simFlushPersistentMarkers()

                    # if not done:
                    #     step_bar.update_progress()

                print("\n\n--> Epoque complete")
                print("- Total nb. steps taken:", total_epoque_steps)
                print("- Nb. optimal steps taken:", epoque_optimal_steps)
                print("- Nb. random steps taken:", epoque_random_steps)

                print("\nTrack direction:", agent.reward_function.direction)
                print("Max goal reached:", agent.goal_tracker - 1)
                print("Reward achieved:", round(epoque_reward, 3))

                print("\n")

                # --> Record epoque results
                results.epoque_reward.append(epoque_reward)
                current_ep_batch_reward.append(epoque_reward)

                # --> Clean up trace marks
                if settings.rl_behavior_settings.show_tracelines == "individual":
                    client.simFlushPersistentMarkers()

                # ===========================================================================
                # ----- Display batch results
                if batch_counter == settings.rl_behavior_settings.batch_epoque_size:
                    # --> Clean up trace marks
                    if settings.rl_behavior_settings.show_tracelines == "batch":
                        client.simFlushPersistentMarkers()

                    # --> Calculate last epoque batch average and add to ep_bacth_reward
                    avg_reward = round(sum(current_ep_batch_reward)/len(current_ep_batch_reward), 3)

                    results.avg_reward_per_batch.append(avg_reward)
                    results.best_individual_reward_per_batch.append(max(current_ep_batch_reward))

                    # --> Plot batch diagrams
                    if settings.rl_behavior_settings.plot_epoque_batch_reward:
                        results.plot_results()

                    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print("- Batch results:")
                    print("Batch average reward:", avg_reward)
                    print("Batch max reward:", round(max(current_ep_batch_reward), 3))

                    print("Total nb. steps taken:", total_batch_steps)
                    print("Nb. random steps taken:", batch_random_steps)

                    # --> Save models
                    if settings.rl_behavior_settings.save_model_on_batch:
                        if isinstance(agent, Vector_DDQL_agent):
                            agent.model.save_checkpoint(epoque)
                        elif isinstance(agent, Vector_DDPG_agent):
                            agent.actor_model.save_checkpoint(epoque)
                            agent.critic_model.save_checkpoint(epoque)

                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

                    # --> Reset batch trackers
                    current_ep_batch_reward = []
                    batch_counter = 0

                    random_steps = 0
                    steps = 0

            # ======================== RESULTS ==============================================
            results.run_stop_time = datetime.datetime.now()

            print("\n\n--RL optimisation process complete --")
            results.gen_result_recap_file()

        # ======================== Testing agent ==============================================
        elif settings.rl_behavior_settings.run_mode == 1:
            agent.reset(False, False)

            for i in range(len(agent.reward_function.goal_dict)):
                print("Current goal:", i)
                agent.goal_tracker = i

                while agent.reward_function.get_distance_from_goal(agent.observation) > 2:
                    print("Step")
                    action = np.argmax(agent.get_qs())
                    _, _, _ = agent.step(action)

            # if epoque % 100 == 0:
            # --> Record networks
            # agent.actor_model.save_checkpoint(str(epoque))
            # agent.critic_model.save_checkpoint(str(epoque))

            # --> Record replay memory
            # agent.memory.save_replay_memory(str(epoque))

            # print("\n", epoque_reward)
            # Append epoque reward to a list and log stats (every given number of epoques)
            # ep_rewards.append(epoque_reward)
            # if not epoque % AGGREGATE_STATS_EVERY or epoque == 1:
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
