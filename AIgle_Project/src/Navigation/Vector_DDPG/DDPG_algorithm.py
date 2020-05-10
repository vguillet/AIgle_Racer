
##################################################################################################################
"""

"""

# Built-in/Generic Imports
import os
import sys
import time

# Libs
from tqdm import tqdm
import numpy as np


# Own modules
from AIgle_Project.Settings.SETTINGS import SETTINGS
from AIgle_Project.src.Navigation.Tools.ML_tools import ML_tools
from AIgle_Project.src.Navigation.Tools.RL_tools import RL_tools

from AIgle_Project.src.Navigation.Vector_DDPG.DDPG_agent import DDPG_agent

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '26/04/2020'

##################################################################################################################


class DDPG_vector_based_navigation:
    def __init__(self, client, memory_ref, actor_ref, critic_ref):
        # --> Initialise settings
        settings = SETTINGS()
        settings.rl_behavior_settings.gen_dql_settings()

        # --> Initialise tools
        ml_tools = ML_tools()
        rl_tools = RL_tools()



        # ----- Create agent
        agent = DDPG_agent(client, "1",
                           memory_type="simple",
                           memory_ref=memory_ref,
                           actor_ref=actor_ref,
                           critic_ref=critic_ref)

        # --> Episode rewards
        ep_rewards = []

        # --> Iterate over episodes
        for episode in tqdm(range(1, settings.rl_behavior_settings.episodes + 1), ascii=True, unit='episodes'):
            # ----- Reset
            # --> Reset episode reward and step number
            episode_reward = 0
            step = 1

            # --> Reset agent/environment
            agent.reset()

            # --> Get initial state
            current_state = agent.observation

            # --> Reset flag and start iterating until episode ends
            done = False

            # ----- Compute new episode parameters
            # TODO: Connect episode parameters
            learning_rate, discount, epsilon = rl_tools.get_episode_parameters(episode, settings)

            while not done:
                # --> Get a random value
                if np.random.random() > settings.rl_behavior_settings.epsilon:
                    # --> Get best action from main model
                    action = agent.get_qs()
                else:
                    # Get random action
                    action = np.random.randint(0, len(agent.action_lst))

                # --> Perform step using action
                new_state, reward, done = agent.step(action)

                state_t_pls_1, rwrd_t, done_t = temp[0], temp[1], temp[2]
                state_t_pls_1 = (state_t_pls_1 - self.observ_min) / (self.observ_max - self.observ_min)
                ri.append(rwrd_t)
                self.buffer.add_experience(
                    state_t.ravel(), action_t, rwrd_t, np.array(state_t_pls_1, self.dflt_dtype), done_t)

                state_t = np.array(state_t_pls_1, dtype=self.dflt_dtype).reshape(1, state_dim)
                if not rand:
                    if isinstance(self.buffer, Prioritized_experience_replay):
                        states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices = self.buffer.sample_batch()
                        self.train_networks(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch,
                                            indices)
                    else:
                        states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = self.buffer.sample_batch()
                        self.train_networks(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch,
                                            None)

                    self.actor_target = self.update_target(self.actor_target, self.actor, self.tow)
                    self.critic_target = self.update_target(self.critic_target, self.critic, self.tow)
                if done_t or t == self.T - 1:
                    rr = np.sum(ri)
                    self.r.append(rr)
                    if self.verbose: print('Episode %d : Total Reward = %f' % (episode, rr))
                    if self.plot:
                        plt.plot(self.r)
                        plt.pause(0.0001)
                    break
                if rand: experience_cnt += 1
                if experience_cnt > self.explore_time: rand = False

            if self.model_save_freq:
                if episode % self.model_save_freq == 0:
                    self.actor.save('actor_model.h5')
                    self.critic.save('critic_model.h5')
                    self.actor_target.save('actor_model.h5')
                    self.critic_target.save('critic_model.h5')
                    with open('buffer', 'wb') as file:
                        pickle.dump({'buffer': self.buffer}, file)
