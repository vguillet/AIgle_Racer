##################################################################################################################
"""
This class contains the results_gen class, used to generate results form the various runs
"""

# Built-in/Generic Imports
import datetime
import time
import os

# Own modules
from AIgle_Project.Settings.SETTINGS import SETTINGS
from AIgle_Project.src.Navigation.Tools.math_tools import math_tools

__version__ = '1.1.1'
__author__ = 'Victor Guillet'
__date__ = '10/09/2019'


##################################################################################################################


class RL_results:
    def __init__(self, settings):

        # ---- Fetch EVOA settings
        self.settings = settings
        self.run_name = self.settings.rl_behavior_settings.run_name

        # --> Run trackers
        self.run_start_time = datetime.datetime.now()
        self.run_stop_time = None

        # --> Epoque trackers
        self.epoque_reward = []
        self.avg_reward_per_batch = []
        self.best_individual_reward_per_batch = []

        self.epoque_tau = []
        self.epoque_discount = []
        self.epoque_epsilon = []

    def gen_stats(self):
        # ------------ Generating further informations
        # -------- Determine best fit lines
        # --> For average fitness
        a_avg_r, b_avg_r = math_tools().best_fit(range(len(self.avg_reward_per_batch)), self.avg_reward_per_batch)
        self.gradient_bestfit_avg_r = b_avg_r
        self.yfit_avg_r = [a_avg_r + b_avg_r * xi for xi in range(len(self.avg_reward_per_batch))]

        # --> For best fitness individual
        a_best_r, b_best_r = math_tools().best_fit(range(len(self.best_individual_reward_per_batch)),
                                                   self.best_individual_reward_per_batch)
        self.gradient_bestfit_best_r = b_best_r
        self.yfit_best_r = [a_best_r + b_best_r * xi for xi in range(len(self.avg_reward_per_batch))]

    def gen_result_recap_file(self):
        # -- Create results file
        path = r"AIgle_Project/src/Navigation/Saved_models/Vector_ddql" \
               + '/' + self.settings.rl_behavior_settings.training_type \
               + "/" + self.run_name

        full_file_name = path + "/" + "Run_summary.txt"

        if not os.path.exists(path):
            os.makedirs(path)

        self.results_file = open(full_file_name, "w+")

        self.results_file.write("====================== " + self.run_name + " ======================\n")
        self.results_file.write("\n~~~~~~~~~~~ Run configuration recap: ~~~~~~~~~~~\n")

        self.results_file.write("Run name: " + self.settings.rl_behavior_settings.run_name)

        self.results_file.write("\n\n-----------> RL main parameters:" + "\n")
        self.results_file.write("Algorithm type: " + self.settings.rl_behavior_settings.algorithm)
        self.results_file.write("Epoque count: " + str(self.settings.rl_behavior_settings.epoques))

        if self.settings.rl_behavior_settings.model_ref is None:
            self.results_file.write("\nStarting model ref: None")

        else:
            self.results_file.write("\nStarting model ref:" + self.settings.rl_behavior_settings.model_ref)

        self.results_file.write("\n\n-----------> Replay memory parameters:" + "\n")
        self.results_file.write("Replay memory type:" + self.settings.rl_behavior_settings.memory_type)
        self.results_file.write("Replay memory size:" + str(self.settings.rl_behavior_settings.memory_size))
        self.results_file.write("Min replay memory type:" + str(self.settings.rl_behavior_settings.min_replay_memory_size))

        if self.settings.rl_behavior_settings.memory_ref is None:
            self.results_file.write("\nReplay memory ref: None")

        else:
            self.results_file.write("\nReplay memory ref:" + self.settings.rl_behavior_settings.memory_ref)

        self.results_file.write("\n\n-----------> Learning parameters:" + "\n")
        self.results_file.write("Minibatch size =" + str(self.settings.rl_behavior_settings.minibatch_size))
        self.results_file.write("Discount =" + str(self.settings.rl_behavior_settings.discount))

        self.results_file.write("\nTau =" + str(self.settings.rl_behavior_settings.tau))

        if self.settings.rl_behavior_settings.hard_update_target_every is None:
            self.results_file.write("Hard update weights: Disabled")

        else:
            self.results_file.write("Hard update weights every:" + str(self.settings.rl_behavior_settings.hard_update_target_every))

        self.results_file.write("\n\n-----------> Exploration settings:" + "\n")
        self.results_file.write("Epsilon =" + str(self.settings.rl_behavior_settings.epsilon))
        self.results_file.write("Random starting position =" + str(self.settings.agent_settings.random_starting_point))

        self.results_file.write("\n\n-----------> Decay settings:" + "\n")
        self.results_file.write(
            "Tau decay function: "
            + self.settings.rl_behavior_settings.decay_functions[self.settings.rl_behavior_settings.tau_decay])

        self.results_file.write(
            "Discount decay function: "
            + self.settings.rl_behavior_settings.decay_functions[self.settings.rl_behavior_settings.discount_decay])

        self.results_file.write(
            "Epsilon decay function: "
            + self.settings.rl_behavior_settings.decay_functions[self.settings.rl_behavior_settings.epsilon_decay])

        self.results_file.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        self.results_file.write("-----------> Run stats: \n")
        # self.results_file.write("Start time:" + self.run_start_time.strftime('%X %x %Z') + "\n")
        self.results_file.write("End time: " + time.strftime('%X %x %Z') + "\n")
        self.results_file.write("Run time: " + str(self.run_stop_time - self.run_start_time) + "s\n")

        self.results_file.write("\nAverage computing time per epoque: "
                                + str(
            (self.run_stop_time - self.run_start_time) / self.settings.rl_behavior_settings.epoques) + "s\n")

        self.results_file.write("\n\n-----------> Rewards results:" + "\n")
        self.results_file.write("Max average reward achieved: " + str(max(self.avg_reward_per_batch)) + "\n")
        self.results_file.write(
            "Max individual reward achieved: " + str(max(self.best_individual_reward_per_batch)) + "\n")

        self.results_file.write(
            "\nAverage reward best fit line gradient achieved: " + str(self.gradient_bestfit_avg_r) + "\n")
        self.results_file.write(
            "Individual reward best fit line gradient achieved: " + str(self.gradient_bestfit_best_r) + "\n")

        self.results_file.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        self.results_file.write("-----------> Validation benchmark results: \n")

        self.results_file.write(str() + "\n")

        print("\n-- EVOA run results summary successfully generated -- \n")
        self.results_file.close()
        return

    def plot_results(self):
        import matplotlib.pyplot as plt

        if len(self.avg_reward_per_batch) > 1:
            plt.plot(self.avg_reward_per_batch, label="Avg individual reward")
            plt.plot(self.best_individual_reward_per_batch, label="Best individual reward")

            if len(self.avg_reward_per_batch) > 2:
                self.gen_stats()
                plt.plot(self.yfit_avg_r, label="Avg individual reward trendline")
                plt.plot(self.yfit_best_r, label="Best individual reward trendline")

            plt.xlabel("Epoques (batch size = " +
                       str(self.settings.rl_behavior_settings.batch_epoque_size) + ")")
            plt.ylabel("Cumulated reward")
            plt.grid()
            plt.legend()
            plt.show()

        # plt.plot(ep_tau)
        # plt.xlabel("Epoques")
        # plt.ylabel("Tau")
        # plt.grid()
        # plt.show()
        #
        # plt.plot(ep_discount)
        # plt.xlabel("Epoques")
        # plt.ylabel("Discount")
        # plt.grid()
        # plt.show()
        #
        # plt.plot(ep_epsilon)
        # plt.xlabel("Epoques")
        # plt.ylabel("Epsilon")
        # plt.grid()
        # plt.show()




        # # --> Fitness plot
        # plt.plot(range(len(self.avg_reward_per_batch)), self.avg_reward_per_batch, label="Average fitness per gen")
        # plt.plot(range(len(self.avg_reward_per_batch)), self.yfit_avg_f, "k", dashes=[6, 2])
        #
        # plt.plot(range(len(self.best_individual_reward_per_batch)), self.best_individual_reward_per_batch,
        #          label="Best individual fitness per gen")
        # plt.plot(range(len(self.best_individual_reward_per_batch)), self.yfit_best_f, "k", dashes=[6, 2])
        #
        # plt.plot([
        #              self.settings.signal_training_settings.nb_of_generations - self.settings.signal_training_settings.exploitation_phase_len - self.invalid_slice_count,
        #              self.settings.signal_training_settings.nb_of_generations - self.settings.signal_training_settings.exploitation_phase_len - self.invalid_slice_count],
        #          [0, 100], label="End of exploration phase")
        #
        # plt.title("Fitness per gen; " + self.ticker + self.run_name)
        # plt.ylabel("Fitness %")
        # plt.xlabel("Generation #")
        # plt.legend()
        # plt.grid()
        #
        # plt.show()
        #
        # # --> Net Worth plot
        # plt.plot(range(len(self.avg_net_worth_per_gen)), self.avg_net_worth_per_gen, label="Average net worth per gen")
        # plt.plot(range(len(self.avg_net_worth_per_gen)), self.yfit_avg_nw, "k", dashes=[6, 2])
        #
        # plt.plot(range(len(self.best_individual_net_worth_per_gen)), self.best_individual_net_worth_per_gen,
        #          label="Best individual net worth per gen")
        # plt.plot(range(len(self.best_individual_net_worth_per_gen)), self.yfit_best_nw, "k", dashes=[6, 2])
        #
        # plt.plot(range(len(self.data_slice_metalabel_pp)), self.data_slice_metalabel_pp,
        #          label="Metalabel net worth per gen")
        # plt.plot([
        #              self.settings.signal_training_settings.nb_of_generations - self.settings.signal_training_settings.exploitation_phase_len - self.invalid_slice_count,
        #              self.settings.signal_training_settings.nb_of_generations - self.settings.signal_training_settings.exploitation_phase_len - self.invalid_slice_count],
        #          [min(self.avg_net_worth_per_gen), max(self.best_individual_net_worth_per_gen)],
        #          label="End of exploration phase")
        #
        # plt.title("Profit per gen; " + self.ticker + self.run_name)
        # plt.ylabel("Net worth $")
        # plt.xlabel("Generation #")
        # plt.legend()
        # plt.grid()
        #
        # plt.show()
        #
        # plt.plot(range(len(self.nb_parents)), self.nb_parents, label="Number of parents selected per gen")
        # plt.plot(range(len(self.nb_random_ind)), self.nb_random_ind,
        #          label="Number of random individuals selected per gen")
        #
        # plt.title("Number of individuals types per gen; " + self.ticker + self.run_name)
        # plt.ylabel("Number of individuals")
        # plt.xlabel("Generation #")
        # plt.legend()
        # plt.grid()

        plt.show()
