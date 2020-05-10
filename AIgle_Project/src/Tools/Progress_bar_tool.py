import time
from math import modf


class Progress_bar:
    def __init__(self, max_step: int = None, bar_size: int = 30, label: str = None,
                 activity_indicator=True,
                 process_count=True,
                 progress_percent=True,
                 run_time=True,
                 average_run_time=True,
                 eta=True,
                 overwrite_setting=True,
                 bar_type: "Equal, Solid, Circle, Square" = "Equal",
                 activity_indicator_type="Pie stack",
                 rainbow_bar=False):

        # --> Error-proofing input:
        if type(max_step) is not int and max_step is not None:
            raise ValueError("Input for 'max_step' must be an integer or None")

        if type(bar_size) is not int or bar_size <= 0:
            raise ValueError("Input for 'bar_size' must be an integer bigger than 0")

        # --> Initiate Progress bar
        self.overwrite_setting = overwrite_setting
        self.bar_size = bar_size

        if max_step is not None:
            self.run_mode = 1
            self.max_step = max_step
            self.step = max_step / self.bar_size
            self.current = 0
        else:
            self.run_mode = 2
            self.max_step = 99999999999999999999
            self.step = 99999999999999999999
            self.current = 0

        self.print_activity_indicator = activity_indicator
        self.print_process_count = process_count
        self.print_progress_percent = progress_percent
        self.print_run_time = run_time
        self.print_average_run_time = average_run_time
        self.print_eta = eta

        # --> Determine bar properties based on input
        self.progress = True
        self.label = label
        self.current_label = None
        self.current_indicator_pos = 0
        self.colored_bar_lock = 0

        # --> Initiate time tracker
        self.initial_start_time = time.time()
        self.start_time = self.initial_start_time
        self.run_time = 0
        self.run_time_lst = []

        # ---- Colours and Formatting
        # --> Setting up bar formatting
        self.rainbow_bar = rainbow_bar
        self.bar_type = bar_type
        self.indicator_type = activity_indicator_type

        # --> Setting up format library
        self.bar_dict = {"Equal": {"Full": "=",
                                   "Empty": " "},
                         "Solid": {"Full": "█",
                                   "Empty": " "},
                         "Circle": {"Full": "◉",
                                    "Empty": "◯"},
                         "Square": {"Full": "▣",
                                    "Empty": "▢"}}

        self.indicator_dict = {"Bar spinner": ["-", "\\", "|", "/"],
                               "Dots": ["   ", ".  ", ".. ", "..."],
                               "Column": ['⡀', '⡄', '⡆', '⡇', '⣇', '⣧', '⣷', '⣿'],
                               "Pie spinner": ['◷', '◶', '◵', '◴'],
                               "Moon spinner": ['◑', '◒', '◐', '◓'],
                               "Stack": [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'],
                               "Pie stack": ['○', '◔', '◑', '◕', '●']}

        self.colours = {"reset": "\033[0m",
                        "bold": "\033[1m",
                        "italic": "\033[3m",
                        "underline": "\033[4m",
                        "green": "\033[32;1m",
                        "red": "\033[31;1m",
                        "magenta": "\033[35;1m",
                        "yellow": "\033[33;1m",
                        "cyan": "\033[36;1m",
                        "blue": "\033[34;1m"}

        if bar_type not in self.bar_dict.keys():
            raise ValueError("Selected bar type doesn't exist,"
                             "Bar type options: \nEqual  ( = ), \nSolid  ( █ ), \nCircle ( ◉ ), \nSquare ( ▣ )")

        if activity_indicator_type not in self.indicator_dict:
            raise ValueError("Selected activity indicator type doesn't exist,"
                             "Activity indicator type options: Bar spinner, Dots, Column, Pie spinner, Moon spinner, Stack, Pie stack")

    def update_progress(self, current=None, current_process_label=None):
        self.current_label = current_process_label
        if self.run_mode == 2:
            raise ValueError("'max_step' needs to be specified to compute progress")

        self.progress = True

        # --> Calc and record run time
        self.run_time = round(time.time() - self.start_time, 6)

        if current is not None:
            self.current = current + 1
        else:
            self.current += 1

        if self.overwrite_setting:
            print("\r" + self.__progress_bar, end="")
        else:
            print(self.__progress_bar)

        # --> Reset start time for next iteration
        self.start_time = time.time()

    def update_activity(self):
        if self.run_mode == 2:
            self.progress = True
            printed_bar = self.__activity_bar

        else:
            self.progress = False
            printed_bar = self.__progress_bar

        # --> Print bar
        if self.overwrite_setting:
            print("\r" + printed_bar, end="")
        else:
            print(printed_bar)

    # ===============================================================================
    # -------------------------- Loading bar properties -----------------------------
    # ===============================================================================
    @property
    def __progress_bar(self):
        return self.__activity_indicator * self.print_activity_indicator \
               + self.__label \
               + self.__process_count * self.print_process_count \
               + self.__bar \
               + self.__progress_percent * self.print_progress_percent \
               + "  " * (self.print_run_time or self.print_eta or self.print_average_run_time) \
               + self.__average_run_time * self.print_average_run_time \
               + self.__run_time * self.print_run_time \
               + self.__eta * self.print_eta \
               + self.__process_completed_msg

    @property
    def __activity_bar(self):
        return self.__activity_indicator * self.print_activity_indicator \
               + self.__label \
               + self.__process_count * self.print_process_count \
               + self.__average_run_time * self.print_average_run_time \
               + self.__run_time

    @property
    def __activity_indicator(self):
        if self.overwrite_setting is True and self.current != self.max_step:
            self.current_indicator_pos += 1
            if self.current_indicator_pos >= len(self.indicator_dict[self.indicator_type]):
                self.current_indicator_pos = 0
            return "[" + self.colours["cyan"] + self.indicator_dict[self.indicator_type][self.current_indicator_pos] + self.colours[
                "reset"] + "] "
        else:
            return ""

    @property
    def __label(self):
        label_str = ""
        if self.label is not None:
            if len(self.label) <= 6:
                label_str = self.label + " " * (6 - len(self.label)) + " | "
            else:
                label_str = self.label + " | "

        if self.current_label is not None:
            if len(self.label) <= 6:
                label_str = label_str + self.current_label + " " * (6 - len(self.current_label)) + " - "
            else:
                label_str = label_str + self.current_label + " - "

        return label_str

    @property
    def __process_count(self):
        if self.run_mode == 1:
            return self.__aligned_number(self.current, len(str(self.max_step))) + "/" + str(self.max_step)
        else:
            self.current += 1
            return str(self.current) + " iterations"

    @property
    def __bar(self):
        nb_of_steps = int(self.current / self.step)
        self.colored_bar_lock += 1

        if self.overwrite_setting is False or self.current == self.max_step:
            self.colored_bar_lock = -1

        # --> Prefix of bar
        bar = " - ["

        # ---- Create filled portion of bar
        if not self.rainbow_bar:
            # --> Define location of colored section
            if self.colored_bar_lock > nb_of_steps:
                self.colored_bar_lock = 0

            for step in range(nb_of_steps):
                if step == self.colored_bar_lock or step == self.colored_bar_lock - 1:
                    bar = bar + self.colours["cyan"] + self.bar_dict[self.bar_type]["Full"] + self.colours["reset"]
                else:
                    bar = bar + self.bar_dict[self.bar_type]["Full"]

        else:
            rainbow_lst = [self.colours["red"], self.colours["yellow"], self.colours["green"],
                           self.colours["cyan"], self.colours["blue"], self.colours["magenta"]]

            if self.colored_bar_lock >= len(rainbow_lst):
                self.colored_bar_lock = 0

            rainbow = self.colored_bar_lock

            size = 0
            # --> Create filled portion of bar
            for _ in range(nb_of_steps):
                bar = bar + rainbow_lst[rainbow] + self.bar_dict[self.bar_type]["Full"]
                size += 1
                if size > 1:
                    rainbow += 1
                    size = 0

                if rainbow >= len(rainbow_lst):
                    rainbow = 0

        bar = bar + self.colours["reset"] + ">"

        # --> Create empty portion of bar
        for _ in range(self.bar_size - nb_of_steps):
            bar = bar + self.bar_dict[self.bar_type]["Empty"]

        # --> Suffix of bar
        bar = bar + "]"

        return bar

    @property
    def __progress_percent(self):
        if round((self.current / self.max_step) * 100) == 100:
            return " - " + self.colours["green"] + \
                   self.__aligned_number(round((self.current / self.max_step) * 100), 2) + "%" + \
                   self.colours["reset"]
        return " - " + self.__aligned_number(round((self.current / self.max_step) * 100), 2) + "%"

    @property
    def __run_time(self):
        if self.run_mode == 1:
            if self.progress:
                # --> Save run time to runtime list
                self.run_time_lst.append(self.run_time)

            if self.current != self.max_step:
                # --> Create run time string
                run_time_str = self.__formatted_time(self.run_time)
            else:
                # --> Create total run time string
                total_run_time_str = self.__formatted_time(round(time.time() - self.initial_start_time, 4))
                return " - " + self.colours["bold"] + "Total run time: " + self.colours["reset"] + total_run_time_str

        else:
            # --> Calc run time
            self.run_time = round(time.time() - self.start_time, 4)

            # --> Reset start time for next iteration
            self.start_time = time.time()

            # --> Create run time string (including total run time)
            total_run_time_str = self.__formatted_time(round(time.time() - self.initial_start_time, 4))
            run_time_str = self.__formatted_time(self.run_time) + " - " + "Total run time: " + self.colours["reset"] + total_run_time_str

        if len(run_time_str) > 0:
            return " - " + self.colours["bold"] + "Run time: " + self.colours["reset"] + run_time_str
        else:
            return ""

    @property
    def __average_run_time(self):
        if len(self.run_time_lst) > 0:
            return " - " + self.colours["bold"] + "iter/s: " + str(
                self.__formatted_time(round(sum(self.run_time_lst) / len(self.run_time_lst), 4)))
        else:
            return ""

    @property
    def __eta(self):
        if len(self.run_time_lst) > 0:
            eta_str = self.__formatted_time(sum(self.run_time_lst) / len(self.run_time_lst) * (self.max_step - self.current))

            if len(eta_str) > 0:
                return " - " + self.colours["bold"] + "ETA: " + self.colours["reset"] + eta_str
            else:
                return ""

        else:
            return ""

    @property
    def __process_completed_msg(self):
        if self.current == self.max_step:
            return " - " + self.colours["green"] + "Process Completed" + self.colours["reset"]
        else:
            return ""

    # ===============================================================================
    # ----------------------- String formatting functions ---------------------------
    # ===============================================================================
    def __formatted_time(self, formatted_time):

        formatted_time = [0, formatted_time]

        time_dict_keys = ["seconds", "minutes", "hours", "days", "weeks", "months", "years", "decades", "centuries"]
        time_dict = {"seconds": {"max": 60,
                                 "current": 0,
                                 "str_count": 5,
                                 "str": ":"},

                     "minutes": {"max": 60,
                                 "current": 0,
                                 "str_count": 2,
                                 "str": ":"},

                     "hours": {"max": 24,
                               "current": 0,
                               "str_count": 2,
                               "str": ":"},

                     "days": {"max": 365,
                              "current": 0,
                              "str_count": 1,
                              "str": " days, "},

                     "weeks": {"max": 365,
                               "current": 0,
                               "str_count": 1,
                               "str": " weeks, "},

                     "months": {"max": 12,
                                "current": 0,
                                "str_count": 2,
                                "str": " months, "},

                     "years": {"max": 10,
                               "current": 0,
                               "str_count": 1,
                               "str": " years, "},

                     "decades": {"max": 10,
                                 "current": 0,
                                 "str_count": 2,
                                 "str": " decades, "},

                     "centuries": {"max": 99999999999999999999999,
                                   "current": 0,
                                   "str_count": 5,
                                   "str": " centuries, "}}

        # --> Fill time dict
        current_time_key = 0

        while formatted_time[1] / time_dict[time_dict_keys[current_time_key]]["max"] > 1:
            formatted_time = list(modf(formatted_time[1] / time_dict[time_dict_keys[current_time_key]]["max"]))
            if current_time_key == 0:
                time_dict[time_dict_keys[current_time_key]]["current"] = round(
                    formatted_time[0] * time_dict[time_dict_keys[current_time_key]]["max"], 3)
            else:
                time_dict[time_dict_keys[current_time_key]]["current"] = round(
                    formatted_time[0] * time_dict[time_dict_keys[current_time_key]]["max"])

            current_time_key += 1

        if current_time_key != 0:
            time_dict[time_dict_keys[current_time_key]]["current"] = round(formatted_time[1])
        else:
            time_dict[time_dict_keys[current_time_key]]["current"] = round(formatted_time[1] + formatted_time[0], 3)

        # --> Create time string
        time_str = ""
        for key in time_dict_keys:
            if time_dict[key]["current"] != 0:
                if time_dict[key]["current"] != 1:
                    time_str = self.__aligned_number(time_dict[key]["current"], time_dict[key]["str_count"], align_side="left") + \
                               time_dict[key]["str"] + time_str

        # --> Add s unit if time is only seconds
        if len(time_str) < 8:
            return time_str[:-2] + "s"
        else:
            return time_str[:-2]

    @staticmethod
    def __aligned_number(current, req_len, align_side="left"):
        current = str(current)

        while len(current) < req_len:
            if align_side == "left":
                current = "0" + current
            else:
                current = current + "0"
        return current


if __name__ == "__main__":
    import random

    maxi_step = 1000
    "Bar type options: Equal, Solid, Circle, Square"
    "Activity indicator type options: Bar spinner, Dots, Column, Pie spinner, Moon spinner, Stack, Pie stack"
    # bar = Progress_bar(max_step=maxi_step)

    bar = Progress_bar(max_step=maxi_step,
                       label="Demo bar",
                       process_count=True,
                       progress_percent=True,
                       run_time=True,
                       average_run_time=True,
                       eta=True,
                       overwrite_setting=True,
                       bar_type="Equal",
                       activity_indicator_type="Pie stack",
                       rainbow_bar=False)
    
    for i in range(maxi_step):
        for j in range(4):
            # bar.update_activity()
            time.sleep(0.02)
        bar.update_progress()
