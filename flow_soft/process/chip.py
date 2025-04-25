import csv
import os.path

import filelock
import numpy as np

from filepaths import *
import pickle

from process.Signal import Signal
from utils import *


class Time_Helper:

    def __init__(self,generate=False):
        self.order = None  # player->[easy optimal hard的排列]
        self.chip = {}  # player->chip_time
        self.time_begin = {}

        if generate:
            print('=====[time helper]regenerate=====')
            self.read()
        else:
            print('=====[time helper]read from existing files=====')
            self.order = ReadPkl(File.order)
            self.chip = ReadPkl(File.chip)
            self.time_begin = ReadPkl(File.time_align)


    def read(self):
        for folder in os.listdir(Folder.rawValue):
            if not os.path.isdir(os.path.join(Folder.rawValue, folder)):
                continue
            player = playerIdx(folder)  # int
            filename = os.path.join(Folder.rawValue, folder, 'VR_Event.csv')

            with open(filename, 'r') as file:
                timestamps = []
                events = []
                reader = csv.DictReader(file)
                # 循环遍历每一行
                for row in reader:
                    # 按列存储数据
                    timestamps.append(float(row['Timestamp']))
                    events.append(row['event'])

                timestamps = np.array(timestamps)
                time_begin = timestamps[0]
                self.time_begin[player] = time_begin
                timestamps -= time_begin
                events = np.array(events)

            base_open = timestamps[np.array([x == 'begin BaseLineScene' for x in events])]
            assert len(base_open) == 3
            play_open = timestamps[np.array([str(x).startswith("play") and str(x).endswith("open") for x in events])]
            base_close = play_open
            assert len(base_close) == 3

            play_close = timestamps[np.array([str(x).startswith("play") and str(x).endswith("enter QuestionnaireScene")
                                              for x in events])]
            assert len(play_close) == 3

            rate_open = timestamps[np.array([x == 'QuestionnaireScene open' for x in events])]
            rate_close = [base_open[1], base_open[2], timestamps[-1]]
            # rate_close = event_signal.timestamp[event_signal.data.startswith("get 3D flow score")]
            assert len(rate_open) == 3 and len(rate_close) == 3

            chip = [[(base_open[i], base_close[i]), (play_open[i], play_close[i]), (rate_open[i], rate_close[i])]
                    for i in range(len(base_open))]


            self.chip[player] = chip

        print(self.chip.keys())
        WriteFile.pkl(self.chip,File.chip)
        WriteFile.pkl(self.time_begin,File.time_align)

    def chipData(self,player,signal):
        chip = self.chip[player]
        data = []
        for epoch_i in range(3):
            epoch_data = []
            for phase_i in range(3):
                begin = chip[epoch_i][phase_i][0]
                end = chip[epoch_i][phase_i][1]

                condition = np.logical_and(begin < signal.timestamp, signal.timestamp < end)
                slice_signal = Signal(item_name=signal.item_name,
                                      data=signal.data[np.where(condition)],
                                      timestamp=signal.timestamp[np.where(condition)])
                epoch_data.append(slice_signal)

            data.append(epoch_data)
        return data

    def players(self):
        return self.order.keys()
#Time_Helper(generate=True)
#Time_Helper(generate=False)