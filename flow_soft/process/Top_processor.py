import csv
import os.path
import pickle
from collections import namedtuple, defaultdict
import numpy as np

from filepaths import *
from process.Signal import Signal
# feature 类导入
from process.myFeature.GSR_feature import *
from process.myFeature.HR_feature import *
from process.myFeature.EEG_feature import *
from process.myFeature.ET_feature import *
from utils import playerIdx
from .chip import *


class Controller:

    def __init__(self,
                 processed=False):
        self.time_helper = Time_Helper()
        # self.timestamp_chips = None 弃用
        self.processed = processed
        self.features = {player:[{},{},{}] for player in self.time_helper.order}  # player,epoch,phase item:data todo 似乎没有集中的必要

    def players(self):
        #return {1,2,3}
        return self.time_helper.players()
    def chip(self, player, signal):
        return self.time_helper.chipData(player, signal)

    def time_begin(self,player):
        return self.time_helper.time_begin[player]
    """
     def get_folder_list(self, top):
        folders = {}

        for first_level_folder in os.listdir(top):
            first_level_path = os.path.join(top, first_level_folder)
            if os.path.isdir(first_level_path):
                second_level_folders = os.listdir(first_level_path)
                if len(second_level_folders) == 1:
                    second_level_folder = second_level_folders[0]
                    second_level_path = os.path.join(first_level_path, second_level_folder)
                    folders[playerIdx(first_level_folder)] = second_level_path
                else:
                    folders[playerIdx(first_level_folder)] = first_level_path
        return folders
    def getFolder(self, player):
        if self.preprocessed:
            return self.processed_folders[player]
        else:
            return self.origin_folders[player]

    def getOrder(self,player):
        return self.chip_helper[player]
    """

    def writeFeature(self):
        def check(x):
            first_list_length = len(next(iter(x.values())))

            # 使用all()和生成器表达式检查所有列表的长度是否与第一个列表的长度相等
            all_lists_same_length = all(len(lst) == first_list_length for lst in x.values())

            assert all_lists_same_length
            return first_list_length

        assert self.features != {}

        WriteFile.pkl(self.features,filepath=File.features)

        for player in self.players():
            for epoch_i in range(3):
                features = self.features[player][epoch_i]
                difficulty = self.time_helper.order[player][epoch_i]
                folder1 = os.path.join(Folder.feature,str(player))
                folder2 = os.path.join(Folder.feature,str(player),difficulty)
                if not os.path.exists(folder1):
                    os.mkdir(folder1)
                    for _ in ['easy','optimal','hard']:
                        os.mkdir(os.path.join(folder1,_))

                if not os.path.exists(folder2):
                    os.mkdir(folder2)
                filepath = os.path.join(folder2, 'features.csv')
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    feature_item_list = list(features.keys())

                    n = check(features)

                    writer.writerow(['Timestamp'] + feature_item_list)
                    for i in range(n):
                        row = [features[key][i] for key in feature_item_list]
                        time = Feature.window * Feature.step * i
                        writer.writerow([time] + row)


class Top_processor:
    def __init__(self, controller: Controller,
                 player: int,
                 device: str,
                 filelist=None
                 ):

        # 基础参数
        self.father = controller
        self.player = player
        self.device = device  # 类名的简洁存在形式
        self.filelist = filelist  # 要读取的文件列表
        self.PRINT_HEADER = "【数据处理】"

        # 读取数据
        self.data = {}  # item_name -> Signal 二合一
        self.readData()  # 经过预处理后的 已切分好的data

        # 特征列表初始化
       # self.feature_extraction()
        # self.chip_data = {}  # {item_name,epoch:['easy'.'optimal','hard'],phase:[0,1,2]} 现已弃用

    def readData(self):
        if self.father.processed:
            # 直接从pkl中读取
            # data chip_data
            filename = str(self.player) + '-' + self.device + '.pkl'  # player device  比如1-EEG 2-GSR
            filepath = os.path.join(Folder.processed_pkl, filename)
            with open(filepath, 'rb') as file:
                self.data = pickle.load(file)
        else:
            # 从原始文件夹中读取
            self.readRawData()
            self.preprocess()
            self.write_preProcessed_data(csv=False)
            self.chip()
            # 将处理后的数据写入到pkl中
            self.writeData()

    def readRawData(self):
        # 从各个文件中读取数据
        for filename in self.filelist:
            filename = os.path.join(Folder.rawValue, str(self.player), filename)  # todo  需要做提前处理将文件夹名替换一下
            if not os.path.exists(filename):
                raise Exception('文件不存在:', filename)
            # 从文件中读取数据
            # 根据表头进行数据填充
            with open(filename, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                header = reader.fieldnames
                columns = {item_name: [] for item_name in header}  # item_name

                # 循环遍历每一行
                for row in reader:
                    # 按列存储数据
                    for item_name in row:

                        try:
                            columns[item_name].append(float(row[item_name]))
                        except Exception:
                            columns[item_name].append(str(row[item_name]))

                # 整理结构
                timestamp = np.array(columns['Timestamp'])
                if len(timestamp) == 0:
                    continue
                time_begin = self.father.time_begin(self.player)
                for item_name in header:
                    if len(columns[item_name]) == 0:
                        continue
                    if item_name != 'Time' and item_name != 'Timestamp':
                        self.data[item_name] = Signal(
                            item_name=item_name,
                            data=np.array(columns[item_name]),
                            timestamp=timestamp - time_begin
                        )
        self.my_print("=====读取原始数据结束,player=" + str(self.player) + "=====")

    def preprocess(self):
        """
        1 预处理
        2 基线处理（gsr放在预处理后  hr放在特征处理后
        -- 观察分布 --
        3 特征处理
        :return:
        """
        # 写入文件
        # 将预处理后的数据写回

        pass

    def writeData(self):
        """
        写入预处理完毕并切割好的数据
        """
        filename = str(self.player) + '-' + self.device + '.pkl'  # player device  比如1-EEG 2-GSR
        filepath = os.path.join(Folder.processed_pkl, filename)
        WriteFile.pkl(self.data, filepath)

    def chip(self):
        """
        数据拆分，拆分成三轮 每轮有静息-游戏-打分的三个部分
        按timestamp进行切分
        :return:
        """
        for item_name in self.data:
            intact_signal = self.data[item_name]
            self.data[item_name] = self.father.chip(self.player, intact_signal)



    def write_preProcessed_data(self,csv=False):
        # write pkl
        filename = str(self.player) + '-' + self.device + '-intact' + '.pkl'  # player device  比如1-EEG-intact 2-GSR
        filepath = os.path.join(Folder.processed_pkl, filename)
        WriteFile.pkl(self.data, filepath)

        # write csv
        if csv:
            for item_name in self.data:
                signal = self.data[item_name]
                filepath = os.path.join(Folder.processed_csv, str(self.player), item_name + '.csv')
                WriteFile.csv(filepath, header=['Timestamp', item_name],
                              cols=[signal.timestamp, signal.data],
                              length=signal.len())

    def feature_extraction(self,stat_item_list=None):
        if stat_item_list is not None:
            for item in stat_item_list:
                self.calc_feature1(Feature_cls_name='Feature_Statistic',item_name=item)
        self.calc_feature1()  # 对单个signal进行特征提取
        self.calc_feature2()  # 多个signal需要联合进行特征提取



    def calc_feature1(self, Feature_cls_name=None, item_name=None):
        if item_name is None:
            return
        """
        对于单个signal进行特征提取
        """
        #需要处理一些个数匹配问题 todo
        cls = globals().get(Feature_cls_name)
        base_features = []

        cnt = 0
        for epoch_i in range(3):
            base_signal = self.data[item_name][epoch_i][0].slice(window=30)
            _ = cls(base_signal)
            try:
                _.calc(cut=False)
                base_features.append(_.features)
                cnt += 1
            except Exception:
                pass

        assert cnt > 0
        base_feature = {key: sum(d[key] for d in base_features) / cnt for key in base_features[0]}

        # 用于训练模型的特征输出
        for epoch_i in range(3):
            play_signal = self.data[item_name][epoch_i][1].slice(window=150)
            _ = cls(play_signal)
            _.calc(cut=True, base_feature=base_feature)

            # 插一道检查 todo
            self.father.features[self.player][epoch_i].update(_.features)

    def calc_feature2(self, Feature_cls_name=None, item_name_list=None):
        """
        多个signal进行联合特征提取，要求
        多个signal之间必须维度相同

        """
        if self.player == 1:
            x = 0
        # check dim
        if item_name_list is None:
            return
        cls = globals().get(Feature_cls_name)

        base_features = []
        cnt = 0
        for epoch_i in range(3):
            base_signal = {item_name: self.data[item_name][epoch_i][0].slice(window=30)
                           for item_name in item_name_list}
            _ = cls(base_signal)
            try:
                _.calc(cut=False)
                base_features.append(_.features)
                cnt += 1
            except Exception:
                pass

        assert cnt > 0
        base_feature = {key: sum(d[key] for d in base_features) / cnt for key in base_features[0]}

        # 用于训练模型的特征输出
        for epoch_i in range(3):
            play_signal = {item_name: self.data[item_name][epoch_i][1].slice(window=150)
                           for item_name in item_name_list}
            _ = cls(play_signal)
            _.calc(cut=True, base_feature=base_feature)
            self.father.features[self.player][epoch_i].update(_.features)



    def my_print(self, content):
        print(self.PRINT_HEADER, content)
