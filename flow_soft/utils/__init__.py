import csv
import os
import pickle
import time


def playerIdx(s: str):
    # todo  改用正则表达式
    """
    s:����player11
    return : ���(int)
    """
    try:
        idx = int(s)
        return idx
    except Exception:
        # print(s)
        l = s.split('_')
        idx = int(l[1])
        return idx


def getOrder(latinIdx):
    if latinIdx == 0:
        return ['easy', 'optimal', 'hard']
    if latinIdx == 1:
        return ['optimal', 'hard', 'easy']
    if latinIdx == 2:
        return ['hard', 'easy', 'optimal']


def getLatinIdx(order: list):
    return 2 - order.index('hard')


def latinIdx2EpochIdx(latinIdx):
    if latinIdx == 0:
        return 0, 1, 2
    if latinIdx == 1:
        return 2, 0, 1
    if latinIdx == 2:
        return 1, 2, 0


class WriteFile:
    @staticmethod
    def pkl(data, filepath):

        with open(filepath, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def csv(file, header, cols, length):
        folder_path = os.path.dirname(file)
        # 确保文件夹存在，如果不存在，则创建文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 给每个数据单独开一个文件
            writer.writerow(header)
            for i in range(length):
                row = [cols[_][i] for _ in range(len(header))]
                writer.writerow(row)

def ReadPkl(filepath):
    with open(filepath,'rb') as file:
        data = pickle.load(file)
    return data
def dimCheck(signal_list):
    dim = None
    for signal in signal_list:
        if dim is None:
            dim = len(signal.timestamp)

        if dim != len(signal.timestamp):
            # raise Exception("聯合特徵大小不般配")
            return False

    return True


class Record:
    output = False
    def __init__(self, content):
        self.start_time = time.time()
        self.content = content
        if Record.output:
            print(content, '=====start=====')

    def end(self):
        if Record.output:
            print(self.content, '=====end=====', ' duration (s):', str(time.time() - self.start_time))
