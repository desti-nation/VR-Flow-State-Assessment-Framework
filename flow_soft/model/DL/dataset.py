import pickle
import os,sys
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录（即model/mymodel）
current_dir = os.path.dirname(current_file_path)

# 计算flow_soft的根目录路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
print(root_dir)
# 将process目录添加到模块搜索路径
sys.path.append(root_dir)
import numpy as np
import sklearn
from filepaths import Folder

from process.Top_processor import *
# from process.EEG_processor import EEG_processor
# from process.GSR_processor import GSR_processor
# from process.VR_processor import VR_processor
from process.myFeature.Feature import Feature
from process.Signal import Signal

controller = Controller(processed=True)
# x = [_ for _ in controller.players()]
# print(len(x),x)

# def preprocess():
#     for player in controller.players():
#         print('[player=', player, ']')
#         EEG_processor(controller=controller, player=player)
#         GSR_processor(controller=controller, player=player)
#         VR_processor(controller=controller, player=player)

class Setup:
    def __init__(self,window=10,step=0.5,label_n=3,normalize=True):
        self.window = window
        self.step = step
        self.label_n = label_n
        

        self.normalize = normalize

        # dataset info 
        self.subject_n = None
        self.subjects = None
        self.channels = None
        self.example_x = None
        self.sample_rates = None

        
    def fill_dataset_info(self,subjects,channels,example_x,sample_rates):
        self.subjects = subjects
        self.subject_n = len(subjects)
        self.channels = channels
        self.example_x = example_x
        self.sample_rates = sample_rates

    

class Dataset:
    channel_n = 4
    def __init__(self,x,y,setup,x_subject,y_subject):
        self.x = x
        self.y = y
        
        self.setup = setup

        self.x_subject = x_subject
        self.y_subject = y_subject

        self.align_channel()

    def align_channel(self):
        def __align__(x):
            aligned_x = []
            MAX_SAMPLE_RATE = max(self.setup.sample_rates)
            TENSOR_SIZE = MAX_SAMPLE_RATE * self.setup.window
            
            for channel_idx in range(Dataset.channel_n):
                if self.setup.sample_rates[channel_idx] < TENSOR_SIZE:
                    # x[channel_idx] original shape: (N,T) T=sample_rate*window
                    pad_x = np.pad(x[channel_idx], ((0, 0),  (0, TENSOR_SIZE- x[channel_idx].shape[1])), 'constant', constant_values=(-1)) 
                    aligned_x.append(pad_x)
                else:
                    aligned_x.append(x[channel_idx])

            return np.array(aligned_x) # (C,N,T)
        self.x = __align__(self.x)
        for subject in self.x_subject:
            self.x_subject[subject] = __align__(self.x_subject[subject])
    def merge_subjects(self,players):
        x = None
        y = None
        for player in players:
            if x is None:
                x = self.x_subject[player]
                y = self.y_subject[player]
                continue

            x = np.concatenate((x,self.x_subject[player]),axis=1)
            y = np.concatenate((y,self.y_subject[player]),axis=0)
        return x,y

def unpickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data


invalid_players = [
    8,13
]

def dataset_filename(label_n, normalize, window,step):
    norm = 'norm' if normalize else 'unnorm'

    return f'dataset_{label_n}_{norm}_{window}_{step}.pkl'
def get_dataset(setup:Setup,save=True, sample_rates=[512,512,512,64],channel=None,read=True):
    filename = dataset_filename(setup.label_n, setup.normalize, setup.window, setup.step)
    if read and os.path.exists(os.path.join(Folder.root, filename)) :
        dataset = unpickle(os.path.join(Folder.root, filename))
        setup.fill_dataset_info(
            subjects=dataset.setup.subjects,
            channels=dataset.setup.channels,
            example_x=dataset.setup.example_x,
            sample_rates=dataset.setup.sample_rates
        )
        return dataset
    
    # normalize=False,label_n=3,sample_rates=[512,512,512,64],window=60,step=0.5,


    # setup = Setup(window=window,step=step,sample_rates=sample_rates,label_n=label_n)
    controller.processed = True
    time_helper = controller.time_helper
    x = [[] for _ in range(Dataset.channel_n)]
    y = []

    x_subject = {}
    y_subject = {}
    for player in controller.players():
        if player in invalid_players :
            continue
   

        x_subject[player] = [[] for _ in range(Dataset.channel_n)]
        y_subject[player] = []
    
        path_eeg = os.path.join(Folder.processed_pkl, str(player) + '-EEG-intact.pkl')
        path_gsr = os.path.join(Folder.processed_pkl, str(player) + '-GSR-intact.pkl')
        path_vr = os.path.join(Folder.processed_pkl, str(player) + '-VR-intact.pkl')

        et_data = unpickle(path_vr)
        left_openness = et_data['left_openness']
        right_openness = et_data['right_openness']
        assert left_openness.data.shape == right_openness.data.shape
        assert np.all(left_openness.timestamp == right_openness.timestamp )
        # print('0:',np.sum(np.logical_or(left_openness.data,right_openness.data) == 0))
        # print('1:',np.sum(np.logical_or(left_openness.data,right_openness.data) == 1))
        
        openness = Signal(item_name='openness',data=np.logical_or(left_openness.data,right_openness.data),timestamp=left_openness.timestamp)
     
        data = [
            unpickle(path_eeg)['rawValue'],
            unpickle(path_gsr)['GSR_uS'],
            unpickle(path_gsr)['PPG_v'],
            openness
        ]
        assert len(data) == len(sample_rates) == Dataset.channel_n

        print('load data from subject#', player)
        for epoch_i in range(3):
            difficulty = time_helper.order[player][epoch_i]
            if setup.label_n == 3:
                label = {'easy':0,'optimal':1,'hard':2}[difficulty] 
            elif setup.label_n == 2:
                label =  {'easy':0,'optimal':1,'hard':0}[difficulty] 
            play_timespan = time_helper.chip[player][epoch_i][1]
            base_timespan = time_helper.chip[player][epoch_i][0]

            window_cnt_check = None

            for channel_idx in range(Dataset.channel_n):
                # print('channel#',channel_idx)
                base_signal = data[channel_idx].slice(begin=base_timespan[0],window=base_timespan[1]-base_timespan[0])
                base_signal = base_signal.slice(window=30)
                base_offset = np.mean(base_signal.data)

                play_signal = data[channel_idx].slice(begin=play_timespan[0],window=play_timespan[1]-play_timespan[0])
                play_signal = play_signal.slice(window=150)
                play_signal.resample(new_sample_rate = sample_rates[channel_idx])
                window_list = Feature.rolling_slice2(window=setup.window,step=setup.step,signal=play_signal)

                # 检查
                if window_cnt_check is None:
                    window_cnt_check = len(window_list)
                else:
                    assert window_cnt_check == len(window_list), 'cur:'+str(len(window_list))+' before:'+str(window_cnt_check)
                
                datashape_check = None

                for _data in window_list:
                    if datashape_check is None:
                        datashape_check = _data.data.shape
                    else:
                        assert datashape_check == _data.shape, 'cur:'+str(_data.data.shape)+' before:'+str(datashape_check)

                x[channel_idx] += [ _ - base_offset for _ in window_list]
                x_subject[player][channel_idx] += [ _ - base_offset for _ in window_list]
            y += [label] * window_cnt_check
            y_subject[player] += [label] * window_cnt_check
            print('window_cnt_check ',window_cnt_check)
    #对于每个channel : x[channel_idx]有cnt_w个窗口数据，每个窗口数据是一个window*sample_rate的列表
    # print('x_shape (for one channel)',x[0].shape,x[1].shape)
    x = [np.array(_) for _ in x] # every channel: (N,L)
    y = np.array(y)

    for player in x_subject:
        x_subject[player] = [np.array(_) for _ in x_subject[player]]
        y_subject[player] = np.array(y_subject[player])

    setup.fill_dataset_info(
        subjects=set(controller.players()) - set(invalid_players),
        channels=[
            'EEG','GSR_uS','PPG_v','blink'
        ],
        example_x= [np.zeros(_.shape)[0:1,np.newaxis,:] for _ in x],
        sample_rates=sample_rates
    )

    if setup.normalize:
        for channel_idx in range(Dataset.channel_n):
            mean,std = np.mean(x[channel_idx]),np.std(x[channel_idx])
            x[channel_idx] = (x[channel_idx]-mean)/std

            for player in x_subject:
                x_subject[player][channel_idx] = (x_subject[player][channel_idx]-mean)/std
  

    print('generate dataset with x shape(for every channel): ',x[-1].shape,' y shape: ',y.shape)
    # enc = sklearn.preprocessing.OneHotEncoder()
    # enc.fit(y.reshape(-1, 1))
    # y = enc.transform(y.reshape(-1, 1)).toarray()
    
    dataset = Dataset(x,y,setup,x_subject,y_subject)
    if save:
        with open(os.path.join(Folder.root, filename), 'wb') as file:
            # pickle.dump(dataset, file)
            pickle.dump(dataset, file, protocol=4)
    return dataset

# from sklearn.model_selection import train_test_split


# # get_dataset(read=False)
# if __name__ == '__main__':

    
#     get_dataset(setup,read=False,save=True)

    # dataset = unpickle(File.dataset)
    # dataset.setup.example_x = [np.zeros(_.shape)[0:1,np.newaxis,:] for _ in dataset.x]
   
    # with open(File.dataset, 'wb') as file:
    #     # pickle.dump(dataset, file)
    #     pickle.dump(dataset, file, protocol=4)

