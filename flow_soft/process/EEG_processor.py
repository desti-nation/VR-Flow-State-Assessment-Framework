from process.Signal import *
from process.Top_processor import *
from signal_capture.utils.dataType import *

bands = [
    'delta',
    'theta',
    'lowAlpha','highAlpha',
    'lowBeta','highBeta',
    'lowGamma','midGamma'
]
"""
  {
    'delta': (0.5, 2.75),
    'theta': (3.5, 6.75),
    'lowAlpha': (7.5, 9.25),
    'highAlpha': (10, 11.75),
    'lowBeta': (13, 16.75),
    'highBeta': (18, 29.75),
    'lowGamma': (31, 39.75),
    'midGamma': (41, 49.75)}
"""



class EEG_processor(Top_processor):
    def __init__(self, controller, player: int):
        super().__init__(controller,
                         player=player,
                         device='EEG',
                         filelist=['Neuro_EEG_Raw.csv', 'Neuro_EEG_Power.csv']
                         )

    def preprocess(self):
        self.data['rawValue'].eeg_preprocess()


    def feature_extraction(self,stat_item_list=None):
        super().feature_extraction(stat_item_list=['rawValue','attention', 'meditation'])

    def calc_feature1(self, Feature_cls_name=None, item_name=None):
        if Feature_cls_name is None and item_name is None:
            super().calc_feature1(Feature_cls_name='Feature_EEG', item_name='rawValue')
        else:
            super().calc_feature1(Feature_cls_name=Feature_cls_name,item_name=item_name)

    def calc_feature2(self, Feature_cls_name=None, item_name_list=None):
        if Feature_cls_name is None and item_name_list is None:
            super().calc_feature2(Feature_cls_name='Feature_EEG_Power', item_name_list=bands)
        else:
            super().calc_feature2(Feature_cls_name=Feature_cls_name, item_name_list=item_name_list)

