import os
import os.path as osp
from dataclasses import dataclass
from typing import List
import numpy as np
from collections import Counter
import pandas as pd
import operator
import joblib


@dataclass
class MARTDatasetFolder:
    """MART Dataset folder path management"""
    autographer_path: str
    pandas_path: str
    processed_path: str
    raw_path: str
    screenshots_path: str


class MARTDatasetHelper:


    def __init__(self, data_path: str = "/MART/MART-Data"):
        data_folder_paths = [osp.join(data_path, folder_name) for folder_name in os.listdir(data_path)]
        autographer_folder_path, pandas_folder_path, processed_folder_path, raw_folder_path, screenshots_folder_path = data_folder_paths
        self.mart_dataset_folder = MARTDatasetFolder(
            autographer_path = autographer_folder_path,
            pandas_path = pandas_folder_path,
            processed_path = processed_folder_path,
            raw_path = raw_folder_path,
            screenshots_path = screenshots_folder_path
        )
    
    
    # Autographer data methods
    @property
    def autographer_list(self) -> List[str]:
        _autographer_list = os.listdir(self.mart_dataset_folder.autographer_path)
        return _autographer_list

    @property
    def autographer_names(self) -> List[str]:
        _autographer_names = [autographer_name.split('.')[0] for autographer_name in self.autographer_list]
        return _autographer_names
    
    
    def get_train_test_data(self) -> tuple:
        autographer_names = self.autographer_names
        train_test_labels = [autographer_name.split('_')[1] for autographer_name in autographer_names]
        train_data = [autographer_names[index] for index, label in enumerate(train_test_labels) if label[:4] != 'pred']
        test_data = [autographer_names[index] for index, label in enumerate(train_test_labels) if label[:4] == 'pred']
        return train_data, test_data
    

    def get_trainAB_data(self) -> tuple: 
        autographer_names = self.autographer_names
        train_test_labels = [autographer_name.split('_')[1] for autographer_name in autographer_names]
        trainAB_labels = [label for label in train_test_labels if label != 'pred']
        trainA_data = [autographer_names[index] for index, label in enumerate(trainAB_labels) if label == 'trainA']
        trainB_data = [autographer_names[index] for index, label in enumerate(trainAB_labels) if label == 'trainB']
        return trainA_data, trainB_data


    def get_train_test_perc(self):
        train_data, test_data = self.get_train_test_data()
        sizes = [len(train_data), len(test_data)]
        perc_train, perc_test = self.__compute_perc(sizes)       
        return perc_train, perc_test

    
    def get_trainAB_perc(self):
        trainA, trainB = self.get_trainAB_data()
        sizes = [len(trainA), len(trainB)]
        perc_trainA, perc_trainB = self.__compute_perc(sizes)
        return perc_trainA, perc_trainB

    
    def get_train_by_activities(self):
        trainA_data, trainB_data = self.get_trainAB_data()
        acts_trainA = [autographer_name.split('_')[2] for autographer_name in trainA_data]
        acts_trainB = [autographer_name.split('_')[2] for autographer_name in trainB_data]
        num_acts_trainAB = Counter(acts_trainA + acts_trainB)
        cnt_by_activities = {}
        for act, cnt in num_acts_trainAB.items():
            try:
                cnt_by_activities[act] += cnt
            except Exception as e:
                cnt_by_activities[act] = cnt
        return cnt_by_activities


    # Processed data methods
    @property
    def processed_mart_data(self):
        autographer_embedded_train_path = osp.join(self.mart_dataset_folder.processed_path, 'autographer_embeded_ft_train.joblib')
        autographer_embedded_test_path = osp.join(self.mart_dataset_folder.processed_path, 'autographer_embeded_ft_test.joblib')
        autographer_embedded_train = joblib.load(autographer_embedded_train_path)
        autographer_embedded_test = joblib.load(autographer_embedded_test_path)
        return autographer_embedded_train, autographer_embedded_test

    
    # Pandas data methods
    @property
    def pandas_mart_data(self):
        trainA_data_path = osp.join(self.mart_dataset_folder.pandas_path, 'trainA.csv')
        trainB_data_path = osp.join(self.mart_dataset_folder.pandas_path, 'trainB.csv')
        test_data_path = osp.join(self.mart_dataset_folder.pandas_path, 'test.csv')
        trainA_pd_data = pd.read_csv(trainA_data_path)
        trainB_pd_data = pd.read_csv(trainB_data_path)
        test_pd_data = pd.read_csv(test_data_path)
        return trainA_pd_data, trainB_pd_data, test_pd_data


    def get_train_pd_labels(self):
        trainA_pd_data, trainB_pd_data, _ = self.pandas_mart_data
        trainA_labels = trainA_pd_data['event_id'].tolist()
        trainB_labels = trainB_pd_data['event_id'].tolist()
        train_labels = trainA_labels + trainB_labels
        train_labels = [int(label[3:])-1 for label in train_labels]
        return train_labels


    def get_signal_pd_data(self):
        columns = ['data_HR_activity_median', 'data_HR_activity_min', 'data_HR_activity_max', 'data_HR_activity_average', 'data_HR_activity_std', 'data_HR_activity_len', 
            'data_LEFT_ACC_MAG_median',	'data_LEFT_ACC_MAG_min', 'data_LEFT_ACC_MAG_max', 'data_LEFT_ACC_MAG_average', 'data_LEFT_ACC_MAG_std',	'data_LEFT_ACC_MAG_len', 'data_LEFT_ACC_X_median', 'data_LEFT_ACC_X_min',
            'data_LEFT_ACC_X_max', 'data_LEFT_ACC_X_average', 'data_LEFT_ACC_X_std', 'data_LEFT_ACC_X_len',	'data_LEFT_ACC_Y_median', 'data_LEFT_ACC_Y_min', 'data_LEFT_ACC_Y_max',	
            'data_LEFT_ACC_Y_average', 'data_LEFT_ACC_Y_std', 'data_LEFT_ACC_Y_len', 'data_LEFT_ACC_Z_median', 'data_LEFT_ACC_Z_min', 'data_LEFT_ACC_Z_max', 'data_LEFT_ACC_Z_average', 
            'data_LEFT_ACC_Z_std', 'data_LEFT_ACC_Z_len', 'data_RIGHT_ACC_MAG_median', 'data_RIGHT_ACC_MAG_min', 'data_RIGHT_ACC_MAG_max', 'data_RIGHT_ACC_MAG_average', 'data_RIGHT_ACC_MAG_std', 
            'data_RIGHT_ACC_MAG_len', 'data_RIGHT_ACC_X_median', 'data_RIGHT_ACC_X_min', 'data_RIGHT_ACC_X_max', 'data_RIGHT_ACC_X_average', 'data_RIGHT_ACC_X_std', 'data_RIGHT_ACC_X_len', 'data_RIGHT_ACC_Y_median', 
            'data_RIGHT_ACC_Y_min',	'data_RIGHT_ACC_Y_max', 'data_RIGHT_ACC_Y_average',	'data_RIGHT_ACC_Y_std', 'data_RIGHT_ACC_Y_len',	'data_RIGHT_ACC_Z_median', 'data_RIGHT_ACC_Z_min', 'data_RIGHT_ACC_Z_max', 'data_RIGHT_ACC_Z_average',
            'data_RIGHT_ACC_Z_std',	'data_RIGHT_ACC_Z_len',	'data_HEAD_MAG_by_activity_median',	'data_HEAD_MAG_by_activity_min', 'data_HEAD_MAG_by_activity_max', 'data_HEAD_MAG_by_activity_average', 'data_HEAD_MAG_by_activity_std', 
            'data_HEAD_MAG_by_activity_len', 'data_HEAD_X_by_activity_median', 'data_HEAD_X_by_activity_min', 'data_HEAD_X_by_activity_max', 'data_HEAD_X_by_activity_average',	'data_HEAD_X_by_activity_std', 'data_HEAD_X_by_activity_len', 
            'data_HEAD_Y_by_activity_median', 'data_HEAD_Y_by_activity_min', 'data_HEAD_Y_by_activity_max',	'data_HEAD_Y_by_activity_average', 'data_HEAD_Y_by_activity_std', 'data_HEAD_Y_by_activity_len', 
            'data_HEAD_Z_by_activity_median', 'data_HEAD_Z_by_activity_min', 'data_HEAD_Z_by_activity_max', 'data_HEAD_Z_by_activity_average', 'data_HEAD_Z_by_activity_std', 'data_HEAD_Z_by_activity_len',
            'data_MOUSE_PIX_DISTS_median', 'data_MOUSE_PIX_DISTS_min', 'data_MOUSE_PIX_DISTS_max', 'data_MOUSE_PIX_DISTS_average', 'data_MOUSE_PIX_DISTS_std', 'data_MOUSE_PIX_DISTS_len', 
            'data_MOUSE_TIMEDIFFS_median', 'data_MOUSE_TIMEDIFFS_min', 'data_MOUSE_TIMEDIFFS_max', 'data_MOUSE_TIMEDIFFS_average', 'data_MOUSE_TIMEDIFFS_std', 'data_MOUSE_TIMEDIFFS_len', 'data_MOUSE_VELOCITY_median', 'data_MOUSE_VELOCITY_min', 'data_MOUSE_VELOCITY_max',
            'data_MOUSE_VELOCITY_average', 'data_MOUSE_VELOCITY_std', 'data_MOUSE_VELOCITY_len', 
            'data_EOG_UD_by_activity_median', 'data_EOG_UD_by_activity_min', 'data_EOG_UD_by_activity_max', 'data_EOG_UD_by_activity_average', 'data_EOG_UD_by_activity_std', 
            'data_EOG_UD_by_activity_len', 'data_EOG_LR_by_activity_median', 'data_EOG_LR_by_activity_min', 'data_EOG_LR_by_activity_max', 'data_EOG_LR_by_activity_average', 'data_EOG_LR_by_activity_std', 'data_EOG_LR_by_activity_len'
        ]
        trainA_pd_data, trainB_pd_data, test_pd_data = self.pandas_mart_data
        trainA_signal_data = trainA_pd_data[columns]
        trainB_signal_data = trainB_pd_data[columns]
        test_signal_data = test_pd_data[columns]
        train_signal_data = pd.concat([trainA_signal_data, trainB_signal_data])
        return train_signal_data, test_signal_data

    
    def get_pd_train_test_item_ids(self) -> tuple:
        trainA_pd_data, trainB_pd_data, test_pd_data = self.pandas_mart_data
        train_pd_data = pd.concat([trainA_pd_data, trainB_pd_data])
        train_pd_names = train_pd_data[['sub_id', 'event_id', 'source']]
        test_pd_names = test_pd_data[['sub_id', 'event_id']]
        train_item_ids = []
        test_item_ids = []
        for row in train_pd_names.values:
            sub_id, event_id, source = row
            item_id = f'{sub_id}_{source}_{event_id}'
            train_item_ids.append(item_id)
        for row in test_pd_names.values:
            sub_id, event_id = row
            item_id = f'{sub_id}_{event_id}'
            test_item_ids.append(item_id) 
        return train_item_ids, test_item_ids


    # Other helper functions
    def __compute_perc(self, sizes: List[int]) -> List[float]:
        total = sum(sizes)
        perc = np.array(sizes) / total
        return perc


    def dump_test_result(self, y_prob, output_file_path):
        _, _, test_pd_data = self.pandas_mart_data
        df = test_pd_data[['sub_id', 'event_id']]
        # Create test ids
        test_item_ids = [] 
        for row in df.values:
            sub_id, event_id = row
            test_item_id = f'{sub_id}_{event_id}'
            test_item_ids.append(test_item_id)
        test_item_ids = np.array(test_item_ids)
        sorted_prob_indices_by_act = np.argsort(y_prob, axis=0)[::-1] # Sorted indices in decreasing order
        num_item, num_act = y_prob.shape
        with open(output_file_path, 'w') as f:
            print('group_id: group14 forests', file=f)
            for act_id in range(num_act):
                act_text_value = 'act{:02d}'.format(act_id+1)
                sorted_prob_index_by_act = sorted_prob_indices_by_act[:, act_id]
                item_ids = test_item_ids[sorted_prob_index_by_act].tolist()
                for item_id in item_ids:
                    print(f'{act_text_value} {item_id}', file=f)