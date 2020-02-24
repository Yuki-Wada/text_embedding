from copy import deepcopy
import numpy as np
import pandas as pd

from nlp_model.utils import save_sjis_csv

class HyperParamOptimizer:
    @staticmethod
    def get_dct_value_at_key_list(dct, key_list):
        curr_dct = dct
        for key in key_list:
            curr_dct = curr_dct[key]
        return curr_dct

    @staticmethod
    def assign_dct_value_at_key_list(dct, key_list, value):
        curr_dct = dct
        for key in key_list[:-1]:
            curr_dct = curr_dct[key]
        curr_dct[key_list[-1]] = value

        return curr_dct

    def __init__(self, base_config):
        self.base_config = base_config
        self.optim_result = {
            'score': [],
            'is_best': []
        }

        self.hyper_param_info_to_select_method = {}
        if "hyper_param_optim" not in self.base_config:
            self.optim_count = 1
            self.save_best_config_path = None
            self.save_optim_result_path = None
            return

        for hyper_param_info in base_config["hyper_param_optim"]["hyper_params"]:
            select_method = HyperParamOptimizer.get_dct_value_at_key_list(
                base_config, hyper_param_info
            )
            self.hyper_param_info_to_select_method[tuple(hyper_param_info)] = select_method

        self.optim_count = base_config["hyper_param_optim"]["optim_count"]
        if not self.hyper_param_info_to_select_method:
            self.optim_count = 1

        self.save_best_config_path = base_config["hyper_param_optim"]["save_best_config_path"]
        self.save_optim_result_path = base_config["hyper_param_optim"]["save_optim_result_path"]

    def get_next_config(self):
        next_config = deepcopy(self.base_config)
        for hyper_param_info, select_method in self.hyper_param_info_to_select_method.items():
            select_type = select_method['type']
            if select_type == 'int':
                min_value = select_method['value'][0]
                max_value = select_method['value'][1]
                selected_value = np.random.randint(min_value, max_value + 1)
            if select_type == 'float':
                min_value = select_method['value'][0]
                max_value = select_method['value'][1]
                selected_value = np.random.uniform(min_value, max_value)
            if select_type == 'category':
                values = select_method['value']
                selected_value = np.random.choice(values, 1)[0]
            HyperParamOptimizer.assign_dct_value_at_key_list(
                next_config, hyper_param_info, selected_value
            )

            if hyper_param_info[-1] not in self.optim_result:
                self.optim_result[hyper_param_info[-1]] = []
            self.optim_result[hyper_param_info[-1]].append(selected_value)

        self.optim_result['score'].append(None)
        self.optim_result['is_best'].append(None)

        return next_config

    def add_score(self, score):
        self.optim_result['score'][-1] = score
        if len(self.optim_result['score']) == 1 or \
            score > np.max(self.optim_result['score'][:-1]):
            self.optim_result['is_best'] = \
                [False] * (len(self.optim_result['score']) - 1) + [True]
        else:
            self.optim_result['is_best'][-1] = False

    def get_optim_result(self):
        optim_result = pd.DataFrame(self.optim_result)
        return optim_result

    def save_optim_result(self):
        return save_sjis_csv(self.get_optim_result(), self.save_optim_result_path)
