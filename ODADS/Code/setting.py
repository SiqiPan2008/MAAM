from dataclasses import dataclass
from typing import Dict
from Utils import utils
import re
import os

@dataclass
class Setting:
    use_small: bool

    root: str
    data_folder: str
    data_folder_s: str
    images_folder: str
    A_folder: str
    D1_folder: str
    D2_folder: str

    feature_extract: bool
    batch_size: int
    test_batch_size: int
    LR: float
    use_cross_valid: bool
    save_model_frequency: int
    use_top_probs: bool

    A_top_probs_max_num: int
    A_top_probs_min_prob: float
    D2_top_probs_max_num: int
    D2_top_probs_min_prob: float
    
    A_T_num_epochs: int
    A_F_num_epochs: int
    D1_num_epochs: int
    D2_num_epochs: int
    O_train_class_size: int
    O_test_class_size: int
    F_train_class_size: int
    F_test_class_size: int
    D1_train_class_size: int
    D1_test_class_size: int
    D2_train_class_size: int
    D2_test_class_size: int

    test_batch_size_s: int
    A_T_num_epochs_s: int
    A_F_num_epochs_s: int
    D1_num_epochs_s: int
    D2_num_epochs_s: int
    O_train_class_size_s: int
    O_test_class_size_s: int
    F_train_class_size_s: int
    F_test_class_size_s: int
    D1_train_class_size_s: int
    D1_test_class_size_s: int
    D2_train_class_size_s: int
    D2_test_class_size_s: int

    @staticmethod
    def from_dict(data: Dict) -> 'Setting':
        setting = Setting(**data)
        setting.use_small = setting.use_small == "True"
        
        setting.data_folder = setting.data_folder_s if setting.use_small else setting.data_folder 
        data_folder_with_root = os.path.join(setting.root, setting.data_folder)
        setting.images_folder = os.path.join(data_folder_with_root, setting.images_folder)
        setting.A_folder = os.path.join(data_folder_with_root, setting.A_folder)
        setting.D1_folder = os.path.join(data_folder_with_root, setting.D1_folder)
        setting.D2_folder = os.path.join(data_folder_with_root, setting.D2_folder)
        
        setting.feature_extract = setting.feature_extract == "True"
        setting.batch_size = int(setting.batch_size)
        setting.test_batch_size = int(setting.test_batch_size)
        setting.LR = float(setting.LR)
        setting.use_cross_valid = setting.use_cross_valid == "True"
        setting.save_model_frequency = int(setting.save_model_frequency)
        setting.use_top_probs = setting.use_top_probs == "True"

        setting.A_top_probs_max_num = int(setting.A_top_probs_max_num)
        setting.A_top_probs_min_prob = float(setting.A_top_probs_min_prob)
        setting.D2_top_probs_max_num = int(setting.D2_top_probs_max_num)
        setting.D2_top_probs_min_prob = float(setting.D2_top_probs_min_prob)
        
        setting.A_T_num_epochs = int(setting.A_T_num_epochs_s \
            if setting.use_small else setting.A_T_num_epochs)
        setting.A_F_num_epochs = int(setting.A_F_num_epochs_s \
            if setting.use_small else setting.A_F_num_epochs)
        setting.D1_num_epochs = int(setting.D1_num_epochs_s \
            if setting.use_small else setting.D1_num_epochs)
        setting.D2_num_epochs = int(setting.D2_num_epochs_s \
            if setting.use_small else setting.D2_num_epochs)
        
        setting.O_train_class_size = int(setting.O_train_class_size_s \
            if setting.use_small else setting.O_train_class_size)
        setting.O_test_class_size = int(setting.O_test_class_size_s \
            if setting.use_small else setting.O_test_class_size)
        setting.F_train_class_size = int(setting.F_train_class_size_s \
            if setting.use_small else setting.F_train_class_size)
        setting.F_test_class_size = int(setting.F_test_class_size_s \
            if setting.use_small else setting.F_test_class_size)
        setting.D1_train_class_size = int(setting.D1_train_class_size_s \
            if setting.use_small else setting.D1_train_class_size)
        setting.D1_test_class_size = int(setting.D1_test_class_size_s \
            if setting.use_small else setting.D1_test_class_size)
        setting.D2_train_class_size = int(setting.D2_train_class_size_s \
            if setting.use_small else setting.D2_train_class_size)
        setting.D2_test_class_size = int(setting.D2_test_class_size_s \
            if setting.use_small else setting.D2_test_class_size)
        return setting
    
    def get_net(self, name: str) -> bool:
        return name[:3]
    
    def is_abnormity(self, name: str) -> bool:
        return name[3] == 'A'
    
    def is_OCT(self, name: str) -> bool:
        return name[4] == 'O'
    
    def is_transfer_learning(self, name: str) -> bool:
        match = re.search(r" - ([FT])", name)
        training_type = ""
        if match:
            training_type = match.group(1)[0]
        return training_type == 'T'
    
    def is_diagnosis1(self, name: str) -> bool:
        return name[3:5] == "D1"
    
    def is_diagnosis2(self, name: str) -> bool:
        return name[3:5] == "D2"
    
    def get_disease_name(self, name: str) -> str:
        return name[12:]
    
    def get_abnormity_num(self, name: str = "All Abnormities") -> int:
        criteria = utils.get_criteria()
        if name == "All Abnormities":
            return len(criteria["All"]["OCT"] + criteria["All"]["Fundus"])
        elif self.is_OCT(name) or name == "OCT Abnormities":
            return len(criteria["All"]["OCT"])
        else:
            return len(criteria["All"]["Fundus"])
        
    def get_disease_abnormity_num(self, name, type: str = "All Abnormities") -> int:
        criteria = utils.get_criteria()
        disease_name = self.get_disease_name(name)
        disease_name = disease_name if disease_name else name
        if type == "All Abnormities":
            return len(criteria[disease_name]["OCT"] + criteria[disease_name]["Fundus"])
        elif type == "OCT Abnormities":
            return len(criteria[disease_name]["OCT"])
        else:
            return len(criteria[disease_name]["Fundus"])
        
    def get_abnormities(self, name: str = "All Abnormities") -> list:
        criteria = utils.get_criteria()
        oct_abnormities = [("OCT", abnormity) for abnormity in criteria["All"]["OCT"]]
        fundus_abnormities = [("Fundus", abnormity) for abnormity in criteria["All"]["Fundus"]]
        if name == "All Abnormities":
            return  oct_abnormities + fundus_abnormities
        elif self.is_OCT(name) or name == "OCT Abnormities":
            return oct_abnormities
        else:
            return fundus_abnormities
    
    def get_correct_abnormities(self, name: str) -> list:
        criteria = utils.get_criteria()
        disease_name = self.get_disease_name(name)
        disease_name = disease_name if disease_name else name
        return [("OCT", abnormity) for abnormity in criteria[disease_name]["OCT"]] + \
            [("Fundus", abnormity) for abnormity in criteria[disease_name]["Fundus"]]
            
    def get_incorrect_abnormities(self, name: str) -> list:
        all_abnormities = self.get_abnormities("All Abnormities")
        correct_abnormities = self.get_correct_abnormities(name)
        return [abnormity for abnormity in all_abnormities if abnormity not in correct_abnormities]
    
    def get_num_epochs(self, name: str) -> int:
        if self.is_abnormity(name):
            if self.is_transfer_learning(name):
                return self.A_T_num_epochs
            else:
                return self.A_F_num_epochs
        elif self.is_diagnosis1(name):
            return self.D1_num_epochs
        elif self.is_diagnosis2(name):
            return self.D2_num_epochs
        
    def get_img_folder(self, name: str) -> str:
        img_folder = os.path.join(self.images_folder, name[5:7])
        img_type = "OCT" if self.is_OCT(name) else "Fundus"
        return os.path.join(img_folder, img_type)
    
    def get_folder_path(self, name: str) -> str:
        if self.is_abnormity(name):
            return self.A_folder
        elif self.is_diagnosis1(name):
            return self.D1_folder
        elif self.is_diagnosis2(name):
            return self.D2_folder
        
    def get_wt_file_name(self, name: str) -> str:
        return name[:5] + "TRWT" + name[9:]
    
    def get_o_mr_name(self, name: str) -> str:
        return name[:3] + "AOTRMR"
    
    def get_f_mr_name(self, name: str) -> str:
        return name[:3] + "AFTRMR"
    
    def get_rs_file_name(self, name: str) -> str:
        return name[:7] + "RS" + name[9:]
    
    def get_training_rs_name(self, name: str) -> str:
        return name[:5] + "TRRS" + name[9:]
    
    def get_testing_rs_file_name(self, name: str) -> str:
        return name[:5] + "TORS" + name[9:]
    
    def get_transfer_learning_wt(self, name: str) -> str:
        return name[:7] + "WT - T"
    
    def get_d1_single_disease_rs(self, name: str, disease: str) -> str:
        return name + " - " + disease
    
    def get_d1_single_disease_wt(self, name: str, disease: str) -> str:
        return name[:3] + "D1TRWT - " + disease
    
    def get_d2_input_length(self) -> int:
        criteria = utils.get_criteria()
        return sum([
            (len(criteria[disease]["OCT"]) + len(criteria[disease]["Fundus"]) + 1) \
            for disease in criteria.keys() \
            if disease not in ["Normal", "All"]
        ])
        
    def get_disease_num(self, include_normal: bool = True) -> int:
        criteria = utils.get_criteria()
        return len(criteria) - 1 if include_normal else len(criteria) - 2
    
    def get_diseases(self, include_normal: bool = True) -> list:
        criteria = utils.get_criteria()
        if include_normal:
            diseases = [disease for disease in criteria.keys() if disease != "All"]
        else:
            diseases = [disease for disease in criteria.keys() if disease != "All" and disease != "Normal"]
        return diseases
    
    def get_epoch_num(self, filename: str) -> int:
        return int(filename[-7:-4])
    
    def get_abnormity_model_datasizes(self, name: str) -> tuple:
        if self.is_OCT(name):
            return (self.O_train_class_size, self.O_test_class_size)
        else:
            return (self.F_train_class_size, self.F_test_class_size)