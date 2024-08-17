from dataclasses import dataclass
from typing import Dict
from Utils import utils
import re
import os

@dataclass
class Setting:
    root: str
    images_folder: str
    A_folder: str
    D1_folder: str
    D2_folder: str

    feature_extract: str
    batch_size: int
    A_T_num_epochs: int
    A_F_num_epochs: int
    D1_num_epochs: int
    D2_num_epochs: int
    LR: float
    use_cross_valid: bool
    save_model_frequency: int
    use_top_probs: bool
    
    O_train_class_size: int
    O_test_class_size: int
    F_train_class_size: int
    F_test_class_size: int
    D1_train_class_size: int
    D1_test_class_size: int
    D2_train_class_size: int
    D2_test_class_size: int

    @staticmethod
    def from_dict(data: Dict) -> 'Setting':
        setting = Setting(**data)
        setting.feature_extract = bool(setting.feature_extract)
        setting.batch_size = int(setting.batch_size)
        setting.A_T_num_epochs = int(setting.A_T_num_epochs)
        setting.A_F_num_epochs = int(setting.A_F_num_epochs)
        setting.D1_num_epochs = int(setting.D1_num_epochs)
        setting.D2_num_epochs = int(setting.D2_num_epochs)
        setting.LR = float(setting.LR)
        setting.use_cross_valid = bool(setting.use_cross_valid)
        setting.save_model_frequency = int(setting.save_model_frequency)
        setting.use_top_probs = bool(setting.use_top_probs)
        setting.O_train_class_size = int(setting.O_train_class_size)
        setting.O_test_class_size = int(setting.O_test_class_size)
        setting.F_train_class_size = int(setting.F_train_class_size)
        setting.F_test_class_size = int(setting.F_test_class_size)
        setting.D1_train_class_size = int(setting.D1_train_class_size)
        setting.D1_test_class_size = int(setting.D1_test_class_size)
        setting.D2_train_class_size = int(setting.D2_train_class_size)
        setting.D2_test_class_size = int(setting.D2_test_class_size)
        return setting
    
    def get_net(self, name: str) -> bool:
        return name[0:3]
    
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
        match = re.search(r" - (\w*)", name)
        disease_name = ""
        if match:
            disease_name = match.group(1)[0]
        return disease_name
    
    def get_abnormities_num(self, name: str = "All Abnormities") -> int:
        criteria = utils.get_criteria()
        if name == "All Abnormities":
            return len(criteria["All"]["OCT"] + criteria["All"]["Fundus"])
        elif self.is_OCT(name) or name == "OCT Abnormities":
            return len(criteria["All"]["OCT"])
        else:
            return len(criteria["All"]["Fundus"])
        
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
        return name[:6] + "TRWT" + name[10:]
    
    def get_o_mr_name(self, name: str) -> str:
        return name[:4] + "AOTRMR"
    
    def get_f_mr_name(self, name: str) -> str:
        return name[:4] + "AFTRMR"
    
    def get_rs_file_name(self, name: str) -> str:
        return name[:8] + "RS" + name[10:]
    
    def get_transfer_learning_wt(self, name: str) -> str:
        return name[:13] + "T"
    
    def get_d1_single_disease_cmd(self, name: str, disease: str) -> str:
        return name + " - " + disease