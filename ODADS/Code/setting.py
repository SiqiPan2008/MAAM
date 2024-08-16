from dataclasses import dataclass
from typing import Dict

@dataclass
class Setting:
    root: str
    images_folder: str
    ab_folder: str
    d1_folder: str
    d2_folder: str

    @staticmethod
    def from_dict(data: Dict) -> 'Setting':
        return Setting(**data)
    
    def get_net(name: str):
        return name[0:3]
    

        
