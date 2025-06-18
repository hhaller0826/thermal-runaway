import os
import re
import pandas as pd
from src.config import config

class Organization:
    def __init__(self, name, input_dir=None, output_dir=None):
        self.name = name
        self.input_dir = input_dir if input_dir else f'{config.RAW_DATA_DIR}{name}/'
        self.output_dir = output_dir if output_dir else f'{config.PROCESSED_DATA_DIR}{name}/'

        self.files = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

_battery_types = ['NFP', 'LCO']
class OakRidge(Organization):
    def __init__(self):
        super().__init__('oakridge', input_dir=f'{config.RAW_DATA_DIR}oakridge/excel')
        self.file_info = pd.read_excel('data/raw/oakridge/main.xlsx', header=None)
        self.file_info = self._process_file_info()

    def _process_file_info(self):
        main_df = pd.read_excel('data/raw/oakridge/main.xlsx', header=None)
        file_info = {}

        for file in self.files:
            
            file_info[file] = {
                'soc': int(match.group(1)) if (match := re.search(r'(\d+)[S0]OC', file)) else None,
                'battery_type': next((bt for bt in _battery_types if bt in file), None),
            }
        return file_info

    def _time_substrings():
        return ['Time']
    
    def _temp_substrings():
        return ['°C', '[C]']



    