# Based on Microsoft BatteryML repo
import os
import re
import logging

from typing import List
from tqdm.auto import tqdm
from pathlib import Path

from src.config import config
from src.preprocessing.battery_data import BatteryData, TimeseriesData

class BasePreprocessor:
    """
    Super class for preprocessor objects
    """
    BATTERY_TYPES = ['NFP', 'LCO']
    CATHODES = ['LCO', 'LFP', 'NMC-LMO', 'NMC-LCO', 'NMC', 'LMO-LNO', 'LMO', 'LNO', 'NCA', 'NFP']
    ANODES = ['graphite']

    def __init__(self,
                name: str,
                *,
                display_name: str = None,
                output_dir: str = None,
                silent: bool = True):
        self.name = name
        self.display_name = display_name or name
        self.output_dir = output_dir or f'{config.PROCESSED_DATA_DIR}{name}/'
        self.silent = silent

    def process(self, *args, **kwargs) -> List[BatteryData]:
        """Main logic for preprocessing data."""
  
    def _process_cells(self, inputdir:Path, cells:set=None, *args, **kwargs) -> List[BatteryData]:
        assert os.path.exists(inputdir), f'Input path does not exist: {inputdir}'
        if cells==None:
            cells = set(
                f.stem.split()[0]
                for f in inputdir.glob('*')
                if f.is_file()
            )
        # create output folder if it doesn't already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
          
        process_batteries_num = 0
        skip_batteries_num = 0
        for cell in tqdm(cells, desc=f'Processing {self.display_name} cells'):
            # judge whether to skip the processed file
            whether_to_skip = self.check_processed_file(cell)
            if whether_to_skip == True:
                skip_batteries_num += 1
                continue

            # get data from the file
            try:
                timeseries_data = self.get_timeseries_data(inputdir=inputdir, cell=cell, *args, **kwargs)
            except:
                skip_batteries_num += 1
                continue

            # store data
            battery = self.get_cell_info(cell=cell, timeseries_data=timeseries_data, *args, **kwargs)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num
    
    def get_timeseries_data(self, *args, **kwargs) -> List[TimeseriesData]:
        """ """

    def get_cell_info(self, *args, **kwargs) -> BatteryData:
        """ """

    def __call__(self, *args, **kwargs):
        process_batteries_num, skip_batteries_num = self.process(
            *args, **kwargs)
        if not self.silent:
            print(f'Successfully processed {process_batteries_num} batteries.')
            print(f'Skip processing {skip_batteries_num} batteries.')

    def check_processed_file(self, processed_file: str):
        expected_pkl_path = os.path.join(
            self.output_dir, (f"{processed_file}.pkl"))
        if os.path.exists(expected_pkl_path) and os.path.getsize(expected_pkl_path) > 0:
            logging.info(
                f'Skip processing {processed_file}, pkl file already exists and is not empty.')
            return True
        elif os.path.exists(expected_pkl_path) and os.path.getsize(expected_pkl_path) == 0:
            logging.info(
                f'Found empty pkl file for {processed_file}.')
        return False

    def dump_single_file(self, battery: BatteryData):
        battery.dump(f'{self.output_dir}/{battery.cell_id}.pkl')

    def summary(self, batteries: List[BatteryData]):
        print(f'Successfully processed {len(batteries)} batteries.')

