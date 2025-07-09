# Based on Microsoft BatteryML repo
import os
import re
import pandas as pd

from tqdm.auto import tqdm
from typing import List
from pathlib import Path

from src.preprocessing.base import BasePreprocessor
from src.preprocessing.battery_data import BatteryData, TimeseriesData

class HealthyArchivePreprocessor(BasePreprocessor):
    def process(self, parentdir=None, *args, **kwargs):
        inputdir = Path(parentdir) if parentdir else Path(f'data/raw/healthy_archive_data/{self.name}/')
        cells = set(
            f.stem.split()[0]
            for f in inputdir.glob('*timeseries*')
            if f.is_file()
        )

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
                timeseries_data = self.get_timeseries_data(inputdir=inputdir, cell=cell)
            except:
                skip_batteries_num += 1
                continue

            # store data
            battery = self.get_cell_info(cell=cell, timeseries_data=timeseries_data)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num
    
    def get_timeseries_data(self, inputdir, cell, **kwargs) -> List[TimeseriesData]:
        """ 
        Get a list of TimeseriesData objects from the given filepath
        """
        expanded_cell = f"{cell}.csv"
        filename = os.path.join(inputdir, expanded_cell)
        df = pd.read_csv(filename)
        df = df.dropna(axis=1)

        return [
            TimeseriesData(
                time_in_s=df['Test_Time (s)'],
                temperature_in_C=df['Cell_Temperature (C)']
            )
        ]
        
    def get_cell_info(self, cell, timeseries_data, **kwargs) -> BatteryData:
        """ """

    
class HNEIPreprocessor(HealthyArchivePreprocessor):
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='hnei', display_name='HNEI', output_dir=output_dir, silent=silent)

    def get_cell_info(self, cell, timeseries_data, **kwargs):
        return BatteryData(
            cell_id=cell,
            organization='hnei',
            timeseries_data=timeseries_data,
            is_healthy=True,
            battery_type='LCO',
            anode_material='graphite',
            cathode_material='LCO',
            nominal_capacity_in_Ah=2.8,
            form_factor='cylindrical_18650',
        )

class OXPreprocessor(HealthyArchivePreprocessor):
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='oxford', display_name='Oxford', output_dir=output_dir, silent=silent)

    def get_cell_info(self, cell, timeseries_data, **kwargs):
        return BatteryData(
            cell_id=cell,
            organization='oxford',
            timeseries_data=timeseries_data,
            is_healthy=True,
            battery_type='LCO',
            anode_material='graphite',
            cathode_material='LCO',
            nominal_capacity_in_Ah=0.72,
            form_factor='pouch',
        )
    
