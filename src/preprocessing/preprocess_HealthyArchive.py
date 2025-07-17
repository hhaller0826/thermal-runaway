# Based on Microsoft BatteryML repo
import os
import re
import pandas as pd

from tqdm.auto import tqdm
from typing import List
from pathlib import Path

from src.builders import PREPROCESSORS
from src.preprocessing.base import BasePreprocessor
from src.data.battery_data import BatteryData, TimeseriesData

class HealthyArchivePreprocessor(BasePreprocessor):
    def process(self, parentdir=None, *args, **kwargs):
        inputdir = Path(parentdir) if parentdir else Path(f'data/raw/healthy_archive_data/{self.name}/')
        cells = set(
            f.stem.split()[0]
            for f in inputdir.glob('*timeseries*')
            if f.is_file()
        )
        return super()._process_cells(inputdir=inputdir, cells=cells)
    
    def get_timeseries_data(self, inputdir, cell, **kwargs) -> List[TimeseriesData]:
        """ 
        Get a list of TimeseriesData objects from the given filepath
        """
        expanded_cell = f"{cell}.csv"
        filename = os.path.join(inputdir, expanded_cell)
        df = pd.read_csv(filename)
        df = df.dropna(axis=1)
        assert df.size > 0

        return [TimeseriesData(
                time_in_s=df['Test_Time (s)'],
                temperature_in_C=df['Cell_Temperature (C)']
            )]
        
    def get_cell_info(self, cell, timeseries_data, **kwargs) -> BatteryData:
        """ """
        return BatteryData(
            cell_id=cell,
            organization=self.name,
            timeseries_data=timeseries_data,
            is_healthy=True,

            anode_material=self.get_anode(cell),
            cathode_material=self.get_cathode(cell),
            nominal_capacity_in_Ah=self.get_capacity(cell),
            form_factor=self.get_form_factor(cell)
        )
    
    def get_anode(self, cell): return 'graphite'
    def get_cathode(self, cell): return 'LCO'
    def get_capacity(self, cell): None
    def get_form_factor(self, cell): None

@PREPROCESSORS.register()
class CALCEPreprocessor(HealthyArchivePreprocessor):
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='calce', display_name='Center for Advanced Life Cycle Engineering', output_dir=output_dir, silent=silent)
    
    def get_capacity(self, cell): 
        return 1.1 if 'CS' in cell.upper() else 1.35
    
    def get_form_factor(self, cell):
        return 'prismatic'
    
@PREPROCESSORS.register()
class HNEIPreprocessor(HealthyArchivePreprocessor):
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='hnei', display_name='Hawaii Natural Energy Institute', output_dir=output_dir, silent=silent)
    
    def get_capacity(self, cell): 
        return 2.8
    
    def get_form_factor(self, cell):
        return 'cylindrical_18650'
    
class MichiganPreprocessor(HealthyArchivePreprocessor):
    # NOTE: This isn't really implemented but it's fine bc the Michigan files don't have temp data
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='michigan', display_name='Michigan', output_dir=output_dir, silent=silent)

@PREPROCESSORS.register()
class OXPreprocessor(HealthyArchivePreprocessor):
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='oxford', display_name='Oxford', output_dir=output_dir, silent=silent)

    def get_capacity(self, cell): 
        return 0.72
    
    def get_form_factor(self, cell):
        return 'pouch'

@PREPROCESSORS.register()
class SNLPreprocessor(HealthyArchivePreprocessor):
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='snl', display_name='Sandia National Lab', output_dir=output_dir, silent=silent)

    def process(self, parentdir=None, *args, **kwargs):
        if parentdir is None: parentdir = f'data/raw/healthy_archive_data/{self.name}/'

        process_batteries_num = 0
        skip_batteries_num = 0

        for cathode in ['LFP', 'NCA', 'NMC']:
            inputdir = Path(f'{parentdir}{cathode}/')
            cells = set(
                f.stem.split()[0]
                for f in inputdir.glob('*timeseries*')
                if f.is_file()
            )

            processed, skipped = super()._process_cells(inputdir=inputdir, cells=cells, cathode=cathode)
            process_batteries_num += processed
            skip_batteries_num += skipped

        return process_batteries_num, skip_batteries_num
    
    def get_cathode(self, cell):
        return cell.split('_')[2]
    
    def get_capacity(self, cell): 
        if 'NMC' in cell:
            if '15C' in cell:
                return 3 * 0.9
            return 3
        if 'NCA' in cell:
            if '20-80' in cell:  # Only use 60% of the capacity
                return 1.92  # 3.2 * 0.6
            if '15C' in cell:  # Low temperature compromise
                return 3.2 * 0.9
            return 3.2
        return 1.1
    
    def get_form_factor(self, cell):
        return 'cylindrical_18650'
    

@PREPROCESSORS.register()
class ULPurduePreprocessor(HealthyArchivePreprocessor):
    def __init__(self, *, output_dir = None, silent = True):
        super().__init__(name='ul-purdue', display_name='Underwriters Lab - Purdue University', output_dir=output_dir, silent=silent)
    
    def get_capacity(self, cell): 
        capacity = 3.4
        if '2.5-96.5' in cell:  # Only charge for 94%
            capacity *= 0.94
        return capacity
    
    def get_form_factor(self, cell):
        return 'cylindrical_18650'
