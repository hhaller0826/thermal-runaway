# Based on Microsoft BatteryML repo
import os
import logging
from typing import List
from tqdm.auto import tqdm
from pathlib import Path


from src.config import config
from src.preprocessing.battery_data import BatteryData


class BasePreprocessor:


   BATTERY_TYPES = ['NFP', 'LCO']
   CATHODES = ['LCO', 'LFP', 'NMC-LMO', 'NMC-LCO', 'NMC', 'LMO-LNO', 'LMO', 'LNO', 'NCA']
   ANODES = ['graphite']


   def __init__(self,
                name: str,
                output_dir: str = None,
                silent: bool = True):
       self.name = name
       self.output_dir = output_dir if output_dir else f'{config.PROCESSED_DATA_DIR}{name}/'
       self.silent = silent
  
   def process(self, *args, **kwargs) -> List[BatteryData]:
       """Main logic for preprocessing data."""


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




class HealthyZipProcessor(BasePreprocessor):
   def process(self, parentdir=None, globstr='*', *args, **kwargs):
       inputdir = Path(parentdir) if parentdir else Path(f'data/raw/{self.name}/')
       cells = set(
           f.stem.split()[0]
           for f in inputdir.glob(globstr)
           if f.is_file()
       )


       process_batteries_num = 0
       skip_batteries_num = 0
       for cell in tqdm(cells, desc='Processing Oxford cells'):
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
  
   def get_timeseries_data(self, **kwargs):
       """ """


   def get_cell_info(self, **kwargs):
       """ """

