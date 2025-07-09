# Based on Microsoft BatteryML repo
import os
import re
import pandas as pd

from tqdm.auto import tqdm
from typing import List
from pathlib import Path

from src.preprocessing.base import BasePreprocessor
from src.preprocessing.battery_data import BatteryData, TimeseriesData

class OXPreprocessor(BasePreprocessor):
   def __init__(self,
                name = 'oxford',
                output_dir = 'data/preprocessed/oxford/',
                silent = True):
       super().__init__(name, output_dir, silent)


   def process(self, parentdir='data/raw/oxford/', **kwargs):
       inputdir = Path(parentdir)
       cells = set(
           f.stem.split()[0]
           for f in inputdir.glob('*timeseries*')
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
               timeseries = get_timeseries_data(inputdir, cell)


           except:
               skip_batteries_num += 1
               continue


           # store data
           battery = get_cell_info(cell, timeseries)
           self.dump_single_file(battery)
           process_batteries_num += 1


           if not self.silent:
               tqdm.write(f'File: {battery.cell_id} dumped to pkl file')


       return process_batteries_num, skip_batteries_num
  
def get_timeseries_data(inputdir, cell) -> List[TimeseriesData]:
   """
   Get a list of TimeseriesData objects from the given filepath
   """
   expanded_cell = f"{cell}.csv"
   filename = os.path.join(inputdir, expanded_cell)
   df = pd.read_csv(filename)


   return [TimeseriesData(
       time_in_s=df['Test_Time (s)'],
       temperature_in_C=df['Cell_Temperature (C)']
   )]


def get_cell_info(cell, timeseries_data) -> BatteryData:
  
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
