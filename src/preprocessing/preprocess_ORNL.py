# Based on Microsoft BatteryML repo
import os
import re
import pandas as pd

from tqdm.auto import tqdm
from typing import List
from pathlib import Path

from src.builders import PREPROCESSORS
from src.preprocessing.base import BasePreprocessor
from src.preprocessing.battery_data import BatteryData, TimeseriesData

@PREPROCESSORS.register()
class ORNLPreprocessor(BasePreprocessor):
    def __init__(self, name='oakridge', *, display_name = 'Oak Ridge National Lab', output_dir = None, silent = True):
        super().__init__(name, display_name=display_name, output_dir=output_dir, silent=silent)

    def process(self, parentdir='data/raw/oakridge/excel/', **kwargs):
        return super()._process_cells(inputdir=Path(parentdir))

    def get_timeseries_data(self, inputdir, cell) -> List[TimeseriesData]:
       """
       Get a list of TimeseriesData objects from the given filepath
       """
       expanded_cell = f"{cell}.xlsx"
       filename = os.path.join(inputdir, expanded_cell)
       df = pd.read_excel(filename)

       if cell.startswith('SNL_'):
           return get_snl_failure_data(df)

       elif 'TC1 (°C)' in df.columns:
           return get_tcn_failure_data(df)

       elif expanded_cell in _reltime_MAXC:
           return [TimeseriesData(time_in_s=df['reltime'], temperature_in_C=df['MAX [C]'])]
      
       elif expanded_cell in _reltime_Fn2:
           return [TimeseriesData(time_in_s=df['reltime'], temperature_in_C=df['Function 2 [C]'])]

       elif expanded_cell in _reltime_Fn3:
           return [TimeseriesData(time_in_s=df['reltime'], temperature_in_C=df['Function 3 [C]'])]
      
       elif cell == 'LCO4000mAh-0SOC-cell1':
           return [TimeseriesData(time_in_s=df['reltime'], temperature_in_C=df['Max temp (C) '])]
      
       elif cell == 'LCO_4Ah_30SOC_cell1_MAX':
           return [
               TimeseriesData(time_in_s=df['reltime'], temperature_in_C=df['3x3 temp (C)']),
               TimeseriesData(time_in_s=df['reltime'], temperature_in_C=df['Max temp (C) ']),
               TimeseriesData(time_in_s=df['reltime.1'], temperature_in_C=df['Function 2 [C]']),
           ]
      
       elif cell == 'NMC_10Ah_70SOC_cell2_MAX':
           return [TimeseriesData(time_in_s=df['reltime (s)'], temperature_in_C=df['Temperature [C]'])]

       elif cell == 'LCO6400mAh-40SOC-cell1-Load-Voltage':
           return [TimeseriesData(time_in_s=df['reltime'], temperature_in_C=df['Temp (C)'])]
      
       elif cell == 'LFP_15Ah_50SOC_cell2':
           return [TimeseriesData(time_in_s=df['Reltime'], temperature_in_C=df['c'])]
      
       else:
           temp_substrings = ['°C', '[C]', 'temp']
           time_col = next(col for col in df.columns if 'time' in str(col).lower())
           temp_col = next(col for col in df.columns if any(sub in str(col) for sub in temp_substrings))
           return [TimeseriesData(
               time_in_s=df[time_col],
               temperature_in_C=df[temp_col]
           )]

    def get_cell_info(self, cell, timeseries_data) -> BatteryData:
        org = 'snl' if "SNL_" in cell else 'oakridge'
        soc = int(match.group(1)) if (match := re.search(r'(\d+)S[O0]C', cell)) else None
        ah = float(match.group(1)) if (match := re.search(r'(\d+)Ah', cell)) else float(match.group(1))/1000. if (match := re.search(r'(\d+)mAh', cell)) else None
        cathode = next((cat for cat in BasePreprocessor.CATHODES if cat in cell), None)
        battery_type = next((bt for bt in BasePreprocessor.BATTERY_TYPES if bt in cell), None),

        return BatteryData(
            cell_id=cell,
            organization=org,
            timeseries_data=timeseries_data,
            is_healthy=False,
            state_of_charge=soc,
            battery_type=battery_type,
            anode_material='graphite',
            cathode_material=cathode,
            nominal_capacity_in_Ah=ah,
            form_factor='cylindrical_18650', # this was form_factor for SNL in BatteryML...
        )

def get_snl_failure_data(dataframe) -> List[TimeseriesData]:
   """
   Thermal runaway data for files starting with "SNL_" have the following columns:
   ['Test Time [s]', 'Displacement [mm]', 'Penetrator Force [mm]', 'vCell [V]', 'tAmbient [C]', 'TC1 near positive terminal [C]', 'TC2 near negative terminal [C]', 'TC3 bottom - bottom [C]', 'TC4 bottom - top [C]', 'TC5 above punch [C]', 'TC6 below punch [C]']
   """
   time = dataframe['Test Time [s]']
   tc1 = dataframe['TC1 near positive terminal [C]']
   tc2 = dataframe['TC2 near negative terminal [C]']
   tc3 = dataframe['TC3 bottom - bottom [C]']
   tc4 = dataframe['TC4 bottom - top [C]']
   tc5 = dataframe['TC5 above punch [C]']
   tc6 = dataframe['TC6 below punch [C]']

   return [
       TimeseriesData(time_in_s=time, temperature_in_C=tc1),
       TimeseriesData(time_in_s=time, temperature_in_C=tc2),
       TimeseriesData(time_in_s=time, temperature_in_C=tc3),
       TimeseriesData(time_in_s=time, temperature_in_C=tc4),
       TimeseriesData(time_in_s=time, temperature_in_C=tc5),
       TimeseriesData(time_in_s=time, temperature_in_C=tc6),
   ]

def get_tcn_failure_data(dataframe) -> List[TimeseriesData]:
   """
   Thermal runaway data for files with the following columns:
   ['Time (second)', 'Load (lb)', 'Voltage (V)', 'Unnamed: 3', 'Unnamed: 4', 'Time (sec)', 'Penetrator Force (N)', 'Cell Voltage (V)', 'Displacement (mm)', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Time (sec) ', 'TC1 (°C)', 'TC2 (°C)', 'TC3 (°C)', 'TC4 (°C)']
   """
   time = dataframe['Time (sec) ']
   tc1 = dataframe['TC1 (°C)']
   tc2 = dataframe['TC2 (°C)']
   tc3 = dataframe['TC3 (°C)']
   tc4 = dataframe['TC4 (°C)']

   return [
       TimeseriesData(time_in_s=time, temperature_in_C=tc1),
       TimeseriesData(time_in_s=time, temperature_in_C=tc2),
       TimeseriesData(time_in_s=time, temperature_in_C=tc3),
       TimeseriesData(time_in_s=time, temperature_in_C=tc4),
   ]

_reltime_MAXC = ['LCO_4000mAh-10SOC_cell2_MAX.xlsx', 'NMC_10000mAh-30SOC_cell1_MAX.xlsx', 'LCO_4000mAh-40SOC_cell2_MAX.xlsx', 'NMC_10000mAh-60SOC_cell1_MAX.xlsx', 'NMC_10000mAh-10SOC_cell1MAX.xlsx', 'NMC_10000mAh-90SOC_cell1_MAX.xlsx', 'NMC_10000mAh-40SOC_cell1_MAX.xlsx', 'LCO_4000mAh-50SOC_cell2_MAX.xlsx', 'LCO_4000mAh-0SOC_cell2_MAX.xlsx', 'NMC_10000mAh-70SOC_cell1_MAX.xlsx', 'NMC_10000mAh-50SOC_cell2_MAX.xlsx', 'LFP_15000mAh_10SOC_max.xlsx', 'NMC_10000mAh-20SOC_cell1_MAX.xlsx']
_reltime_Fn2 = ['LCO_4000mAh-50SOC_cell1_MAX.xlsx', 'NMC_10000mAh-0SOC_cell1_MAX.xlsx', 'LFP_15Ah_0SOC_MAX.xlsx', 'LFP_15Ah_100SOC_MAX.xlsx', 'LCO_4Ah_60SOC_cell1_MAX.xlsx', 'NMC_10000mAh-50SOC_cell1_MAX.xlsx', 'LFP_15Ah_100SOC_cell2_MAX.xlsx', 'LFP_15Ah_20SOC_cell1_MAX.xlsx', 'LCO_4Ah_20SOC_cell2_MAX.xlsx', 'LFP_15Ah_80SOC__cell2_MAX_2.xlsx', 'LCO_4Ah_70SOC_cell1_MAX.xlsx', 'LCO_4Ah_20SOC_cell1_MAX.xlsx', 'LCO_4Ah_60SOC_cell2_MAX.xlsx', 'LFP_15Ah_50SOC_cell1_MAX.xlsx', 'LCO_4Ah_10SOC_cell1_MAX.xlsx', 'LCO_4000mAh-40SOC_cell1_MAX.xlsx', 'NMC_10000mAh-80SOC_cell1_MAX.xlsx', 'LCO_4Ah_100SOC_cell1_MAX.xlsx', 'NMC_10000mAh-100SOC_cell1_MAX.xlsx', 'LFP_15Ah_60SOC_cell1_MAX.xlsx']
_reltime_Fn3 = ['LFP_15Ah_40SOC_cell1_MAX.xlsx', 'LFP_15Ah_60SOC_cell2_MAX.xlsx', 'LFP_15Ah_80SOC__cell1_MAX_2.xlsx', 'LCO_4Ah_30SOC_cell2_MAX.xlsx', 'LFP_15Ah_40SOC_cell2_MAX.xlsx']
