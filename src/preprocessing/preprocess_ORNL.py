# Based on Microsoft BatteryML repo
import os
import re
import pandas as pd
from typing import List
from pathlib import Path

from src.builders import PREPROCESSORS
from src.data import BatteryData, TimeseriesData

from .base import BasePreprocessor

@PREPROCESSORS.register()
class ORNLPreprocessor(BasePreprocessor):
    def __init__(self, name='oakridge', *, display_name = 'Oak Ridge National Lab', output_dir = None, silent = True):
        super().__init__(name, display_name=display_name, output_dir=output_dir, silent=silent)

    def process(self, parentdir=None, **kwargs):
        inputdir = Path(parentdir) if parentdir else Path('data/raw/oakridge/excel/')
        return super()._process_cells(inputdir=inputdir)

    def get_timeseries_data(self, inputdir, cell) -> List[TimeseriesData]:
        """
        Get a list of TimeseriesData objects from the given filepath
        """
        expanded_cell = f"{cell}.xlsx"
        filename = os.path.join(inputdir, expanded_cell)
        df = pd.read_excel(filename)
        df = df.dropna(axis=1, how='all')

        if cell.startswith('SNL_'):
            return get_snl_failure_data(df)

        elif any(f'TC{n} (°C)' in df.columns for n in range(1, 5)):
            return get_tcn_failure_data(df)

        elif expanded_cell in _reltime_MAXC:
            tsd = [_get_timeseries_data(df, time_col='reltime', temp_col='MAX [C]')]
        
        elif expanded_cell in _reltime_Fn2:
            tsd = [_get_timeseries_data(df, time_col='reltime', temp_col='Function 2 [C]')]

        elif expanded_cell in _reltime_Fn3:
            tsd = [_get_timeseries_data(df, time_col='reltime', temp_col='Function 3 [C]')]
        
        elif cell == 'LCO4000mAh-0SOC-cell1':
            tsd = [_get_timeseries_data(df, time_col='reltime', temp_col='Max temp (C) ')]
        
        elif cell == 'LCO_4Ah_30SOC_cell1_MAX':
            tsd = [
                _get_timeseries_data(df, time_col='reltime', temp_col='3x3 temp (C)'),
                _get_timeseries_data(df, time_col='reltime', temp_col='Max temp (C) '),
                _get_timeseries_data(df, time_col='reltime.1', temp_col='Function 2 [C]'),
            ]
        
        elif cell == 'NMC_10Ah_70SOC_cell2_MAX':
            tsd = [_get_timeseries_data(df, time_col='reltime (s)', temp_col='Temperature [C]')]

        elif cell == 'LCO6400mAh-40SOC-cell1-Load-Voltage':
            tsd = [_get_timeseries_data(df, time_col='reltime', temp_col='Temp (C)')]
        
        elif cell == 'LFP_15Ah_50SOC_cell2':
            tsd = [_get_timeseries_data(df, time_col='Reltime', temp_col='c')]
        
        else:
            temp_substrings = ['°C', '[C]', 'temp']
            time_col = next(col for col in df.columns if 'time' in str(col).lower())
            temp_col = next(col for col in df.columns if any(sub in str(col) for sub in temp_substrings))
            tsd = [_get_timeseries_data(df, time_col=time_col, temp_col=temp_col)]
        
        return [ts for ts in tsd if ts is not None]

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

            has_gas=False,            
            state_of_charge=soc,
            battery_type=battery_type,
            anode_material='graphite',
            cathode_material=cathode,
            nominal_capacity_in_Ah=ah,
            form_factor='pouch', # this was form_factor for SNL in BatteryML...
        )

def _get_timeseries_data(df: pd.DataFrame, time_col: str, temp_col: str) -> TimeseriesData:
    if any(col not in df.columns for col in [time_col, temp_col]): return
    if len(time:=df[time_col]) < 1 or len(temp:=df[temp_col]) < 1: return

    return TimeseriesData(time_in_s=time, temperature_in_C=temp)

def get_snl_failure_data(df: pd.DataFrame) -> List[TimeseriesData]:
    """
    Thermal runaway data for files starting with "SNL_" have the following columns:
    ['Test Time [s]', 'Displacement [mm]', 'Penetrator Force [mm]', 'vCell [V]', 'tAmbient [C]', 'TC1 near positive terminal [C]', 'TC2 near negative terminal [C]', 'TC3 bottom - bottom [C]', 'TC4 bottom - top [C]', 'TC5 above punch [C]', 'TC6 below punch [C]']
    """
    if (time:='Test Time [s]') not in df.columns: return []
    tc_dict = {
        'TC1 near positive terminal [C]': 'tc1, positive-terminal',
        'TC2 near negative terminal [C]': 'tc2, negative-terminal',
        'TC3 bottom - bottom [C]': 'tc3, bottom-bottom',
        'TC4 bottom - top [C]': 'tc4, bottom-top',
        'TC5 above punch [C]': 'tc5, above-punch',
        'TC6 below punch [C]': 'tc6, below-punch',
    }
    return [
       TimeseriesData(time_in_s=df[time], temperature_in_C=df[tc], description=tc_dict[tc]) 
       for tc in tc_dict if tc in df.columns
    ]

def get_tcn_failure_data(df: pd.DataFrame) -> List[TimeseriesData]:
    """
    Thermal runaway data for files with the following columns:
    ['Time (second)', 'Load (lb)', 'Voltage (V)', 'Unnamed: 3', 'Unnamed: 4', 'Time (sec)', 'Penetrator Force (N)', 'Cell Voltage (V)', 'Displacement (mm)', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Time (sec) ', 'TC1 (°C)', 'TC2 (°C)', 'TC3 (°C)', 'TC4 (°C)']
    """
    if (time:='Time (sec) ') not in df.columns: return []
    tc_dict = {
       'TC1 (°C)': 'tc1',
       'TC2 (°C)': 'tc2',
       'TC3 (°C)': 'tc3',
       'TC4 (°C)': 'tc4',
    }

    return [
       TimeseriesData(time_in_s=df[time], temperature_in_C=df[tc], description=tc_dict[tc]) 
       for tc in tc_dict if tc in df.columns
    ]

_reltime_MAXC = ['LCO_4000mAh-10SOC_cell2_MAX.xlsx', 'NMC_10000mAh-30SOC_cell1_MAX.xlsx', 'LCO_4000mAh-40SOC_cell2_MAX.xlsx', 'NMC_10000mAh-60SOC_cell1_MAX.xlsx', 'NMC_10000mAh-10SOC_cell1MAX.xlsx', 'NMC_10000mAh-90SOC_cell1_MAX.xlsx', 'NMC_10000mAh-40SOC_cell1_MAX.xlsx', 'LCO_4000mAh-50SOC_cell2_MAX.xlsx', 'LCO_4000mAh-0SOC_cell2_MAX.xlsx', 'NMC_10000mAh-70SOC_cell1_MAX.xlsx', 'NMC_10000mAh-50SOC_cell2_MAX.xlsx', 'LFP_15000mAh_10SOC_max.xlsx', 'NMC_10000mAh-20SOC_cell1_MAX.xlsx']
_reltime_Fn2 = ['LCO_4000mAh-50SOC_cell1_MAX.xlsx', 'NMC_10000mAh-0SOC_cell1_MAX.xlsx', 'LFP_15Ah_0SOC_MAX.xlsx', 'LFP_15Ah_100SOC_MAX.xlsx', 'LCO_4Ah_60SOC_cell1_MAX.xlsx', 'NMC_10000mAh-50SOC_cell1_MAX.xlsx', 'LFP_15Ah_100SOC_cell2_MAX.xlsx', 'LFP_15Ah_20SOC_cell1_MAX.xlsx', 'LCO_4Ah_20SOC_cell2_MAX.xlsx', 'LFP_15Ah_80SOC__cell2_MAX_2.xlsx', 'LCO_4Ah_70SOC_cell1_MAX.xlsx', 'LCO_4Ah_20SOC_cell1_MAX.xlsx', 'LCO_4Ah_60SOC_cell2_MAX.xlsx', 'LFP_15Ah_50SOC_cell1_MAX.xlsx', 'LCO_4Ah_10SOC_cell1_MAX.xlsx', 'LCO_4000mAh-40SOC_cell1_MAX.xlsx', 'NMC_10000mAh-80SOC_cell1_MAX.xlsx', 'LCO_4Ah_100SOC_cell1_MAX.xlsx', 'NMC_10000mAh-100SOC_cell1_MAX.xlsx', 'LFP_15Ah_60SOC_cell1_MAX.xlsx']
_reltime_Fn3 = ['LFP_15Ah_40SOC_cell1_MAX.xlsx', 'LFP_15Ah_60SOC_cell2_MAX.xlsx', 'LFP_15Ah_80SOC__cell1_MAX_2.xlsx', 'LCO_4Ah_30SOC_cell2_MAX.xlsx', 'LFP_15Ah_40SOC_cell2_MAX.xlsx']
