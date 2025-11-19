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
class CRPSPreprocessor(BasePreprocessor):
    def __init__(self, name='crps', *, display_name = 'Cell Report Physical Science', output_dir = None, silent = True):
        super().__init__(name, display_name=display_name, output_dir=output_dir, silent=silent)

    def get_timeseries_data(self, inputdir, cell) -> List[TimeseriesData]:
        """
        Get a list of TimeseriesData objects from the given filepath
        """
        expanded_cell = f"{cell}.csv"
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
            state_of_charge=soc,
            battery_type=battery_type,
            anode_material='graphite',
            cathode_material=cathode,
            nominal_capacity_in_Ah=ah,
            form_factor='pouch', # this was form_factor for SNL in BatteryML...
        )