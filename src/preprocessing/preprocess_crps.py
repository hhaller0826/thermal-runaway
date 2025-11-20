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
        df = pd.read_csv(filename)
        df = df.dropna(axis=1)
        assert df.size > 0

        return [TimeseriesData(
                time_in_s=df['Test_Time (s)'],
                h2_ppmo=df['H2 (ppmo)']
                temperature_in_C=df['Cell_Temperature (C)']
            )]

    def get_cell_info(self, cell, timeseries_data) -> BatteryData:
        org = 'snl' if "SNL_" in cell else 'CRPS'
        
        return BatteryData(
            cell_id=cell,
            organization=org,
            timeseries_data=timeseries_data,
            is_healthy=False,
        )