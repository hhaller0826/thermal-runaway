# Based on Microsoft BatteryML repo
import os
import re
import pandas as pd
from typing import List
from pathlib import Path

from src.builders import PREPROCESSORS
from src.data import BatteryData, TimeseriesData

from .base import BasePreprocessor

class ULPreprocessor(BasePreprocessor):
    def __init__(self, name='UL_Final', *, display_name = 'Cell Report Physical Science', output_dir = None, silent = True):
        super().__init__(name, display_name=display_name, output_dir=output_dir, silent=silent)

    def _parentdir(self) -> str: """ """

    def process(self, parentdir=None, **kwargs):
        inputdir = Path(parentdir) if parentdir else Path(self._parentdir())
        return super()._process_cells(inputdir=inputdir)

    def get_timeseries_data(self, inputdir, cell) -> List[TimeseriesData]:
        """ 
        Get a list of TimeseriesData objects from the given filepath
        """
        expanded_cell = f"{cell}.csv"
        filename = os.path.join(inputdir, expanded_cell)
        df = pd.read_csv(filename)
        df = df.dropna(axis=1)
        assert df.size > 0

        df['zeros'] = 0
        return [TimeseriesData(
                time_in_s=df['time'],
                co_ppm=df['CO'],
                h2_ppmo=df['H2'] if 'H2' in df.columns else df['zeros'],
                co2_ppm=df['zeros'],
                temperature_in_C=df['temp'],
                description="H2 only"
            )]

    

# @PREPROCESSORS.register()
# class TROverchargePreprocessor(CRPSPreprocessor):
#     def _parentdir(self): return 'data/raw/crps/ThermalRunaway_Overcharge'
    
@PREPROCESSORS.register()
class ULNormPreprocessor(ULPreprocessor):
    def _parentdir(self): return 'data/raw/Final Datasets/Normal/UL_Final_Norm'

    def get_cell_info(self, cell, timeseries_data) -> BatteryData:
        return BatteryData(
            cell_id=cell,
            organization=self.name,
            timeseries_data=timeseries_data,
            is_healthy=True,

            has_gas=True,
        )

@PREPROCESSORS.register()
class ULTRPreprocessor(ULPreprocessor):
    def _parentdir(self): return 'data/raw/Final Datasets/TR/UL_Final_TR'

    def get_cell_info(self, cell, timeseries_data) -> BatteryData:
        return BatteryData(
            cell_id=cell,
            organization=self.name,
            timeseries_data=timeseries_data,
            is_healthy=False,

            has_gas=True,
        )

