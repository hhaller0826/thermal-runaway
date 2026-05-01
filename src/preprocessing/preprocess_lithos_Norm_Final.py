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
class LithosNormPreprocessor(BasePreprocessor):
    def __init__(self, name='Lithos_Norm_Final', *, display_name = 'Lithos Healthy Data', output_dir = None, silent = True):
        super().__init__(name, display_name=display_name, output_dir=output_dir, silent=silent)

    def process(self, parentdir=None, **kwargs):
        inputdir = Path(parentdir) if parentdir else Path('data/raw/Final Datasets/Normal/Lithos_Final_Norm')
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

        return [TimeseriesData(
                time_in_s=pd.Series(list(range(len(df)))),
                co_ppm=df['CO'],
                h2_ppm=df['H2'],
                co2_ppm=df['CO2'],
                temperature_in_C=df['Temp'],
                description="BeforeTest" + cell[-1:]
            )]

    def get_cell_info(self, cell, timeseries_data) -> BatteryData:
        return BatteryData(
            cell_id=cell,
            organization=self.name,
            timeseries_data=timeseries_data,
            is_healthy=True,
            
            has_gas = True,
            description='; '.join([str(getattr(ts, 'description')) for ts in timeseries_data])
        )