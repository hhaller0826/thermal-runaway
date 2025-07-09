# Based on Microsoft BatteryML repo
import pickle
import pandas as pd

from typing import List

class TimeseriesData:
   def __init__(self,
                *,
                time_in_s: List[float],
                temperature_in_C: List[float] = None,
                **kwargs):
       self.time_in_s = time_in_s
       self.temperature_in_C = temperature_in_C
      
       self.additional_data = {}
       for key, val in kwargs.items():
           self.additional_data[key] = val

   def to_dict(self):
       return {
           'time_in_s': self.time_in_s,
           'temperature_in_C': self.temperature_in_C,
           **self.additional_data
       }
  
   def to_df(self):
       return pd.DataFrame(self.to_dict())
  
   def display(self, n=None):
       return pd.DataFrame({'Time (s)': self.time_in_s, 'Temp (°C)': self.temperature_in_C}).head(n)
  
   @staticmethod
   def load(obj):
       return TimeseriesData(**obj)

class BatteryData:
   # NOTE: We could split this into cycles for the healthy data but since
   # the unhealthy data isn't cycled I figured there prob won't be a use case
   def __init__(self,
                cell_id: str,
                *,
                organization: str = None,
                timeseries_data: TimeseriesData = None,
                is_healthy: bool = True,
                state_of_charge: float = None,
                battery_type: str = None,
                anode_material: str = None,
                cathode_material: str = None,
                electrolyte_material: str = None,
                nominal_capacity_in_Ah: float = None, # TODO: is this the same as Ah in file names
                form_factor: str = None, # TODO: ???
                description: str = None,
                **kwargs):
       self.cell_id = cell_id
       self.organization = organization
      
       if timeseries_data != None:
           self.timeseries_data = timeseries_data
       elif 'time_in_s' in kwargs:
           self.timeseries_data = TimeseriesData(
               time_in_s=kwargs['time_in_s'],
               temperature_in_C=kwargs.get('temperature_in_C')
           )
       self.is_healthy = is_healthy

       self.state_of_charge = state_of_charge
       self.battery_type = battery_type
       self.anode_material = anode_material
       self.cathode_material = cathode_material
       self.electrolyte_material = electrolyte_material
       self.nominal_capacity_in_Ah = nominal_capacity_in_Ah
       self.form_factor = form_factor
       self.description = description

       for key, val in kwargs.items():
           setattr(self, key, val)
  
   def to_dict(self):
       result = {}
       for key, val in self.__dict__.items():
           if not callable(val) and not key.startswith('_'):
               if key == 'cycle_data' or 'protocol' in key:
                   result[key] = [cell.to_dict() for cell in val]
               elif hasattr(val, 'to_dict'):
                   result[key] = val.to_dict()
               else:
                   result[key] = val
       return result
  
   def to_df(self):
       return pd.DataFrame(self.timeseries_data.to_dict())
  
   def dump(self, path):
       with open(path, 'wb') as fout:
           pickle.dump(self.to_dict(), fout)

   def print_description(self):
       print(f'**************description of battery cell {self.cell_id}**************')
       for key, val in self.__dict__.items():
           if key == 'cycle_data':
               print(f'cycle length: {len(val)}')
           elif val != None:
               print(f'{key}: {val}')

   @staticmethod
   def load(path):
       with open(path, 'rb') as fin:
           obj = pickle.load(fin)
       return BatteryData(**obj)
