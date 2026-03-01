from .preprocess_ORNL import ORNLPreprocessor
from .preprocess_HealthyArchive import (
    CALCEPreprocessor, 
    HNEIPreprocessor, 
    MichiganPreprocessor, 
    OXPreprocessor, 
    SNLPreprocessor, 
    ULPurduePreprocessor
)
from .preprocess_crps import TROverchargePreprocessor, TROverheatPreprocessor
from .preprocess_lithosBG import LithosPreprocessor
from .preprocess_synthGasTR import SynthGasTRPreprocessor

SUPPORTED_SOURCES = {
    'DATASETS': ['CALCE', 'HNEI', 'OX', 'ORNL', 'SNL', 'ULPurdue', 'SynthGasTR', 'Lithos'],
}