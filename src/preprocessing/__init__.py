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

from .preprocess_crps_Final import CRPSTRPreprocessor, CRPSNormPreprocessor
from .preprocess_lithos_Norm_Final import LithosNormPreprocessor
from .preprocess_lithos_TR_Final import LithosTRPreprocessor
from .preprocess_lithos2_Norm_Final import Lithos2NormPreprocessor
from .preprocess_lithos2_TR_Final import Lithos2TRPreprocessor
from .preprocess_UL_Final import ULNormPreprocessor, ULTRPreprocessor

SUPPORTED_SOURCES = {
    'DATASETS': ['CALCE', 'HNEI', 'OX', 'ORNL', 'SNL', 'ULPurdue', 
                'CRPSTR', 'CRPSNorm', 'LithosNorm', 'LithosTR',
                'Lithos2Norm', 'Lithos2TR', 'ULNorm', 'ULTR'],
}