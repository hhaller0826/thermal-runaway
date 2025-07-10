
from .preprocess_ORNL import ORNLPreprocessor
from .preprocess_HealthyArchive import CALCEPreprocessor, HNEIPreprocessor, MichiganPreprocessor, OXPreprocessor, SNLPreprocessor, ULPurduePreprocessor

SUPPORTED_SOURCES = {
    'DATASETS': ['CALCE', 'HNEI', 'OX', 'ORNL', 'SNL', 'ULPurdue'],
}