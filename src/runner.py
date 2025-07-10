from src.preprocessing import SUPPORTED_SOURCES
from src.builders import PREPROCESSORS

def preprocess(orgs_to_skip=[], silent=False):
    for org in SUPPORTED_SOURCES['DATASETS']:
        if org not in orgs_to_skip:
            config = {
                'name': f'{org}Preprocessor'
            }
            processor = PREPROCESSORS.build(config)
            pr, sk = processor.process()
            if not silent: print(f'{pr} processed, {sk} skipped\n')