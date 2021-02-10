import argparse
from pathlib import Path
import copy
from event_types import event_types

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Read the data from the DL2 root files and save them in pickle format'
            'for fast reading.'
        )
    )

    args = parser.parse_args()
    # Prod5
    dl2_file_name = (
        '/lustre/fs22/group/cta/users/maierg/analysis/AnalysisData/'
        'prod5-Paranal-20deg-sq08-LL/EffectiveAreas/'
        'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210921-V3/BDT.DL2.50h-V3.g20210921/'
        'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root'
    )
    dtf = event_types.extract_df_from_dl2(dl2_file_name)
    event_types.save_dtf(dtf)
