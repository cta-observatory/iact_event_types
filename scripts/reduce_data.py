import argparse
from event_types import event_types

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Read the data from the DL2 root files and save them in pickle format'
            'for fast reading.'
        )
    )

    args = parser.parse_args()

    dl2_file_path = (
        '/lustre/fs22/group/cta/users/maierg/analysis/'
        'AnalysisData/prod5-Paranal-20deg-sq10-LL/EffectiveAreas/'
        'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210921-V3/BDT.50h-V3.g20210921/'
    )
    dl2_file_list = [
                     # Prod5 baseline (do not use anymore)
                     # 'electron_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                     # 'proton_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                     # 'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                     # 'gamma_cone.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',

                     # Prod5 Threshold (alpha?)
                     # 'electron_onSource.S-M6C5-14MSTs40SSTs-MSTF_ID0.eff-0.root',
                     # 'proton_onSource.S-M6C5-14MSTs40SSTs-MSTF_ID0.eff-0.root',
                     'gamma_onSource.S-M6C5-14MSTs40SSTs-MSTF_ID0.eff-0.root',
                     # 'gamma_cone.S-M6C5-14MSTs40SSTs-MSTF_ID0.eff-0.root',
                    ]

    for filename in dl2_file_list:
        dtf = event_types.extract_df_from_dl2('{}/{}'.format(dl2_file_path, filename))
        event_types.save_dtf(dtf, filename.replace('.root', ''))
