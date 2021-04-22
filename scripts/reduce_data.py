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
    # Prod5
    dl2_file_list = [
                     'electron_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                     'proton_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                     'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                    ]
    dl2_file_path = '/home/thassan/Paquetes/last_version/iact_event_types/eventDisplay_files'

    for filename in dl2_file_list:
        dtf = event_types.extract_df_from_dl2("{}/{}".format(dl2_file_path, filename))
        event_types.save_dtf(dtf, filename.replace(".root", ""))
