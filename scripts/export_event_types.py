import argparse
from pathlib import Path
from event_types import event_types
from scipy.stats import mstats
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Train event classes models.'
            'Results are saved in the models directory.'
        )
    )

    args = parser.parse_args()

    dl2_file_list = [
                     'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                     'electron_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
                     'proton_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root']
    selected_models = [
        'MLP_small',
    ]
    trained_models = event_types.load_models(selected_models)
    log_e_reco_bins = None
    for dl2_file in dl2_file_list:
        print("Exporting file: {}".format(dl2_file))
        dtf = event_types.load_dtf(dl2_file.replace(".root", ""))
        print("Total number of events: {}".format(len(dtf)))
        if "gamma" in dl2_file:
            dtf_e, log_e_reco_bins = event_types.bin_data_in_energy(dtf, return_bins=True)
        else:
            dtf_e = event_types.bin_data_in_energy(dtf, log_e_reco_bins=log_e_reco_bins)
        print("Divided data into the following energy bins: {}".format(dtf_e.keys()))
        # It is using a constant seed of 0, which means that the same training/testing events will be used here.
        if "gamma" in dl2_file:
            dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e, random_state=777)
        else:
            dtf_e_test = dtf_e

        # To match the format needed by partition_event_types
        dtf_e_test_formatted = {'default': dtf_e_test}
        d_types = event_types.partition_event_types(dtf_e_test_formatted, trained_models, n_types=3,
                                                    type_bins='equal statistics')
        # We add the event type value to each energy bin within the test sample
        for energy_key in dtf_e_test.keys():
            dtf_e_test[energy_key]['event_type'] = d_types['MLP_small'][energy_key]['reco']

        # Start creating the event_type column within the original dataframe:
        dtf['event_type'] = -99
        for energy_key in dtf_e_test.keys():
            if "gamma" in dl2_file:
                dtf.loc[dtf_e_train[energy_key].index.values, 'event_type'] = -1
            dtf.loc[dtf_e_test[energy_key].index.values, 'event_type'] = d_types['MLP_small'][energy_key]['reco']

        with open(dl2_file.replace(".root", ".txt"), "w") as txt_file:
            for value in dtf['event_type']:
                txt_file.write("{}\n".format(value))
