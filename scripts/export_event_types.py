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

    # List of DL2 files from which we want to export event types
    # IMPORTANT: Be aware that the gamma file goes first, as it will be used to calculate the event type threshlds.
    dl2_file_list = [
        # Prod-5 full CTA-S array:
        # 'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
        # 'electron_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
        # 'proton_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root'
        # Prod-5 full CTA-N array:
        # 'gamma_onSource.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0.root',
        # 'electron_onSource.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0.root',
        # 'proton_onSource.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0.root'
        # Prod-5 CTA-S threshold array:
        'gamma_onSource.S-M6C5-14MSTs40SSTs-MSTF_ID0.eff-0.root',
        'electron_onSource.S-M6C5-14MSTs40SSTs-MSTF_ID0.eff-0.root',
        'proton_onSource.S-M6C5-14MSTs40SSTs-MSTF_ID0.eff-0.root'
    ]
    if "gamma" not in dl2_file_list[0]:
        raise ValueError("The first DL2 file to analyze must be the gamma file, as we need it to comput the" +
                         "event type thresholds.")

    labels, train_features = event_types.nominal_labels_train_features()

    selected_model = 'MLP_tanh'

    trained_model = event_types.load_models([selected_model])
    # Get the energy binning from the trained model
    e_ranges = list(trained_model[next(iter(trained_model))].keys())
    # Sometimes they do not come in order... Here we fix that case.
    e_ranges.sort()
    log_e_reco_bins = np.log10(
        event_types.extract_energy_bins(e_ranges)
    )

    # Energy binning (in log10 TeV) used to separate event types. We use the binning usually used in
    # sensitivity curves, extended to lower and higher energies.
    event_type_log_e_bins = np.arange(-1.7, 2.5, 0.2)
    # Number of event types we want to classify our data:
    n_types = 3
    # This variable will store the event type partitioning container.
    event_type_partition = None

    for dl2_file in dl2_file_list:
        print('Exporting file: {}'.format(dl2_file))
        dtf = event_types.load_dtf(dl2_file.replace('.root', ''))
        print('Total number of events: {}'.format(len(dtf)))

        dtf_e = event_types.bin_data_in_energy(dtf, log_e_reco_bins=log_e_reco_bins)

        # We only separate training statistics in case of exporting a gamma_cone file.
        if 'gamma_cone' in dl2_file:
            # Using a constant seed of 777, same as in the training/testing events
            dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e, random_state=777)
        else:
            dtf_e_test = dtf_e

        if 'gamma' in dl2_file:
            # To match the format needed by partition_event_types
            dtf_e_test_formatted = {'default': dtf_e_test}
            # Add the predicted Y_diff to the data frame:
            dtf_test = event_types.add_predict_column(dtf_e_test_formatted, trained_model)
            # Divide the Y_diff distributions into a discrete number of event types (n_types)
            d_types, event_type_partition = event_types.partition_event_types(
                dtf_test,
                labels=labels,
                log_e_bins=event_type_log_e_bins,
                n_types=n_types,
                return_partition=True
            )
        else:
            # Calculate event types for proton and electron events, using the same event type thresholds as in the
            # gamma-like gammas:
            dtf_e_formatted = {'default': dtf_e_test}
            dtf_test = event_types.add_predict_column(dtf_e_formatted, trained_model)
            d_types = event_types.partition_event_types(
                dtf_test,
                labels=labels,
                log_e_bins=event_type_log_e_bins,
                n_types=n_types,
                event_type_bins=event_type_partition
            )
        # Start creating the event_type column within the original dataframe:
        dtf['event_type'] = -99
        # from IPython import embed
        # embed()
        for energy_key in dtf_e_test.keys():
            if 'gamma_cone' in dl2_file:
                dtf.loc[dtf_e_train[energy_key].index.values, 'event_type'] = -1
        dtf.loc[dtf_test['default'].index.values, 'event_type'] = dtf_test['default']['event_type']

        print("A total of {} events will be written.".format(len(dtf['event_type'])))
        for event_type in [-99, 1, 2, 3]:
            print("A total of {} events of type {}".format(np.sum(dtf['event_type'] == event_type), event_type))

        with open(dl2_file.replace('.root', '.txt'), 'w') as txt_file:
            for value in dtf['event_type']:
                txt_file.write('{}\n'.format(value))
