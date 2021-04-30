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
        'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
        'electron_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root',
        'proton_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root'
    ]
    if "gamma" not in dl2_file_list[0]:
        raise ValueError("The first DL2 file to analyze must be the gamma file, as we need it to comput the" +
                         "event type thresholds.")

    selected_model = 'MLP_logistic'

    trained_model = event_types.load_models([selected_model])
    # Get the energy binning from the trained model
    e_ranges = list(trained_model[next(iter(trained_model))].keys())
    # Sometimes they do not come in order... Here we fix that case.
    e_ranges.sort()
    log_e_reco_bins = np.log10(
        event_types.extract_energy_bins(e_ranges)
    )

    # This variable will store the event type partitioning container.
    event_type_partition = None

    for dl2_file in dl2_file_list:
        print('Exporting file: {}'.format(dl2_file))
        dtf = event_types.load_dtf(dl2_file.replace('.root', ''))
        print('Total number of events: {}'.format(len(dtf)))
        if 'gamma' in dl2_file:
            # We separate cut class 7, as it was not used for the model training, but will have associated
            # event types too:
            dtf_e = event_types.bin_data_in_energy(dtf[dtf['cut_class'] != 7], log_e_reco_bins=log_e_reco_bins)
            dtf_7_e = event_types.bin_data_in_energy(dtf[dtf['cut_class'] == 7], log_e_reco_bins=log_e_reco_bins)
        else:
            dtf_e = event_types.bin_data_in_energy(dtf, log_e_reco_bins=log_e_reco_bins)

        # Using a constant seed of 777, same as in the training/testing events
        if 'gamma' in dl2_file:
            dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e, random_state=777)
        else:
            dtf_e_test = dtf_e

        if 'gamma' in dl2_file:
            # To match the format needed by partition_event_types
            dtf_e_test_formatted = {'default': dtf_e_test}
            # Calculate event types for the gamma-like gamma events that was not used for training (test sample):
            d_types, event_type_partition = event_types.partition_event_types(
                dtf_e_test_formatted,
                trained_model,
                n_types=3,
                type_bins='equal statistics',
                return_partition=True
            )
            # Calculate event types for the subsample of non-gamma-like gamma events (cut class 7), using the
            # same event type thresholds as in the gamma-like gammas:
            dtf_e_7_formatted = {'default': dtf_7_e}
            d_types_7 = event_types.partition_event_types(
                dtf_e_7_formatted,
                trained_model,
                n_types=3,
                type_bins='equal statistics',
                event_type_bins=event_type_partition
            )
        else:
            # Calculate event types for proton and electron events, using the same event type thresholds as in the
            # gamma-like gammas:
            dtf_e_test_formatted = {'default': dtf_e_test}
            d_types = event_types.partition_event_types(
                dtf_e_test_formatted,
                trained_model,
                n_types=3,
                type_bins='equal statistics',
                event_type_bins=event_type_partition
            )
        # Start creating the event_type column within the original dataframe:
        dtf['event_type'] = -99
        for energy_key in dtf_e_test.keys():
            if 'gamma' in dl2_file:
                dtf.loc[dtf_e_train[energy_key].index.values, 'event_type'] = -1
                dtf.loc[dtf_7_e[energy_key].index.values, 'event_type'] = (
                    d_types_7[selected_model][energy_key]['reco']
                )
            dtf.loc[dtf_e_test[energy_key].index.values, 'event_type'] = (
                d_types[selected_model][energy_key]['reco']
            )
        # labels, train_features = event_types.nominal_labels_train_features()
        # plot_list = event_types.plot_matrix(dtf, train_features, labels, n_types=3, plot_events=20000)
        # for i, plot in enumerate(plot_list):
        #     if "gamma" in dl2_file:
        #         plot.savefig("gamma_features_{}.pdf".format(i))
        #     elif "proton" in dl2_file:
        #         plot.savefig("proton_features_{}.pdf".format(i))
        #     elif "electron" in dl2_file:
        #         plot.savefig("electron_features_{}.pdf".format(i))

        print("A total of {} events will be written.".format(len(dtf['event_type'])))
        print("A total of {} events were set as -99...".format(len(dtf[dtf['event_type'] == -99])))

        with open(dl2_file.replace('.root', '.txt'), 'w') as txt_file:
            for value in dtf['event_type']:
                txt_file.write('{}\n'.format(value))
