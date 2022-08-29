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
    
    gamma = [
        'gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0',
        'gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-1',
        'gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-2',
        'gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-3',
        'gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-4',
        'gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-5',
    ]
    electron = [
        'electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0',
        'electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-1',
        'electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-2',
        'electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-3',
        'electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-4',
        'electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-5',
    ]
    proton = [
        'proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0',
        'proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-1',
        'proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-2',
        'proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-3',
        'proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-4',
        'proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-5',
    ]

    particles = [gamma, electron, proton]

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
    model = trained_model[next(iter(trained_model))]
    suffix = model[next(iter(model))]['test_data_suffix']

    # Energy binning (in log10 TeV) used to separate event types. We use the binning usually used in
    # sensitivity curves, extended to lower and higher energies.
    event_type_log_e_bins = np.arange(-1.7, 2.5, 0.2)
    # Camera offset binning (in degrees) used to separate event types. The binning is a test, should be changed for
    # better performance.
    event_type_offset_bins = np.arange(0, 6, 1)
    event_type_offset_bins = np.append(event_type_offset_bins, 10)
    # Number of event types we want to classify our data:
    n_types = 3
    # This variable will store the event type partitioning container.
    event_type_partition = None

    for particle in particles:
        print('Exporting files: {}'.format(particle))
        dtf = event_types.load_all_dtfs(particle)
        print('Total number of events: {}'.format(len(dtf)))

        dtf_e = event_types.bin_data_in_energy(dtf, log_e_reco_bins=log_e_reco_bins)

        # We only separate training statistics in case of exporting a gamma_cone file.
        if particle is gamma:
            # Using a constant seed of 777, same as in the training/testing events
            dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e, random_state=777)
        else:
            dtf_e_test = dtf_e

        if particle is gamma:
            # To match the format needed by partition_event_types
            dtf_e_test_formatted = {suffix: dtf_e_test}
            # Add the predicted Y_diff to the data frame:
            dtf_test = event_types.add_predict_column(dtf_e_test_formatted, trained_model)
            # Divide the Y_diff distributions into a discrete number of event types (n_types)
            d_types, event_type_partition = event_types.partition_event_types(dtf_test, labels=labels,
                                                                              log_e_bins=event_type_log_e_bins,
                                                                              n_types=n_types, return_partition=True)
        else:
            # Calculate event types for proton and electron events, using the same event type thresholds as in the
            # gamma-like gammas:
            dtf_e_formatted = {suffix: dtf_e_test}
            dtf_test = event_types.add_predict_column(dtf_e_formatted, trained_model)
            d_types = event_types.partition_event_types(dtf_test, labels=labels, log_e_bins=event_type_log_e_bins,
                                                        n_types=n_types, event_type_bins=event_type_partition)
        # Start creating the event_type column within the original dataframe:
        dtf['event_type'] = -99

        for energy_key in dtf_e_test.keys():
            if 'gamma_cone' in dl2_file:
                dtf.loc[dtf_e_train[energy_key].index.values, 'event_type'] = -1
        dtf.loc[dtf_test[suffix].index.values, 'event_type'] = d_types[suffix][energy_key]['reco']

        print("A total of {} events will be written.".format(len(dtf['event_type'])))
        dtf_7 = dtf[dtf['cut_class']!=7]
        for event_type in [-99, 1, 2, 3]:
            print("A total of {} events of type {}".format(np.sum(dtf['event_type'] == event_type), event_type))
            print("A total of {} events of type {} for gamma-like events".format(
                np.sum(dtf_7['event_type'] == event_type), event_type
            ))

        with open(dl2_file.replace('.root', '.txt'), 'w') as txt_file:
            for value in dtf['event_type']:
                txt_file.write('{}\n'.format(value))
