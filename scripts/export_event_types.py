import argparse

import numpy as np

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train event classes models." "Results are saved in the models directory."
    )

    args = parser.parse_args()

    gamma = [
        # CTA-N
        "gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0",
        "gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-1",
        "gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-2",
        "gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-3",
        "gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-4",
        "gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-5",
        # CTA-S
        # 'gamma_cone.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-0',
        # 'gamma_cone.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-1',
        # 'gamma_cone.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-2',
        # 'gamma_cone.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-3',
        # 'gamma_cone.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-4',
        # 'gamma_cone.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-5',
    ]
    electron = [
        # CTA-N
        "electron_onSource.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0",
        "electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-1",
        "electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-2",
        "electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-3",
        "electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-4",
        "electron.N.D25-4LSTs09MSTs-MSTN_ID0.eff-5",
        # CTA-S
        # 'electron_onSource.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-0',
        # 'electron.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-1',
        # 'electron.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-2',
        # 'electron.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-3',
        # 'electron.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-4',
        # 'electron.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-5',
    ]
    proton = [
        # CTA-N
        "proton_onSource.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0",
        "proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-1",
        "proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-2",
        "proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-3",
        "proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-4",
        "proton.N.D25-4LSTs09MSTs-MSTN_ID0.eff-5",
        # CTA-S
        # 'proton_onSource.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-0',
        # 'proton.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-1',
        # 'proton.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-2',
        # 'proton.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-3',
        # 'proton.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-4',
        # 'proton.S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF_ID0.eff-5',
    ]

    particles = [gamma, electron, proton]

    labels, train_features = event_types.nominal_labels_train_features()

    selected_model = "MLP_tanh"

    trained_model = event_types.load_models([selected_model])
    # Get the energy binning from the trained model
    e_ranges = list(trained_model[next(iter(trained_model))].keys())
    # Sometimes they do not come in order... Here we fix that case.
    e_ranges.sort()
    log_e_reco_bins = np.log10(event_types.extract_energy_bins(e_ranges))
    model = trained_model[next(iter(trained_model))]
    suffix = model[next(iter(model))]["test_data_suffix"]

    # Energy binning (in log10 TeV) used to separate event types. We use the binning usually used in
    # sensitivity curves, extended to lower and higher energies.
    event_type_log_e_bins = np.arange(-1.7, 2.5, 0.2)
    # Camera offset binning (in degrees) used to separate event types. The binning is a test,
    # should be changed for better performance.
    # event_type_offset_bins = np.arange(0, 5, 1)
    # event_type_offset_bins = np.append(event_type_offset_bins, 10)
    n_offset_bins = 5

    # Number of event types we want to classify our data:
    n_types = 3
    # This variable will store the event type partitioning container.
    event_type_partition = None

    for particle in particles:
        print("Exporting files: {}".format(particle))
        dtf = event_types.load_dtf(particle)
        print("Total number of events: {}".format(len(dtf)))

        # We only bin in energy here to have the same file as in the training and be able to have
        # the exact same training/testing events' separation.
        dtf_e = event_types.bin_data_in_energy(dtf, log_e_reco_bins=log_e_reco_bins)

        # We only separate training statistics in case of exporting a gamma_cone file.
        if particle is gamma:
            # Using a constant seed of 777, same as in the training/testing events.
            dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e, random_state=777)
        else:
            # bin_data_in_energy creates a big offset bin that has to be removed before
            # add_predict_column. We fix it here.
            dtf_e_test = dict()
            for energy_key in dtf_e.keys():
                dtf_e_test[energy_key] = dtf_e[energy_key][next(iter(dtf_e[energy_key]))]

        # To match the format needed by partition_event_types
        dtf_e_test_formatted = {suffix: dtf_e_test}
        # Add the predicted Y_diff to the data frame:
        dtf_test = event_types.add_predict_column(dtf_e_test_formatted, trained_model)

        if particle is gamma:

            # Divide the Y_diff distributions into a discrete number of event types (n_types)
            d_types, event_type_partition = event_types.partition_event_types(
                dtf_test,
                labels=labels,
                log_e_bins=event_type_log_e_bins,
                n_offset_bins=n_offset_bins,
                n_types=n_types,
                return_partition=True,
            )

        else:
            # Calculate event types for proton and electron events, using the same event type
            # thresholds as in the gammas:
            d_types = event_types.partition_event_types(
                dtf_test,
                labels=labels,
                log_e_bins=event_type_log_e_bins,
                n_offset_bins=n_offset_bins,
                n_types=n_types,
                event_type_bins=event_type_partition,
            )

        # Start creating the event_type column within the original dataframe,
        # setting the default value: -3.
        # There shouldn't be many -3s at the end. If that's the case, it must be understood.
        dtf["event_type"] = -3

        if particle is gamma:
            # Assign type -1 to the events used for the training.
            for energy_key in dtf_e_train.keys():
                dtf.loc[dtf_e_train[energy_key].index.values, "event_type"] = -1

        dtf.loc[dtf_test[suffix].index.values, "event_type"] = dtf_test[suffix]["event_type"]

        print("A total of {} events will be written.".format(len(dtf["event_type"])))
        dtf_7 = dtf[dtf["cut_class"] != 7]
        for event_type in [-3, -2, -1, 1, 2, 3]:
            print(
                "A total of {} events of type {}".format(
                    np.sum(dtf["event_type"] == event_type), event_type
                )
            )
            print(
                "A total of {} events of type {} for gamma-like events".format(
                    np.sum(dtf_7["event_type"] == event_type), event_type
                )
            )
        # Summary:
        # Types 1, 2 and 3 are actual reconstructed event types, where 1 is the best type and
        # all three should have similar statistics.
        # Type -1 is assigned to the train sample.
        # Type -2 is the default value when assigning event types to the test sample in
        # partition_event_types_energy and partition_event_types_energy_and_offset. As that function
        # works for each bin of energy and offset, events outside these bins will have type -2.
        # Type -3 is the default value for the complete tables of each particle.
        # i.e. events with no other type assigned.

        file_name = (
            particle[0].replace(".eff-0", ".txt").replace("_cone", "").replace("_onSource", "")
        )
        with open(file_name, "w") as txt_file:
            for value in dtf["event_type"]:
                txt_file.write("{}\n".format(value))
