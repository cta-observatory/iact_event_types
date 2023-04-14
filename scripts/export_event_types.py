import argparse

import numpy as np

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Obtain the event types for all the events in the DL2 files."
    )

    args = parser.parse_args()

    start_from_DL2 = False
    on_source = False
    bins_off_axis_angle = [0, 1, 2, 3, 4, 5]
    location = "North"
    if location == "North":  # LaPalma
        layout_desc = "N.D25-4LSTs09MSTs-MSTN"
        path = "../../data/LongFiles/North/"
    elif location == "South":  # Paranal
        layout_desc = 'S-M6C8aj-14MSTs37SSTs-MSTF'
        path = "../../data/LongFiles/South/"
    else:
        raise ValueError("Location not recognized. Must be North or South.")

    if start_from_DL2:
        if on_source:
            gamma = [path + f"gamma_onSource.{layout_desc}_ID0.eff-0.root"]
            proton = [path + f"proton_onSource.{layout_desc}_ID0.eff-0.root"]
            electron = [path + f"electron_onSource.{layout_desc}_ID0.eff-0.root"]
        else:
            gamma = [path + f"gamma_cone.{layout_desc}_ID0.eff-{i}.root" for i in bins_off_axis_angle]
            proton = [path + f"proton.{layout_desc}_ID0.eff-{i}.root" for i in bins_off_axis_angle]
            electron = [path + f"electron.{layout_desc}_ID0.eff-{i}.root" for i in bins_off_axis_angle]
    else:
        if on_source:
            gamma = [f"gamma_onSource.{layout_desc}_ID0.eff-0"]
            proton = [f"proton_onSource.{layout_desc}_ID0.eff-0"]
            electron = [f"electron_onSource.{layout_desc}_ID0.eff-0"]
        else:
            gamma = [f"gamma_cone.{layout_desc}_ID0.eff-{i}" for i in bins_off_axis_angle]
            electron = [f"electron.{layout_desc}_ID0.eff-{i}" for i in bins_off_axis_angle]
            proton = [f"proton.{layout_desc}_ID0.eff-{i}" for i in bins_off_axis_angle]

    particles = [gamma, electron, proton]

    labels, train_features = event_types.nominal_labels_train_features()
    # Select the model we want to use to classify the events.
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
    if on_source:
        n_offset_bins = 1
    else:
        n_offset_bins = 6

    # Number of event types we want to classify our data:
    n_types = 3
    # This variable will store the event type partitioning container.
    event_type_partition = None

    for particle in particles:
        print("Exporting files: {}".format(particle))
        if start_from_DL2:
            dtf = event_types.extract_df_from_multiple_dl2(particle)
        else:
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
            particle[0].replace(".eff-0", ".txt").replace(".root", "")
        )
        with open(file_name, "w") as txt_file:
            for value in dtf["event_type"]:
                txt_file.write("{}\n".format(value))
