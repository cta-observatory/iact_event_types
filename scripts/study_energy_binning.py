import argparse

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train regressor models with various energy binnings."
            "Results are saved in the models directory."
        )
    )

    args = parser.parse_args()

    start_from_DL2 = False
    bins_off_axis_angle = [0, 1, 2, 3, 4, 5]
    layout_desc = "N.D25-4LSTs09MSTs-MSTN"  # LaPalma
    # layout_desc = 'S-M6C8aj-14MSTs37SSTs-MSTF'  # Paranal
    if start_from_DL2:
        dl2_file_names = [
            "../../data/LongFiles/North/"
            f"gamma_cone.{layout_desc}_ID0.eff-{i}.root" for i in bins_off_axis_angle
        ]
        dtf = event_types.extract_df_from_multiple_dl2(dl2_file_names)
    else:
        dtf = event_types.load_dtf(
            [f"gamma_cone.{layout_desc}_ID0.eff-{i}" for i in bins_off_axis_angle]
        )

    dtf = dtf.dropna()

    labels, train_features = event_types.nominal_labels_train_features()

    all_models = event_types.define_regressors()

    bin_numbers = [5, 10, 15, 20, 25]

    for bin_number in bin_numbers:
        print(f"Training with {bin_number} bins")
        name = f"{bin_number}_bins"
        models_to_train = dict()
        models_to_train[name] = dict()
        models_to_train[name]["train_features"] = train_features
        models_to_train[name]["labels"] = labels
        models_to_train[name]["model"] = all_models["MLP_tanh"]
        models_to_train[name]["test_data_suffix"] = name

        dtf_e = event_types.bin_data_in_energy(dtf, n_bins=bin_number)
        dtf_e_train, dtf_e_test = event_types.split_data_train_test(
            dtf_e, test_size=0.75, random_state=777
        )

        trained_models = event_types.train_models(dtf_e_train, models_to_train)
        event_types.save_models(trained_models)
        event_types.save_test_dtf(dtf_e_test, suffix=name)
