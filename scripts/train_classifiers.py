import argparse

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Train event classes models." "Results are saved in the models directory.")
    )

    args = parser.parse_args()

    n_types = 3

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
            dl2_file_names = [path + f"gamma_onSource.{layout_desc}_ID0.eff-0.root"]
        else:
            dl2_file_names = [path + f"gamma_cone.{layout_desc}_ID0.eff-{i}.root" for i in bins_off_axis_angle]
        dtf = event_types.extract_df_from_multiple_dl2(dl2_file_names)
    else:
        if on_source:
            dtf = event_types.load_dtf([f"gamma_onSource.{layout_desc}_ID0.eff-0"])
        else:
            dtf = event_types.load_dtf(
                [f"gamma_cone.{layout_desc}_ID0.eff-{i}" for i in bins_off_axis_angle]
            )

    dtf_e = event_types.bin_data_in_energy(dtf)

    labels, train_features = event_types.nominal_labels_train_features()
    dtf_e = event_types.add_event_types_column(dtf_e, labels)

    dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e)

    all_models = event_types.define_classifiers()
    selected_models = [
        "MLP_classifier",
        # 'MLP_relu_classifier',
        # 'MLP_logistic_classifier',
        # 'MLP_uniform_classifier',
        # 'BDT_classifier',
        # 'random_forest_classifier',
        # 'ridge_classifier',
        # # 'ridgeCV_classifier', # unnecessary, same as the ridge classifier
        # 'SVC_classifier',  # Fails to evaluate for some reason, all SVC based fail
        # 'SGD_classifier',
        # 'Gaussian_process_classifier',  # Takes forever to train
        # 'bagging_svc_classifier',  # Fails to evaluate for some reason, all SVC based fail
        # 'bagging_dt_classifier',
        # 'oneVsRest_classifier',  # Fails to evaluate for some reason
        # 'gradient_boosting_classifier',
    ]

    models_to_train = dict()
    for this_model in selected_models:
        this_model_name = "{}_ntypes_{:d}".format(this_model, n_types)
        models_to_train[this_model_name] = dict()
        models_to_train[this_model_name]["train_features"] = train_features
        models_to_train[this_model_name]["labels"] = "event_type_{:d}".format(n_types)
        models_to_train[this_model_name]["model"] = all_models[this_model]
        models_to_train[this_model_name]["test_data_suffix"] = "classification"

    trained_models = event_types.train_models(dtf_e_train, models_to_train)
    event_types.save_models(trained_models)
    event_types.save_test_dtf(dtf_e_test, "classification")
