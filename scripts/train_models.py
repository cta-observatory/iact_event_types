import argparse

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Train event classes models." "Results are saved in the models directory.")
    )

    args = parser.parse_args()

    start_from_DL2 = False
    if start_from_DL2:
        # Prod5
        dl2_file_name = (
            "/lustre/fs22/group/cta/users/maierg/analysis/"
            "AnalysisData/prod5-Paranal-20deg-sq10-LL/EffectiveAreas/"
            "EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210921-V3/BDT.50h-V3.g20210921/"
            "gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root"
        )
        dtf = event_types.extract_df_from_dl2(dl2_file_name)
    else:
        dtf = event_types.load_dtf(
            [f"gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-{i}" for i in range(6)]
        )

    # For the training, make sure we do not use events with cut_class == 7 (non gamma-like events)
    # dtf = dtf[dtf['cut_class'] != 7].dropna()
    # try using cut_class == 7 (non gamma-like events)
    dtf = dtf.dropna()

    dtf_e = event_types.bin_data_in_energy(dtf, n_bins=20)

    dtf_e_train, dtf_e_test = event_types.split_data_train_test(
        dtf_e, test_size=0.75, random_state=777
    )

    labels, train_features = event_types.nominal_labels_train_features()

    all_models = event_types.define_regressors()
    selected_models = [
        # 'linear_regression',
        # 'BDT',
        # 'SVR',  # Do not use, performs bad and takes forever to apply
        # "random_forest_300_15",
        # "random_forest_300_20",
        # "random_forest_500_15",
        # "random_forest_500_20",
        # "random_forest_2000_5",
        "MLP_tanh",
        # 'MLP_relu',
        # 'MLP_logistic',
        # 'MLP_uniform',
        # 'MLP_lbfgs',  # Do not use, very slow to train
        # 'MLP_tanh_large',
        # 'BDT_small',
        # 'ridge',
        # 'linear_SVR',
        # 'SGD',
    ]

    models_to_train = dict()
    for this_model in selected_models:
        models_to_train[this_model] = dict()
        models_to_train[this_model]["train_features"] = train_features
        models_to_train[this_model]["labels"] = labels
        models_to_train[this_model]["model"] = all_models[this_model]
        models_to_train[this_model]["test_data_suffix"] = "default"

    trained_models = event_types.train_models(dtf_e_train, models_to_train)
    event_types.save_models(trained_models)
    event_types.save_test_dtf(dtf_e_test, suffix="default")
