import argparse
from pathlib import Path
from event_types import event_types

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Train event classes models.'
            'Results are saved in the models directory.'
        )
    )

    args = parser.parse_args()

    # dl2_file_name = '/lustre/fs21/group/cta/users/maierg/analysis/AnalysisData/uploadDL2/Paranal_20deg/gamma_cone.S.3HB9-FD_ID0.eff-0.root'
    dl2_file_name = '/lustre/fs21/group/cta/users/maierg/analysis/AnalysisData/uploadDL2/Paranal_20deg/gamma_onSource.S.3HB9-FD_ID0.eff-0.root'
    dtf = event_types.extract_df_from_dl2(dl2_file_name)
    dtf_e = event_types.bin_data_in_energy(dtf)

    dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e)

    labels, train_features = event_types.nominal_labels_train_features()

    all_models = event_types.define_regressors()
    selected_models = [
        # 'linear_regression',
        # 'random_forest',  # Do not use, performs bad and takes lots of disk space
        # 'MLP',
        # 'MLP_relu',
        # 'MLP_logistic',
        # 'MLP_uniform',
        'MLP_small',
        # 'MLP_lbfgs',
        # 'BDT',  # Do not use, performs bad and takes lots of disk space
        # 'ridge',
        # 'SVR',  # Do not use, performs bad and takes forever to apply
        # 'linear_SVR',
        # 'SGD',
    ]

    models_to_train = dict()
    for this_model in selected_models:
        models_to_train[this_model] = dict()
        models_to_train[this_model]['train_features'] = train_features
        models_to_train[this_model]['labels'] = labels
        models_to_train[this_model]['model'] = all_models[this_model]

    trained_models = event_types.train_models(
        dtf_e_train,
        models_to_train
    )
    event_types.save_models(trained_models)
    event_types.save_test_dtf(dtf_e_test)
