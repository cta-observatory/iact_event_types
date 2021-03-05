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

    n_types = 3

    # dl2_file_name = (
    #     '/lustre/fs21/group/cta/users/maierg/analysis/AnalysisData/uploadDL2/'
    #     'Paranal_20deg/gamma_onSource.S.3HB9-FD_ID0.eff-0.root'
    # )
    dl2_file_name = (
        '/lustre/fs22/group/cta/users/maierg/analysis/AnalysisData/'
        'prod5-Paranal-20deg-sq08-LL/EffectiveAreas/'
        'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210921-V3/BDT.DL2.50h-V3.g20210921/'
        'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root'
    )
    dtf = event_types.extract_df_from_dl2(dl2_file_name)
    dtf_e = event_types.bin_data_in_energy(dtf)

    labels, train_features = event_types.nominal_labels_train_features()
    dtf_e = event_types.add_event_types_column(dtf_e, labels)

    dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e)

    all_models = event_types.define_classifiers()
    selected_models = [
        # 'MLP_classifier',
        # 'MLP_relu_classifier',
        # 'MLP_logistic_classifier',
        # 'MLP_uniform_classifier',
        'MLP_small_classifier',
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
        this_model_name = '{}_ntypes_{:d}'.format(this_model, n_types)
        models_to_train[this_model_name] = dict()
        models_to_train[this_model_name]['train_features'] = train_features
        models_to_train[this_model_name]['labels'] = 'event_type_{:d}'.format(n_types)
        models_to_train[this_model_name]['model'] = all_models[this_model]
        models_to_train[this_model_name]['test_data_suffix'] = 'classification'

    trained_models = event_types.train_models(
        dtf_e_train,
        models_to_train
    )
    event_types.save_models(trained_models)
    event_types.save_test_dtf(dtf_e_test, 'classification')
