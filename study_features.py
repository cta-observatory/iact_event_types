import argparse
from pathlib import Path
import copy
from event_types import event_types

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Train event classes model with various training features.'
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

    # vars_to_remove = train_features

    models_to_train = dict()
    # for this_var in vars_to_remove:
    #     _vars = copy.copy(train_features)
    #     _vars.remove(this_var)
    #     model_name = 'MLP_{}'.format(this_var)
    #     models_to_train[model_name] = dict()
    #     models_to_train[model_name]['train_features'] = _vars
    #     models_to_train[model_name]['labels'] = labels
    #     models_to_train[model_name]['model'] = all_models['MLP_small']

    features = dict()
    features['All'] = train_features
    features['features_5'] = ['img2_ang', 'log_SizeSecondMax', 'log_EmissionHeight', 'av_dist', 'av_cross', 'MWR', 'MLR', 'MSCW', 'MSCL', 'log_EmissionHeightChi2', 'log_DispDiff']
    features['features_6'] = features['features_5'] + ['MSWOL']
    features['features_7'] = features['features_6'] + ['MWOL']
    features['features_8'] = features['All'] + ['MSWOL'] + ['MWOL']

    for features_name, these_features in features.items():
        print(features_name, these_features)
        models_to_train[features_name] = dict()
        models_to_train[features_name]['train_features'] = these_features
        models_to_train[features_name]['labels'] = labels
        models_to_train[features_name]['model'] = all_models['MLP_small']

    trained_models = event_types.train_models(
        dtf_e_train,
        models_to_train
    )
    event_types.save_models(trained_models)
    event_types.save_test_dtf(dtf_e_test)
