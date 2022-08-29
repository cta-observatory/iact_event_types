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
    start_from_DL2 = False

    labels, train_features = event_types.nominal_labels_train_features()

    all_models = event_types.define_regressors()

    models_to_train = dict()

    features = dict()
    features['all_features'] = train_features
    features['no_width'] = copy.copy(train_features)
    for feature_name in train_features:
        if feature_name.endswith('_width'):
            features['no_width'].remove(feature_name)
    features['no_length'] = copy.copy(train_features)
    for feature_name in train_features:
        if feature_name.endswith('_length'):
            features['no_length'].remove(feature_name)
    features['no_dispCombine'] = copy.copy(train_features)
    for feature_name in train_features:
        if feature_name.endswith('_dispCombine'):
            features['no_dispCombine'].remove(feature_name)
    features['old_features'] = copy.copy(train_features)
    for feature_name in train_features:
        if any(feature_name.endswith(suffix) for suffix in ['_width', '_length', '_dispCombine']):
            features['old_features'].remove(feature_name)

    for features_name, these_features in features.items():
        print(features_name, these_features)
        models_to_train[features_name] = dict()
        models_to_train[features_name]['train_features'] = these_features
        models_to_train[features_name]['labels'] = labels
        models_to_train[features_name]['model'] = all_models['random_forest']
        models_to_train[features_name]['test_data_suffix'] = 'default'

    if start_from_DL2:
        # Prod3b
        # dl2_file_name = (
        #     '/lustre/fs21/group/cta/users/maierg/analysis/AnalysisData/uploadDL2/'
        #     'Paranal_20deg/gamma_onSource.S.3HB9-FD_ID0.eff-0.root'
        # )
        # Prod5
        dl2_file_name = (
            '/lustre/fs22/group/cta/users/maierg/analysis/AnalysisData/'
            'prod5-Paranal-20deg-sq08-LL/EffectiveAreas/'
            'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210921-V3/BDT.DL2.50h-V3.g20210921/'
            'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root'
        )
        dtf = event_types.extract_df_from_dl2(dl2_file_name)
    else:
        dtf = event_types.load_dtf('gamma_onSource.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0')

    dtf_e = event_types.bin_data_in_energy(dtf)

    dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e)

    trained_models = event_types.train_models(
        dtf_e_train,
        models_to_train
    )
    event_types.save_models(trained_models)
    event_types.save_test_dtf(dtf_e_test)
