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
    if start_from_DL2:
        # Prod5
        dl2_file_name = (
            '/lustre/fs22/group/cta/users/maierg/analysis/AnalysisData/'
            'prod5-Paranal-20deg-sq08-LL/EffectiveAreas/'
            'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210921-V3/BDT.DL2.50h-V3.g20210921/'
            'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root'
        )
        dtf = event_types.extract_df_from_dl2(dl2_file_name)
    else:
        dtf = event_types.load_dtf()

    dtf_e = event_types.bin_data_in_energy(dtf)

    labels, train_features = event_types.nominal_labels_train_features()

    all_models = event_types.define_regressors()

    test_data_frac = dict()
    test_data_frac['test_size_55p'] = 0.55
    test_data_frac['test_size_65p'] = 0.65
    # test_data_frac['test_size_75p'] = 0.75
    test_data_frac['test_size_85p'] = 0.85
    test_data_frac['test_size_95p'] = 0.95
    # test_data_frac['test_size_99p'] = 0.99

    for test_frac_name, test_frac in test_data_frac.items():
        print('Training with {:.0%}'.format(test_frac))
        models_to_train = dict()
        models_to_train[test_frac_name] = dict()
        models_to_train[test_frac_name]['train_features'] = train_features
        models_to_train[test_frac_name]['labels'] = labels
        models_to_train[test_frac_name]['model'] = all_models['MLP_small']
        models_to_train[test_frac_name]['test_data_suffix'] = test_frac_name

        dtf_e_train, dtf_e_test = event_types.split_data_train_test(dtf_e, test_size=test_frac)

        trained_models = event_types.train_models(
            dtf_e_train,
            models_to_train
        )
        event_types.save_models(trained_models)
        event_types.save_test_dtf(dtf_e_test, suffix=test_frac_name)
