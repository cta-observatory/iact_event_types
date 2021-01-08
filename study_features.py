import argparse
from pathlib import Path
import copy
import event_classes

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
    dtf = event_classes.extract_df_from_dl2(dl2_file_name)
    dtf_e = event_classes.bin_data_in_energy(dtf)

    dtf_e_train, dtf_e_test = event_classes.split_data_train_test(dtf_e)

    labels, train_features = event_classes.nominal_labels_train_features()

    all_models = event_classes.define_regressors()

    vars_to_remove = [
        'loss_sum',
        'NTrig',
        'meanPedvar_Image',
        'av_fui',
        'av_cross',
        'av_crossO',
        'av_R',
        'av_ES',
        'MWR',
        'MLR',
    ]

    models_to_train = dict()
    for this_var in vars_to_remove:
        _vars = copy.copy(train_features)
        _vars.remove(this_var)
        model_name = 'MLP_{}'.format(this_var)
        models_to_train[model_name] = dict()
        models_to_train[model_name]['train_features'] = _vars
        models_to_train[model_name]['labels'] = labels
        models_to_train[model_name]['model'] = all_models['MLP_small']

    trained_models = event_classes.train_models(
        dtf_e_train,
        models_to_train
    )
    event_classes.save_models(trained_models)
    event_classes.save_test_dtf(dtf_e_test)
