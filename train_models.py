import argparse
import event_classes

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Train event classes models.'
            'Results are saved in the models directory.'
        )
    )
    parser.add_argument(
        '-l',
        '--labels',
        help='Labels to train on.',
        default='log_ang_diff'
    )

    args = parser.parse_args()

    dl2_file_name = '../DL2/gamma_onSource.S.3HB9-FD_ID0.eff-0.root'
    dtf = event_classes.extract_df_from_dl2(dl2_file_name)
    dtf_e = event_classes.bin_data_in_energy(dtf)

    dtf_e_train, dtf_e_test = event_classes.split_data_train_test(dtf_e)

    labels = args.labels
    train_features = [
        'log_reco_energy',
        'log_NTels_reco',
        'array_distance',
        'img2_ang',
        'log_SizeSecondMax',
        'MSCW',
        'MSCL',
        'log_EChi2S',
        'log_av_size'
    ]

    all_models = event_classes.define_regressors()
    models_to_train = {'linear_regression': all_models['linear_regression']}
    trained_models = event_classes.train_models(
        dtf_e_train,
        train_features,
        labels,
        models_to_train
    )
    event_classes.save_models(trained_models)
    # event_classes.save_test_dtf(dtf_e_test)

    for this_trained_model_name, this_trained_model in trained_models.items():
        plt = event_classes.plot_test_vs_predict(
            dtf_e_test,
            this_trained_model,
            this_trained_model_name,
            train_features,
            labels
        )

        plt.savefig('plots/{}.pdf'.format(this_trained_model_name))
