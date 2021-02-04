import numpy as np
import argparse
from pathlib import Path
from event_types import event_types

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'An example script how to load trained models.'
            'Remember not to use the data used to train these models.'
            'In future perhaps also the test data will be saved.'
        )
    )

    args = parser.parse_args()

    labels, train_features = event_types.nominal_labels_train_features()

    plot_predict_dist = True
    plot_scores = True
    plot_confusion_matrix = True
    n_types = 2

    Path('plots').mkdir(parents=True, exist_ok=True)
    dtf_e_test = event_types.load_test_dtf()

    # models_to_compare = [
    #     # 'linear_regression',
    #     # 'random_forest',
    #     # 'MLP',
    #     # 'MLP_relu',
    #     # 'MLP_logistic',
    #     # 'MLP_uniform',
    #     'MLP_small',
    #     # 'MLP_lbfgs',
    #     # 'BDT',
    #     # 'ridge',
    #     # 'SVR',
    #     # 'linear_SVR',
    #     # 'SGD',
    #     # 'MLP_small_less_vars',
    #     # 'MLP_meanPedvar_av_cross_O',
    # ]
    # models_to_compare = ['MLP_{}'.format(var) for var in train_features]
    # models_to_compare = ['All', 'features_1', 'features_2', 'features_3', 'features_4']
    # models_to_compare = ['All', 'features_5', 'features_6', 'features_7', 'features_8']
    models_to_compare = ['All', 'no_asym', 'no_tgrad_x', 'no_asym_tgrad_x']
    if len(models_to_compare) > 1:
        group_models_to_compare = np.array_split(
            models_to_compare,
            round(len(models_to_compare)/5)
        )
    else:
        group_models_to_compare = [models_to_compare]

    for i_group, these_models_to_compare in enumerate(group_models_to_compare):

        trained_models = event_types.load_models(these_models_to_compare)

        if plot_predict_dist:
            for this_trained_model_name, this_trained_model in trained_models.items():
                plt = event_types.plot_test_vs_predict(
                    dtf_e_test,
                    this_trained_model,
                    this_trained_model_name
                )

                plt.savefig('plots/{}_predict_dist.pdf'.format(this_trained_model_name))
                plt.savefig('plots/{}_predict_dist.png'.format(this_trained_model_name))

            plt.clf()

        if plot_scores:
            plt = event_types.plot_score_comparison(dtf_e_test, trained_models)
            plt.savefig('plots/scores_features_{}.pdf'.format(i_group + 1))
            plt.savefig('plots/scores_features_{}.png'.format(i_group + 1))
            plt.clf()

        if plot_confusion_matrix:

            event_types_lists = event_types.partition_event_types(
                dtf_e_test,
                trained_models,
                n_types
            )
            for this_trained_model_name, this_event_types in event_types_lists.items():
                plt = event_types.plot_confusion_matrix(
                    this_event_types,
                    this_trained_model_name,
                    n_types
                )

                plt.savefig('plots/{}_confusion_matrix_n_types_{}.pdf'.format(
                    this_trained_model_name,
                    n_types
                ))
                plt.savefig('plots/{}_confusion_matrix_n_types_{}.png'.format(
                    this_trained_model_name,
                    n_types
                ))

                plt = event_types.plot_1d_confusion_matrix(
                    this_event_types,
                    this_trained_model_name,
                    n_types
                )

                plt.savefig('plots/{}_1d_confusion_matrix_n_types_{}.pdf'.format(
                    this_trained_model_name,
                    n_types
                ))
                plt.savefig('plots/{}_1d_confusion_matrix_n_types_{}.png'.format(
                    this_trained_model_name,
                    n_types
                ))

                plt.clf()
