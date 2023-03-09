import argparse
from math import ceil
from pathlib import Path

import numpy as np

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="An example script how to compare classifiers.")

    args = parser.parse_args()

    labels, train_features = event_types.nominal_labels_train_features()

    plot_scores = True
    plot_confusion_matrix = True
    plot_1d_conf_matrix = False
    n_types = 3

    Path("plots").mkdir(parents=True, exist_ok=True)

    models_to_compare = [
        "MLP_classifier",
        # 'MLP_relu_classifier',
        # 'MLP_logistic_classifier',
        # 'MLP_uniform_classifier',
        # 'MLP_small_classifier',
        # 'random_forest_classifier',
        # 'BDT_classifier',
        # 'ridge_classifier',
        # 'ridgeCV_classifier',  # unnecessary, same as the ridge classifier
        # 'SVC_classifier',  # Fails to evaluate for some reason, all SVC based fail
        # 'SGD_classifier',
        # 'Gaussian_process_classifier',  # Takes forever to train
        # 'bagging_svc_classifier',  # Fails to evaluate for some reason, all SVC based fail
        # 'bagging_dt_classifier',
        # 'oneVsRest_classifier',  # Fails to evaluate for some reason, all SVC based fail
        # 'gradient_boosting_classifier',
        # 'MLP_small_classifier',
    ]

    models_to_compare = ["{}_ntypes_{:d}".format(m, n_types) for m in models_to_compare]
    if len(models_to_compare) > 1:
        group_models_to_compare = np.array_split(
            models_to_compare, ceil(len(models_to_compare) / 5)
        )
    else:
        group_models_to_compare = [models_to_compare]

    for i_group, these_models_to_compare in enumerate(group_models_to_compare):

        trained_models = event_types.load_models(these_models_to_compare)
        dataset_names = event_types.extract_unique_dataset_names(trained_models)
        dtf_e_test = event_types.load_multi_test_dtfs(dataset_names)

        if plot_scores:
            plt, scores = event_types.plot_score_comparison(dtf_e_test, trained_models)
            plt.savefig(
                "plots/scores_features_classifier_n_types_{}_{}.pdf".format(n_types, i_group + 1)
            )
            plt.savefig(
                "plots/scores_features_classifier_n_types_{}_{}.png".format(n_types, i_group + 1)
            )
            plt.clf()
            event_types.save_scores(scores)

        if plot_confusion_matrix:

            event_types_lists = event_types.predicted_event_types(
                dtf_e_test, trained_models, n_types
            )
            for this_trained_model_name, this_event_types in event_types_lists.items():
                plt = event_types.plot_confusion_matrix(
                    this_event_types, this_trained_model_name, n_types
                )

                plt.savefig(
                    "plots/{}_confusion_matrix_n_types_{}.pdf".format(
                        this_trained_model_name, n_types
                    )
                )
                plt.savefig(
                    "plots/{}_confusion_matrix_n_types_{}.png".format(
                        this_trained_model_name, n_types
                    )
                )

                plt.clf()

                if plot_1d_conf_matrix:

                    plt = event_types.plot_1d_confusion_matrix(
                        this_event_types, this_trained_model_name, n_types
                    )

                    plt.savefig(
                        "plots/{}_1d_confusion_matrix_n_types_{}.pdf".format(
                            this_trained_model_name, n_types
                        )
                    )
                    plt.savefig(
                        "plots/{}_1d_confusion_matrix_n_types_{}.png".format(
                            this_trained_model_name, n_types
                        )
                    )

                    plt.clf()
