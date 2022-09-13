from copy import deepcopy
from math import ceil
from pathlib import Path

import numpy as np

from event_types import event_types

if __name__ == "__main__":

    labels, train_features = event_types.nominal_labels_train_features()

    plot_predict_dist = True
    plot_scores = True
    plot_confusion_matrix = False
    plot_1d_conf_matrix = False
    n_types = 3
    type_bins = list(np.linspace(0, 1, n_types + 1))
    # type_bins = [0, 0.2, 0.8, 1]

    Path("plots").mkdir(parents=True, exist_ok=True)

    models_to_compare = [
        "MLP_tanh",
        "MLP_tanh_trained_inner_offset_bin",
    ]

    if len(models_to_compare) > 1:
        group_models_to_compare = np.array_split(
            models_to_compare, ceil(len(models_to_compare) / 5)
        )
    else:
        group_models_to_compare = [models_to_compare]

    for i_group, these_models_to_compare in enumerate(group_models_to_compare):

        trained_models = event_types.load_models(these_models_to_compare)

        for inner_e_range, all_offsets_e_range in zip(
            trained_models["MLP_tanh_trained_inner_offset_bin"], trained_models["MLP_tanh"]
        ):
            trained_models["MLP_tanh_inner_offset_bin"][inner_e_range] = deepcopy(
                trained_models["MLP_tanh"][all_offsets_e_range]
            )
            trained_models["MLP_tanh_inner_offset_bin"][inner_e_range][
                "test_data_suffix"
            ] = "inner_offset_bin"

        dataset_names = event_types.extract_unique_dataset_names(trained_models)
        dtf_e_test = event_types.load_multi_test_dtfs(dataset_names)

        if plot_predict_dist:
            for this_trained_model_name, this_trained_model in trained_models.items():
                plt = event_types.plot_test_vs_predict(
                    dtf_e_test, this_trained_model, this_trained_model_name
                )

                plt.savefig("plots/{}_predict_dist.pdf".format(this_trained_model_name))
                plt.savefig("plots/{}_predict_dist.png".format(this_trained_model_name))

            plt.clf()

        if plot_scores:
            plt, scores = event_types.plot_score_comparison(dtf_e_test, trained_models)
            plt.savefig("plots/scores_features_{}.pdf".format(i_group + 1))
            plt.savefig("plots/scores_features_{}.png".format(i_group + 1))
            plt.clf()
            event_types.save_scores(scores)

        if plot_confusion_matrix:

            dtf_test = event_types.add_predict_column(dtf_e_test, trained_models)
            # Get the energy binning from the trained model
            e_ranges = list(trained_models[next(iter(trained_models))].keys())
            # Sometimes they do not come in order... Here we fix that case.
            e_ranges.sort()
            log_e_reco_bins = np.log10(event_types.extract_energy_bins(e_ranges))

            event_types_lists = event_types.partition_event_types(
                dtf_test, labels, log_e_reco_bins, n_types, type_bins
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
