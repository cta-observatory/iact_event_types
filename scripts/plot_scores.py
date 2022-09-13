from math import ceil
from pathlib import Path

import numpy as np
from joblib import load

from event_types import event_types

if __name__ == "__main__":

    Path("plots").mkdir(parents=True, exist_ok=True)

    models_to_compare = [
        "random_forest_300_15",
        # "random_forest_300_20",
        # "random_forest_500_15",
        # "random_forest_500_20",
        # "random_forest_2000_5",
        "MLP_tanh",
        "MLP_relu",
        "MLP_logistic",
        "MLP_uniform",
        # 'MLP_lbfgs',  # Do not use, very slow to train
        # 'MLP_tanh_large',
    ]

    if len(models_to_compare) > 1:
        group_models_to_compare = np.array_split(
            models_to_compare, ceil(len(models_to_compare) / 5)
        )
    else:
        group_models_to_compare = [models_to_compare]

    for i_group, these_models_to_compare in enumerate(group_models_to_compare):

        scores = dict()
        for this_model in these_models_to_compare:
            scores[this_model] = load(f"scores/{this_model}.joblib")

        plt = event_types.plot_scores(scores)
        plt.savefig("plots/scores_comparison_{}.pdf".format(i_group + 1))
        plt.savefig("plots/scores_comparison_{}.png".format(i_group + 1))
        plt.clf()
