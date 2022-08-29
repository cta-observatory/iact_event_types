import numpy as np
import argparse
from pathlib import Path
from scipy.stats import mstats
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

    n_types = 3
    type_bins = list(np.linspace(0, 1, n_types + 1))

    models_to_compare = [
        'MLP_tanh',
    ]

    trained_models = event_types.load_models(models_to_compare)
    dataset_names = event_types.extract_unique_dataset_names(trained_models)
    dtf_e_test = event_types.load_multi_test_dtfs(dataset_names)

    dtf_test = event_types.add_predict_column(dtf_e_test, trained_models)
    # Usual binning used in sensitivity curves, extended to lower and higher energies.
    log_e_reco_bins = np.arange(-2.1, 2.5, 0.2)
    # log_e_reco_bins = mstats.mquantiles(
    #     dtf_test['default']['log_reco_energy'].values,
    #     np.linspace(0, 1, 11)
    # )

    # dtf_test['event_type'] = -99
    event_types_now, event_type_thresholds = event_types.partition_event_types(dtf_test, labels=labels,
                                                                               log_e_bins=log_e_reco_bins,
                                                                               n_types=n_types, return_partition=True)
    from IPython import embed
    embed()
