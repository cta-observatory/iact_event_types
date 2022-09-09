import argparse
import copy

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train event classes model with various training features."
            "Results are saved in the models directory."
        )
    )

    args = parser.parse_args()

    labels, train_features = event_types.nominal_labels_train_features()

    features = dict()
    features["all_features"] = train_features
    features["no_reco_diff"] = copy.copy(train_features)
    for feature_name in train_features:
        if feature_name.endswith("log_reco_diff"):
            features["no_reco_diff"].remove(feature_name)
    # features['no_length'] = copy.copy(train_features)
    # for feature_name in train_features:
    #     if feature_name.endswith('_length'):
    #         features['no_length'].remove(feature_name)
    # features['no_dispCombine'] = copy.copy(train_features)
    # for feature_name in train_features:
    #     if feature_name.endswith('_dispCombine'):
    #         features['no_dispCombine'].remove(feature_name)
    # features['old_features'] = copy.copy(train_features)
    # for feature_name in train_features:
    #     if any(feature_name.endswith(suffix) for suffix in ['_width', '_length', '_dispCombine']):
    #         features['old_features'].remove(feature_name)

    models_to_train = dict()
    all_models = event_types.define_regressors()
    for features_name, these_features in features.items():
        print(features_name, these_features)
        models_to_train[features_name] = dict()
        models_to_train[features_name]["train_features"] = these_features
        models_to_train[features_name]["labels"] = labels
        models_to_train[features_name]["model"] = all_models["MLP_tanh"]
        models_to_train[features_name]["test_data_suffix"] = "default"

    dtf = event_types.load_dtf([f"gamma_cone.N.D25-4LSTs09MSTs-MSTN_ID0.eff-{i}" for i in range(6)])

    dtf = dtf.dropna()

    dtf_e = event_types.bin_data_in_energy(dtf, n_bins=20)

    dtf_e_train, dtf_e_test = event_types.split_data_train_test(
        dtf_e, test_size=0.75, random_state=777
    )

    trained_models = event_types.train_models(dtf_e_train, models_to_train)
    event_types.save_models(trained_models)
    event_types.save_test_dtf(dtf_e_test)
