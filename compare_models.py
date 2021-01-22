import argparse
from pathlib import Path
import event_classes

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'An example script how to load trained models.'
            'Remember not to use the data used to train these models.'
            'In future perhaps also the test data will be saved.'
        )
    )

    args = parser.parse_args()

    labels, train_features = event_classes.nominal_labels_train_features()

    #FIXME
    vars_to_remove = ['meanPedvar_Image', 'av_cross', 'av_crossO']
    for this_var in vars_to_remove:
        train_features.remove(this_var)

    # models_to_compare = [
    #     # 'linear_regression',
    #     # 'random_forest',
    #     # 'MLP',
    #     # 'MLP_relu',
    #     # 'MLP_logistic',
    #     # 'MLP_uniform',
    #     # 'MLP_small',
    #     # 'MLP_lbfgs',
    #     # 'BDT',
    #     # 'ridge',
    #     # 'SVR',
    #     # 'linear_SVR',
    #     # 'SGD',
    #     # 'MLP_small_less_vars',
    #     'MLP_meanPedvar_av_cross_O',
    # ]

    models_to_compare = [
        'MLP_loss_sum',
        'MLP_NTrig',
        'MLP_meanPedvar_Image',
        'MLP_av_fui',
        'MLP_av_cross',
        'MLP_av_crossO',
        'MLP_av_R',
        'MLP_av_ES',
        'MLP_MWR',
        'MLP_MLR',
    ]

    trained_models = event_classes.load_models(models_to_compare)
    dtf_e_test = event_classes.load_test_dtf()

    Path('plots').mkdir(parents=True, exist_ok=True)

    for this_trained_model_name, this_trained_model in trained_models.items():
        plt = event_classes.plot_test_vs_predict(
            dtf_e_test,
            this_trained_model,
            this_trained_model_name,
            # train_features,
            # labels
        )

        plt.savefig('plots/{}.pdf'.format(this_trained_model_name))

    plt.clf()

    plt = event_classes.plot_score_comparison(dtf_e_test, trained_models)
    plt.savefig('plots/compare_scores.pdf')
    plt.clf()
