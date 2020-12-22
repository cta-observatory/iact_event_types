import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import Angle
from astropy import units as u
from astropy.table import Table
import pandas as pd
import seaborn as sns
from pathlib import Path
from joblib import dump, load
from scipy.stats import mstats
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn import model_selection, preprocessing, feature_selection, ensemble, metrics
from sklearn.pipeline import make_pipeline


def extract_df_from_dl2(root_filename):
    '''
    Extract a Pandas DataFrame from a ROOT DL2 file.
    Selects all events surviving gamma/hadron cuts from the DL2 file.
    No direction cut is applied on the sample. TODO: should this be an option or studied further?
    The list of variables included in the DataFrame is subject to change.
    TODO: Study further possible variables to use.

    Parameters
    ----------
    root_filename: str or Path
        The location of the DL2 root file name from which to extract the DF.
        TODO: Allow using several DL2 files (in a higher level function?)

    Returns
    -------
    A pandas DataFrame with variables to use in the regression, after cuts.
    '''

    branches = [
        'MCze',
        'MCaz',
        'Ze',
        'Az',
        'size',
        'ErecS',
        'NImages',
        'Xcore',
        'Ycore',
        'Xoff',
        'Yoff',
        'img2_ang',
        'EChi2S',
        'SizeSecondMax',
        'NTelPairs',
        'MSCW',
        'MSCL'
    ]

    particle_file = uproot.open(root_filename)
    data = particle_file['data']
    cuts = particle_file['fEventTreeCuts']

    data_arrays = data.arrays(expressions=branches, library='np')
    cuts_arrays = cuts.arrays(expressions='CutClass', library='np')

    # Cut 1: Events surviving gamma/hadron separation and direction cuts:
    mask_gamma_like_and_direction = cuts_arrays['CutClass'] == 5

    # Cut 2: Events surviving gamma/hadron separation cut and not direction cut:
    mask_gamma_like_no_direction = cuts_arrays['CutClass'] == 0

    gamma_like_events = np.logical_or(mask_gamma_like_and_direction, mask_gamma_like_no_direction)

    # Variables for regression:
    mc_alt = (90 - data_arrays['MCze'][gamma_like_events]) * u.deg
    mc_az = (data_arrays['MCaz'][gamma_like_events]) * u.deg
    reco_alt = (90 - data_arrays['Ze'][gamma_like_events]) * u.deg
    reco_az = (data_arrays['Az'][gamma_like_events]) * u.deg
    # Angular separation bewteen the true vs reconstructed direction
    ang_diff = angular_separation(
        mc_az,  # az
        mc_alt,  # alt
        reco_az,
        reco_alt,
    )

    # Variables for training:
    av_size = [np.average(sizes) for sizes in data_arrays['size'][gamma_like_events]]
    reco_energy = data_arrays['ErecS'][gamma_like_events]
    NTels_reco = data_arrays['NImages'][gamma_like_events]
    x_cores = data_arrays['Xcore'][gamma_like_events]
    y_cores = data_arrays['Ycore'][gamma_like_events]
    array_distance = np.sqrt(x_cores**2. + y_cores**2.)
    x_off = data_arrays['Xoff'][gamma_like_events]
    y_off = data_arrays['Yoff'][gamma_like_events]
    camera_offset = np.sqrt(x_off**2. + y_off**2.)
    img2_ang = data_arrays['img2_ang'][gamma_like_events]
    EChi2S = data_arrays['EChi2S'][gamma_like_events]
    SizeSecondMax = data_arrays['SizeSecondMax'][gamma_like_events]
    NTelPairs = data_arrays['NTelPairs'][gamma_like_events]
    MSCW = data_arrays['MSCW'][gamma_like_events]
    MSCL = data_arrays['MSCL'][gamma_like_events]

    # Build astropy table:
    t = Table()
    t['log_ang_diff'] = np.log10(ang_diff.value)
    t['log_av_size'] = np.log10(av_size)
    t['log_reco_energy'] = np.log10(reco_energy)
    t['log_NTels_reco'] = np.log10(NTels_reco)
    t['array_distance'] = array_distance
    t['img2_ang'] = img2_ang
    t['log_EChi2S'] = np.log10(EChi2S)
    t['log_SizeSecondMax'] = np.log10(SizeSecondMax)
    t['camera_offset'] = camera_offset
    t['log_NTelPairs'] = np.log10(NTelPairs)
    t['MSCW'] = MSCW
    t['MSCL'] = MSCL

    return t.to_pandas()


def bin_data_in_energy(dtf, n_bins=20):
    '''
    Bin the data in dtf to n_bins with equal statistics.

    Parameters
    ----------
    dtf: pandas DataFrame
        The DataFrame containing the data.
        Must contain a 'log_reco_energy' column (used to calculate the bins).
    n_bins: int, default=20
        The number of reconstructed energy bins to divide the data in.

    Returns
    -------
    A dictionary of DataFrames (keys=energy ranges, values=separated DataFrames).
    '''

    dtf_e = dict()

    log_e_reco_bins = mstats.mquantiles(dtf['log_reco_energy'].values, np.linspace(0, 1, n_bins))

    for i_e_bin, log_e_high in enumerate(log_e_reco_bins):
        if i_e_bin == 0:
            continue

        mask = np.logical_and(
            dtf['log_reco_energy'] > log_e_reco_bins[i_e_bin - 1],
            dtf['log_reco_energy'] < log_e_high
        )
        this_dtf = dtf[mask]
        if len(this_dtf) < 1:
            raise RuntimeError('One of the energy bins is empty')

        this_e_range = '{:3.3f} < E < {:3.3f} TeV'.format(
            10**log_e_reco_bins[i_e_bin - 1],
            10**log_e_high
        )

        dtf_e[this_e_range] = this_dtf

    return dtf_e


def split_data_train_test(dtf_e, test_size=0.75):
    '''
    Split the data into training and testing datasets.
    The data is split in each energy range separately with 'test_size'
    setting the fraction of the test sample.

    Parameters
    ----------
    dtf_e: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to split.
        The keys of the dict are the energy ranges of the data.
    test_size: float or int, default=0.75
        If float, should be between 0.0 and 1.0 and represents the proportion of the dataset
        to include in the test split. If int, represents the absolute number of test samples.
        If None it will be set to 0.25.

    Returns
    -------
    Two dictionaries of DataFrames, one for training and one for testing
    (keys=energy ranges, values=separated DataFrames).
    '''

    dtf_e_train = dict()
    dtf_e_test = dict()

    for this_e_range, this_dtf in dtf_e.items():
        dtf_e_train[this_e_range], dtf_e_test[this_e_range] = model_selection.train_test_split(
            this_dtf,
            test_size=test_size,
            random_state=0
        )

    return dtf_e_train, dtf_e_test


def define_regressors():
    '''
    Define regressors to train the data with.
    All possible regressors should be added here.
    Regressors can be simple ones or pipelines that include standardisation or anything else.
    The parameters for the regressors are hard coded since they are expected to more or less
    stay constant once tuned.
    TODO: Include a model selection method in the pipeline?
          That way it can be done automatically separately in each energy bin.
          (see https://scikit-learn.org/stable/modules/feature_selection.html).

    Returns
    -------
    A dictionary of regressors to train.
    '''

    regressors = dict()

    regressors['random_forest'] = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=8)
    regressors['MLP'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPRegressor(
            hidden_layer_sizes=(80, 45),
            solver='adam',
            max_iter=20000,
            activation='tanh',
            # early_stopping=True,
            random_state=0
        )
    )
    regressors['MLP_relu'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPRegressor(
            hidden_layer_sizes=(80, 45),
            solver='adam',
            max_iter=20000,
            activation='relu',
            # early_stopping=True,
            random_state=0
        )
    )
    regressors['MLP_logistic'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPRegressor(
            hidden_layer_sizes=(80, 45),
            solver='adam',
            max_iter=20000,
            activation='logistic',
            # early_stopping=True,
            random_state=0
        )
    )
    regressors['BDT'] = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=30, random_state=0),
        n_estimators=1000, random_state=0
    )
    regressors['linear_regression'] = LinearRegression(n_jobs=4)
    regressors['ridge'] = Ridge(alpha=1.0)
    regressors['SVR'] = SVR(C=10.0, epsilon=0.2)
    regressors['linear_SVR'] = make_pipeline(
        preprocessing.StandardScaler(),
        LinearSVR(random_state=0, tol=1e-5, C=10.0, epsilon=0.2, max_iter=10000)
    )

    return regressors


def train_models(dtf_e_train, train_features, labels, regressors):
    '''
    Train all the models in regressors, using the data in dtf_e_train.
    The models are trained per energy range in dtf_e_train.

    Parameters
    ----------
    dtf_e_train: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to train with.
        The keys of the dict are the energy ranges of the data.
        Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    train_features: list
        List of variable names to train with.
    labels: str
        Name of the variable used as the labels in the training.
    regressors: dict of sklearn regressors
        A dictionary of regressors to train as returned from define_regressors().


    Returns
    -------
    A nested dictionary trained models:
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values=trained models
    '''

    models = dict()
    for this_model, this_regressor in regressors.items():
        models[this_model] = dict()
        for this_e_range in dtf_e_train.keys():

            print('Training {} in the energy range - {}'.format(this_model, this_e_range))
            X_train = dtf_e_train[this_e_range][train_features].values
            y_train = dtf_e_train[this_e_range][labels].values

            models[this_model][this_e_range] = copy.deepcopy(
                this_regressor.fit(X_train, y_train)
            )

    return models


def save_models(trained_models):
    '''
    Save the trained models to disk.
    The path for the models is in models/'regressor name'.
    All models are saved per energy range for each regressor in trained_models.

    Parameters
    ----------
    trained_models: a nested dict of trained sklearn regressor per energy range.
    1st dict:
        keys=model names, values=2nd dict
    2nd dict:
        keys=energy ranges, values=trained models
    '''

    for regressor_name, this_regressor in trained_models.items():
        this_dir = Path('models').joinpath(regressor_name).mkdir(parents=True, exist_ok=True)
        for this_e_range, this_model in this_regressor.items():

            e_range_name = this_e_range.replace(' < ', '-').replace(' ', '_')

            model_file_name = Path('models').joinpath(
                regressor_name,
                '{}.joblib'.format(e_range_name)
            )
            dump(this_model, model_file_name, compress=3)

    return


def save_test_dtf(dtf_e_test):
    '''
    Save the test data to disk so it can be loaded together with load_models().
    The path for the test data is in models/test_data.
    # TODO: This is stupid to save the actual data,
            better to simply save a list of event numbers or something.

    Parameters
    ----------
    dtf_e_test: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to test with.
        The keys of the dict are the energy ranges of the data.
        Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    '''

    this_dir = Path('models').joinpath('test_data').mkdir(parents=True, exist_ok=True)

    test_data_file_name = Path('models').joinpath('test_data').joinpath('dtf_e_test.joblib')
    dump(dtf_e_test, test_data_file_name, compress=3)

    return


def load_test_dtf():
    '''
    Load the test data together with load_models().
    The path for the test data is in models/test_data.
    # TODO: This is stupid to save the actual data,
            better to simply save a list of event numbers or something.

    Returns
    -------
    dtf_e_test: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to test with.
        The keys of the dict are the energy ranges of the data.
        Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    '''

    test_data_file_name = Path('models').joinpath('test_data').joinpath('dtf_e_test.joblib')

    return load(test_data_file_name)


def load_models(regressor_names=list()):
    '''
    Read the trained models from disk.
    The path for the models is in models/'regressor name'.
    All models are saved per energy range for each regressor in trained_models.

    Parameters
    ----------
    regressor_names: list of str
        A list of regressor names to load from disk
        # TODO: take the default list from define_regressors()?

    Returns
    -------
    trained_models: a nested dict of trained sklearn regressor per energy range.
    1st dict:
        keys=model names, values=2nd dict
    2nd dict:
        keys=energy ranges, values=trained models
    '''

    trained_models = defaultdict(dict)

    for regressor_name in regressor_names:
        models_dir = Path('models').joinpath(regressor_name)
        for this_file in sorted(models_dir.iterdir(), key=os.path.getmtime):

            e_range_name = this_file.stem.replace('-', ' < ').replace('_', ' ')

            model_file_name = Path('models').joinpath(
                regressor_name,
                '{}.joblib'.format(e_range_name)
            )
            trained_models[regressor_name][e_range_name] = load(this_file)

    return trained_models


def pearson_correlation(dtf):
    '''
    Calculate the Pearson correlation between all variables in this DataFrame.

    Parameters
    ----------
    dtf: pandas DataFrame
        The DataFrame containing the data.

    Returns
    -------
    A pyplot instance with the Pearson correlation plot.
    '''

    plt.subplots(figsize=[8, 8])
    corr_matrix = dtf.corr(method='pearson')
    sns.heatmap(
        corr_matrix,
        vmin=-1.,
        vmax=1.,
        annot=True,
        fmt='.2f',
        cmap="YlGnBu",
        cbar=True,
        linewidths=0.5
    )
    plt.title('pearson correlations')

    return plt


def plot_test_vs_predict(dtf_e_test, trained_models, trained_model_name, train_features, labels):
    '''
    Plot true values vs. the predictions of the model for all energy bins.

    Parameters
    ----------
    dtf_e_test: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to test with.
        The keys of the dict are the energy ranges of the data.
        Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    trained_models: dict of a trained sklearn regressor per energy range
        (keys=energy ranges, values=trained models).
    trained_model_name: str
        Name of the regressor trained.
    train_features: list
        List of variable names trained with.
    labels: str
        Name of the variable used as the labels in the training.


    Returns
    -------
    A pyplot instance with the test vs. prediction plot.
    '''

    nrows = 5
    ncols = 4

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[14, 18])

    for i_plot, (this_e_range, this_model) in enumerate(trained_models.items()):

        X_test = dtf_e_test[this_e_range][train_features].values
        y_test = dtf_e_test[this_e_range][labels].values

        y_pred = this_model.predict(X_test)

        ax = axs[int(np.floor((i_plot)/ncols)), (i_plot) % 4]

        ax.hist2d(y_pred, y_test, bins=(50, 50), cmap=plt.cm.jet)
#         ax.plot([min(y_test), max(y_test)], [min(y_test),max(y_test)],
#                  linestyle='--', lw=2, color='blue')
        ax.set_title(this_e_range)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        score = round(this_model.score(X_test, y_test), 4)
        print('score({}) = {}'.format(this_e_range, score))

    axs[nrows - 1, ncols - 1].axis('off')
    axs[nrows - 1, ncols - 1].text(
        1.5,
        0.5,
        trained_model_name,
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=18,
        transform=ax.transAxes
    )
    plt.tight_layout()

    return plt


def plot_variable_importance(trained_model, regressor_name, energy_bin, train_features):
    '''
    Plot the importance of the variables for the provided trained_model in the 'energy_bin'.

    Parameters
    ----------
    trained_model: a trained sklearn regressor for one energy range
    regressor_name: str
        The regressor name (as defined in define_regressors())
    energy_bin: str
        The energy bin for this model (as defined in bin_data_in_energy())
    train_features: list
        List of variable names trained with.


    Returns
    -------
    A pyplot instance with the importances plot.
    '''

    if hasattr(trained_model, 'feature_importances_'):

        importances = trained_model.feature_importances_
        dtf_importances = pd.DataFrame({'importance': importances, 'variable': train_features})
        dtf_importances.sort_values('importance', ascending=False)
        dtf_importances['cumsum'] = dtf_importances['importance'].cumsum(axis=0)
        dtf_importances = dtf_importances.set_index('variable')

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=[12, 6])
        fig.suptitle('Features Importance for {}\n{}'.format(
                regressor_name,
                this_e_range
            ),
            fontsize=20
        )
        ax[0].title.set_text('variables')
        dtf_importances[['importance']].sort_values(by='importance').plot(
            kind='barh',
            legend=False,
            ax=ax[0]
        ).grid(axis='x')
        ax[0].set(ylabel='')
        ax[1].title.set_text('cumulative')
        dtf_importances[['cumsum']].plot(
            kind='line',
            linewidth=4,
            legend=False,
            ax=ax[1]
        )
        ax[1].set(
            xlabel='',
            xticks=np.arange(len(dtf_importances)),
            xticklabels=dtf_importances.index
        )
        plt.xticks(rotation=70)
        plt.grid(axis='both')

        return plt
    else:
        print('Warning: importances cannot be calculated for the {} model'.format(regressor_name))
        return None
