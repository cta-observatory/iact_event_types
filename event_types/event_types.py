import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import copy
from collections import defaultdict
# from astropy.coordinates.angle_utilities import angular_separation
# from astropy.coordinates import Angle
# from astropy import units as u
import pandas as pd
import seaborn as sns
from pathlib import Path
from joblib import dump, load
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import mstats
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multiclass import OneVsRestClassifier
from sklearn import model_selection, preprocessing  # , feature_selection, metrics
from sklearn.pipeline import make_pipeline


def setStyle(palette='default', bigPlot=False):
    """
    A function to set the plotting style.
    The function receives the colour palette name and whether it is
    a big plot or not. The latter sets the fonts and marker to be bigger in case it is a big plot.
    The available colour palettes are as follows:

    - classic (default): A classic colourful palette with strong colours and contrast.
    - modified classic: Similar to the classic, with slightly different colours.
    - autumn: A slightly darker autumn style colour palette.
    - purples: A pseudo sequential purple colour palette (not great for contrast).
    - greens: A pseudo sequential green colour palette (not great for contrast).

    To use the function, simply call it before plotting anything.

    Parameters
    ----------
    palette: str
    bigPlot: bool

    Raises
    ------
    KeyError if provided palette does not exist.
    """

    COLORS = dict()
    COLORS['classic'] = ['#ba2c54', '#5B90DC', '#FFAB44', '#0C9FB3', '#57271B', '#3B507D',
                         '#794D88', '#FD6989', '#8A978E', '#3B507D', '#D8153C', '#cc9214']
    COLORS['modified classic'] = ['#D6088F', '#424D9C', '#178084', '#AF99DA', '#F58D46', '#634B5B',
                                  '#0C9FB3', '#7C438A', '#328cd6', '#8D0F25', '#8A978E', '#ffcb3d']
    COLORS['autumn'] = ['#A9434D', '#4E615D', '#3C8DAB', '#A4657A', '#424D9C', '#DC575A',
                        '#1D2D38', '#634B5B', '#56276D', '#577580', '#134663', '#196096']
    COLORS['purples'] = ['#a57bb7', '#343D80', '#EA60BF', '#B7308E', '#E099C3', '#7C438A',
                         '#AF99DA', '#4D428E', '#56276D', '#CC4B93', '#DC4E76', '#5C4AE4']
    COLORS['greens'] = ['#268F92', '#abc14d', '#8A978E', '#0C9FB3', '#BDA962', '#B0CB9E',
                        '#769168', '#5E93A5', '#178084', '#B7BBAD', '#163317', '#76A63F']

    COLORS['default'] = COLORS['classic']

    MARKERS = ['o', 's', 'v', '^', '*', 'P', 'd', 'X', 'p', '<', '>', 'h']
    LINES = [(0, ()),  # solid
             (0, (1, 1)),  # densely dotted
             (0, (3, 1, 1, 1)),  # densely dashdotted
             (0, (5, 5)),  # dashed
             (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
             (0, (5, 1)),  # desnely dashed
             (0, (1, 5)),  # dotted
             (0, (3, 5, 1, 5)),  # dashdotted
             (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
             (0, (5, 10)),  # loosely dashed
             (0, (1, 10)),  # loosely dotted
             (0, (3, 10, 1, 10)),  # loosely dashdotted
             ]

    if palette not in COLORS.keys():
        raise KeyError('palette must be one of {}'.format(', '.join(COLORS)))

    fontsize = {'default': 15, 'bigPlot': 30}
    markersize = {'default': 8, 'bigPlot': 18}
    plotSize = 'default'
    if bigPlot:
        plotSize = 'bigPlot'

    plt.rc('lines', linewidth=2, markersize=markersize[plotSize])
    plt.rc('axes', prop_cycle=(
            cycler(color=COLORS[palette])
            + cycler(linestyle=LINES)
            + cycler(marker=MARKERS))
           )
    plt.rc(
        'axes',
        titlesize=fontsize[plotSize],
        labelsize=fontsize[plotSize],
        labelpad=5,
        grid=True,
        axisbelow=True
    )
    plt.rc('xtick', labelsize=fontsize[plotSize])
    plt.rc('ytick', labelsize=fontsize[plotSize])
    plt.rc('legend', loc='best', shadow=False, fontsize='medium')
    plt.rc('font', family='serif', size=fontsize[plotSize])

    return


def branches_to_read():
    """
    Define a list of branches to read from the ROOT file
    (faster than reading all branches).

    Returns
    -------
    A list of branches names.
    """

    branches = [
        'runNumber',
        'eventNumber',
        'MCe0',
        'MCxoff',
        'MCyoff',
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
        'MSCL',
        'EmissionHeight',
        'EmissionHeightChi2',
        'dist',
        'DispDiff',
        'dESabs',
        'loss',
        'NTrig',
        'meanPedvar_Image',
        'fui',
        'cross',
        'R',
        'ES',
        'asym',
        'tgrad_x',
    ]

    return branches


def nominal_labels_train_features():
    """
    Define the nominal labels variable and training features to train with.

    Returns
    -------
    label: str, train_features: list of str
         Two variables are returned:
             1. the name of the variable to use as the labels in the training.
             2. list of names of variables to used as the training features.
    """

    labels = 'log_ang_diff'

    train_features = [
        'log_reco_energy',
        'log_NTels_reco',
        'array_distance',
        'img2_ang',
        'log_SizeSecondMax',
        'MSCW',
        'MSCL',
        'log_EChi2S',
        'log_EmissionHeight',
        'log_EmissionHeightChi2',
        'log_DispDiff',
        'log_dESabs',
        'NTrig',
        'meanPedvar_Image',
        'MSWOL',
        'log_av_size',
        'log_me_size',
        'log_std_size',
        'av_dist',
        'me_dist',
        'std_dist',
        'av_fui',
        'me_fui',
        'std_fui',
        'av_cross',
        'me_cross',
        'std_cross',
        'av_R',
        'me_R',
        'std_R',
        'av_ES',
        'me_ES',
        'std_ES',
        'sum_loss',
        'av_loss',
        'me_loss',
        'std_loss',
        'av_asym',
        'me_asym',
        'std_asym',
        'av_tgrad_x',
        'me_tgrad_x',
        'std_tgrad_x',
        'camera_offset',
    ]

    return labels, train_features


def extract_df_from_dl2(root_filename):
    """
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
    A pandas DataFrame with variables to use in the regression/classification, after cuts.
    """

    branches = branches_to_read()

    particle_file = uproot.open(root_filename)
    cuts = particle_file['DL2EventTree']
    cuts_arrays = cuts.arrays(expressions='CutClass', library='np')

    # Cut 1: Events surviving gamma/hadron separation and direction cuts:
    mask_gamma_like_and_direction = cuts_arrays['CutClass'] == 5

    # Cut 2: Events surviving gamma/hadron separation cut and not direction cut
    mask_gamma_like_no_direction = cuts_arrays['CutClass'] == 0

    # Cut 0: Events before gamma/hadron and direction cuts (classes 0, 5 and 7)
    gamma_like_events_all = mask_gamma_like_no_direction | mask_gamma_like_and_direction
    gamma_like_events_all = gamma_like_events_all | (cuts_arrays['CutClass'] == 7)

    step_size = 5000  # slightly optimized on my laptop
    data_dict = defaultdict(list)

    for i_event, data_arrays in enumerate(uproot.iterate(
            '{}:data'.format(root_filename),
            step_size=step_size,
            expressions=branches,
            library='np')
    ):

        if i_event > 0:
            if (i_event * step_size) % 100000 == 0:
                print('Extracted {} events'.format(i_event * step_size))

        gamma_like_events = gamma_like_events_all[i_event * step_size:(i_event + 1) * step_size]
        cut_class = cuts_arrays['CutClass'][i_event * step_size:(i_event + 1) * step_size]
        cut_class = cut_class[gamma_like_events]

        # Label to train with:
        x_off = data_arrays['Xoff'][gamma_like_events]
        y_off = data_arrays['Yoff'][gamma_like_events]
        x_off_mc = data_arrays['MCxoff'][gamma_like_events]
        y_off_mc = data_arrays['MCyoff'][gamma_like_events]
        ang_diff = np.sqrt((x_off - x_off_mc) ** 2. + (y_off - y_off_mc) ** 2.)

        # Variables for training:
        runNumber = data_arrays['runNumber'][gamma_like_events]
        eventNumber = data_arrays['eventNumber'][gamma_like_events]
        reco_energy = data_arrays['ErecS'][gamma_like_events]
        true_energy = data_arrays['MCe0'][gamma_like_events]
        camera_offset = np.sqrt(x_off ** 2. + y_off ** 2.)
        NTels_reco = data_arrays['NImages'][gamma_like_events]
        x_cores = data_arrays['Xcore'][gamma_like_events]
        y_cores = data_arrays['Ycore'][gamma_like_events]
        array_distance = np.sqrt(x_cores ** 2. + y_cores ** 2.)
        img2_ang = data_arrays['img2_ang'][gamma_like_events]
        EChi2S = data_arrays['EChi2S'][gamma_like_events]
        SizeSecondMax = data_arrays['SizeSecondMax'][gamma_like_events]
        NTelPairs = data_arrays['NTelPairs'][gamma_like_events]
        MSCW = data_arrays['MSCW'][gamma_like_events]
        MSCL = data_arrays['MSCL'][gamma_like_events]
        EmissionHeight = data_arrays['EmissionHeight'][gamma_like_events]
        EmissionHeightChi2 = data_arrays['EmissionHeightChi2'][gamma_like_events]
        DispDiff = data_arrays['DispDiff'][gamma_like_events]
        dESabs = data_arrays['dESabs'][gamma_like_events]
        NTrig = data_arrays['NTrig'][gamma_like_events]
        meanPedvar_Image = data_arrays['meanPedvar_Image'][gamma_like_events]

        av_size = [np.average(sizes) for sizes in data_arrays['size'][gamma_like_events]]
        me_size = [np.median(sizes) for sizes in data_arrays['size'][gamma_like_events]]
        std_size = [np.std(sizes) for sizes in data_arrays['size'][gamma_like_events]]

        av_dist = [np.average(dists) for dists in data_arrays['dist'][gamma_like_events]]
        me_dist = [np.median(dists) for dists in data_arrays['dist'][gamma_like_events]]
        std_dist = [np.std(dists) for dists in data_arrays['dist'][gamma_like_events]]

        av_fui = [np.average(fui) for fui in data_arrays['fui'][gamma_like_events]]
        me_fui = [np.median(fui) for fui in data_arrays['fui'][gamma_like_events]]
        std_fui = [np.std(fui) for fui in data_arrays['fui'][gamma_like_events]]

        av_cross = [np.average(cross) for cross in data_arrays['cross'][gamma_like_events]]
        me_cross = [np.median(cross) for cross in data_arrays['cross'][gamma_like_events]]
        std_cross = [np.std(cross) for cross in data_arrays['cross'][gamma_like_events]]

        av_R = [np.average(R) for R in data_arrays['R'][gamma_like_events]]
        me_R = [np.median(R) for R in data_arrays['R'][gamma_like_events]]
        std_R = [np.std(R) for R in data_arrays['R'][gamma_like_events]]

        av_ES = [np.average(ES) for ES in data_arrays['ES'][gamma_like_events]]
        me_ES = [np.median(ES) for ES in data_arrays['ES'][gamma_like_events]]
        std_ES = [np.std(ES) for ES in data_arrays['ES'][gamma_like_events]]

        sum_loss = [np.sum(losses) for losses in data_arrays['loss'][gamma_like_events]]
        av_loss = [np.average(losses) for losses in data_arrays['loss'][gamma_like_events]]
        me_loss = [np.median(losses) for losses in data_arrays['loss'][gamma_like_events]]
        std_loss = [np.std(losses) for losses in data_arrays['loss'][gamma_like_events]]

        av_asym = [np.average(asym) for asym in data_arrays['asym'][gamma_like_events]]
        me_asym = [np.median(asym) for asym in data_arrays['asym'][gamma_like_events]]
        std_asym = [np.std(asym) for asym in data_arrays['asym'][gamma_like_events]]

        av_tgrad_x = [np.average(tgrad_x) for tgrad_x in data_arrays['tgrad_x'][gamma_like_events]]
        me_tgrad_x = [np.median(tgrad_x) for tgrad_x in data_arrays['tgrad_x'][gamma_like_events]]
        std_tgrad_x = [np.std(tgrad_x) for tgrad_x in data_arrays['tgrad_x'][gamma_like_events]]

        data_dict['runNumber'].extend(tuple(runNumber))
        data_dict['eventNumber'].extend(tuple(eventNumber))
        data_dict['cut_class'].extend(tuple(cut_class))
        data_dict['log_ang_diff'].extend(tuple(np.log10(ang_diff)))
        data_dict['log_true_energy'].extend(tuple(np.log10(true_energy)))
        data_dict['log_reco_energy'].extend(tuple(np.log10(reco_energy)))
        data_dict['x_off_mc'].extend(tuple(x_off_mc))
        data_dict['y_off_mc'].extend(tuple(y_off_mc))
        data_dict['camera_offset'].extend(tuple(camera_offset))
        data_dict['log_NTels_reco'].extend(tuple(np.log10(NTels_reco)))
        data_dict['array_distance'].extend(tuple(array_distance))
        data_dict['img2_ang'].extend(tuple(img2_ang))
        data_dict['log_EChi2S'].extend(tuple(np.log10(EChi2S)))
        data_dict['log_SizeSecondMax'].extend(tuple(np.log10(SizeSecondMax)))
        data_dict['log_NTelPairs'].extend(tuple(np.log10(NTelPairs)))
        data_dict['MSCW'].extend(tuple(MSCW))
        data_dict['MSCL'].extend(tuple(MSCL))
        data_dict['log_EmissionHeight'].extend(tuple(np.log10(EmissionHeight)))
        data_dict['log_EmissionHeightChi2'].extend(tuple(np.log10(EmissionHeightChi2)))
        data_dict['log_DispDiff'].extend(tuple(np.log10(DispDiff)))
        data_dict['log_dESabs'].extend(tuple(np.log10(dESabs)))
        data_dict['NTrig'].extend(tuple(NTrig))
        data_dict['meanPedvar_Image'].extend(tuple(meanPedvar_Image))
        data_dict['MSWOL'].extend(tuple(MSCW / MSCL))

        data_dict['log_av_size'].extend(tuple(np.log10(av_size)))
        data_dict['log_me_size'].extend(tuple(np.log10(me_size)))
        data_dict['log_std_size'].extend(tuple(np.log10(std_size)))

        data_dict['av_dist'].extend(tuple(av_dist))
        data_dict['me_dist'].extend(tuple(me_dist))
        data_dict['std_dist'].extend(tuple(std_dist))

        data_dict['av_fui'].extend(tuple(av_fui))
        data_dict['me_fui'].extend(tuple(me_fui))
        data_dict['std_fui'].extend(tuple(std_fui))

        data_dict['av_cross'].extend(tuple(av_cross))
        data_dict['me_cross'].extend(tuple(me_cross))
        data_dict['std_cross'].extend(tuple(std_cross))

        data_dict['av_R'].extend(tuple(av_R))
        data_dict['me_R'].extend(tuple(me_R))
        data_dict['std_R'].extend(tuple(std_R))

        data_dict['av_ES'].extend(tuple(av_ES))
        data_dict['me_ES'].extend(tuple(me_ES))
        data_dict['std_ES'].extend(tuple(std_ES))

        data_dict['sum_loss'].extend(tuple(sum_loss))
        data_dict['av_loss'].extend(tuple(av_loss))
        data_dict['me_loss'].extend(tuple(me_loss))
        data_dict['std_loss'].extend(tuple(std_loss))

        data_dict['av_asym'].extend(tuple(av_asym))
        data_dict['me_asym'].extend(tuple(me_asym))
        data_dict['std_asym'].extend(tuple(std_asym))

        data_dict['av_tgrad_x'].extend(tuple(av_tgrad_x))
        data_dict['me_tgrad_x'].extend(tuple(me_tgrad_x))
        data_dict['std_tgrad_x'].extend(tuple(std_tgrad_x))

    return pd.DataFrame(data=data_dict)


def save_dtf(dtf, suffix=''):
    """
    Save the test dataset to disk as it is much quicker
    to read the reduced pickled data than the ROOT file.

    Parameters
    ----------
    dtf: pandas DataFrames
    suffix: str
        The suffix to add to the file name
    """

    Path('reduced_data').mkdir(parents=True, exist_ok=True)

    if suffix != '':
        if not suffix.startswith('_'):
            suffix = '_{}'.format(suffix)

    data_file_name = Path('reduced_data').joinpath(
        'dtf{}.joblib'.format(suffix)
    )
    dump(dtf, data_file_name, compress=3)

    return


def load_dtf(suffix=''):
    """
    Load the reduced data from reduced_data/.

    Parameters
    ----------
    suffix: str
        The suffix added to the file name (the nominal is dtf.joblib)

    Returns
    -------
    dtf: pandas DataFrames of the reduced data
    """

    if suffix != '':
        if not suffix.startswith('_'):
            suffix = '_{}'.format(suffix)

    data_file_name = Path('reduced_data').joinpath(
        'dtf{}.joblib'.format(suffix)
    )

    return load(data_file_name)


def load_all_dtfs(suffix=['']):
    """
    Load the reduced data from reduced_data/ from multiple files into one DataFrame

    Parameters
    ----------
    suffix: list of str
        The suffix added to the file names (the nominal is dtf.joblib)

    Returns
    -------
    dtf_merged: pandas DataFrames of the reduced data
    """

    n = len(suffix)
    all_dtfs = [None] * n

    for i in range(n):
        all_dtfs[i] = load_dtf(suffix[i])

    dtf_merged = pd.concat(all_dtfs, ignore_index=True)

    return dtf_merged


def bin_data_in_energy(dtf, n_bins=20, log_e_reco_bins=None, return_bins=False):
    """
    Bin the data in dtf to n_bins with equal statistics.

    Parameters
    ----------
    dtf: pandas DataFrame
        The DataFrame containing the data.
        Must contain a 'log_reco_energy' column (used to calculate the bins).
    n_bins: int, default=20
        The number of reconstructed energy bins to divide the data in.
    log_e_reco_bins: array-like, None
        In case it is not none, it will be used as the energy bins to divide the data sample

    return_bins: bool
        If true, the function will return the log_e_reco_bins used to bin the data.

    Returns
    -------
    A dictionary of DataFrames (keys=energy ranges, values=separated DataFrames).
    """

    dtf_e = dict()

    if log_e_reco_bins is None:
        log_e_reco_bins = mstats.mquantiles(
            dtf['log_reco_energy'].values,
            np.linspace(0, 1, n_bins)
        )

    for i_e_bin, log_e_high in enumerate(log_e_reco_bins):
        if i_e_bin == 0:
            continue

        mask = np.logical_and(
            dtf['log_reco_energy'] >= log_e_reco_bins[i_e_bin - 1],
            dtf['log_reco_energy'] < log_e_high
        )
        this_dtf = dtf[mask]

        this_e_range = '{:3.3f} < E < {:3.3f} TeV'.format(
            10 ** log_e_reco_bins[i_e_bin - 1],
            10 ** log_e_high
        )
        if len(this_dtf) < 1:
            raise RuntimeError('The range {} is empty'.format(this_e_range))

        dtf_e[this_e_range] = this_dtf
    if return_bins:
        return dtf_e, log_e_reco_bins
    else:
        return dtf_e


def bin_data_in_energy_and_offset(dtf, n_bins=20, log_e_reco_bins=None, offset_bins=None, return_bins=False):
    """
    Bin the data in dtf to n_bins with equal statistics.

    Parameters
    ----------
    dtf: pandas DataFrame
        The DataFrame containing the data.
        Must contain a 'log_reco_energy' column (used to calculate the bins).
    n_bins: int, default=20
        The number of reconstructed energy bins to divide the data in.
    log_e_reco_bins: array-like, None
        In case it is not none, it will be used as the energy bins to divide the data sample
    offset_bins: array-like
        Camera-offset bins (in degrees) to divide the data sample.

    return_bins: bool
        If true, the function will return the log_e_reco_bins used to bin the data.

    Returns
    -------
    A dictionary of DataFrames (keys=energy ranges, values=separated DataFrames).
    """

    dtf_e_offset = dict()

    if offset_bins is None:
        raise ValueError('No offset_bins are inserted.')
    elif offset_bins[0] != 0:
        raise ValueError('The first value of offset_bins must be zero.')

    if log_e_reco_bins is None:
        log_e_reco_bins = mstats.mquantiles(
            dtf['log_reco_energy'].values,
            np.linspace(0, 1, n_bins)
        )

    for i_e_bin, log_e_high in enumerate(log_e_reco_bins):
        if i_e_bin == 0:
            continue

        mask = np.logical_and(
            dtf['log_reco_energy'] >= log_e_reco_bins[i_e_bin - 1],
            dtf['log_reco_energy'] < log_e_high
        )
        this_dtf_e = dtf[mask]

        this_e_range = '{:3.3f} < E < {:3.3f} TeV'.format(
            10 ** log_e_reco_bins[i_e_bin - 1],
            10 ** log_e_high
        )
        if len(this_dtf_e) < 1:
            raise RuntimeError('The range {} is empty'.format(this_e_range))

        dtf_e_offset[this_e_range] = dict()

        for i_offset_bin, offset_high in enumerate(offset_bins):
            if i_offset_bin == 0:
                continue

            mask = np.logical_and(
                this_dtf_e['camera_offset'] >= offset_bins[i_offset_bin - 1],
                this_dtf_e['camera_offset'] < offset_high
            )
            this_dtf_e_offset = this_dtf_e[mask]

            this_offset_range = '{:2.2f} < camera_offset < {:2.2f} deg'.format(
                offset_bins[i_offset_bin - 1],
                offset_high
            )
            if len(this_dtf_e_offset) < 1:
                raise RuntimeError('The range {}, {} is empty'.format(this_e_range, this_offset_range))

            dtf_e_offset[this_e_range][this_offset_range] = this_dtf_e_offset

    if return_bins:
        return dtf_e_offset, log_e_reco_bins
    else:
        return dtf_e_offset


def extract_energy_bins(e_ranges):
    """
    Extract the energy bins from the list of energy ranges.
    This is a little weird function which can probably be avoided if we use a class
    instead of a namespace. However, it is useful for now so...

    Parameters
    ----------
    e_ranges: list of str
        A list of energy ranges in string form as '{:3.3f} < E < {:3.3f} TeV'.

    Returns
    -------
    energy_bins: list of floats
        List of energy bin edges given in e_ranges.
    """

    energy_bins = list()

    for this_range in e_ranges:
        low_e = float(this_range.split()[0])
        energy_bins.append(low_e)

    energy_bins.append(float(list(e_ranges)[-1].split()[4]))  # Add also the upper bin edge

    return energy_bins


def extract_energy_bins_centers(e_ranges):
    """
    Extract the energy bins from the list of energy ranges.
    This is a little weird function which can probably be avoided if we use a class
    instead of a namespace. However, it is useful for now so...

    Parameters
    ----------
    e_ranges: list of str
        A list of energy ranges in string form as '{:3.3f} < E < {:3.3f} TeV'.

    Returns
    -------
    energy_bin_centers: list of floats
        Energy bins calculated as the averages of the energy ranges in e_ranges.
    """

    energy_bin_centers = list()

    for this_range in e_ranges:
        low_e = float(this_range.split()[0])
        high_e = float(this_range.split()[4])

        energy_bin_centers.append((high_e + low_e) / 2.)

    return energy_bin_centers


def split_data_train_test(dtf_e, test_size=0.75, random_state=777):
    """
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
    random_state: int


    Returns
    -------
    Two dictionaries of DataFrames, one for training and one for testing
    (keys=energy ranges, values=separated DataFrames).
    """

    dtf_e_train = dict()
    dtf_e_test = dict()

    for this_e_range, this_dtf in dtf_e.items():
        dtf_e_train[this_e_range], dtf_e_test[this_e_range] = model_selection.train_test_split(
            this_dtf,
            test_size=test_size,
            random_state=random_state
        )

    return dtf_e_train, dtf_e_test


def add_event_type_column(dtf, labels, n_types=2):
    """
    Add an event type column by dividing the data into n_types bins with equal statistics
    based on the labels column in dtf.
    Unlike in most cases in this code, dtf is the DataFrame itself,
    not a dict of energy ranges. This function should be called per energy bin.

    Parameters
    ----------
    dtf: pandas DataFrames
        A DataFrame to add event types to.
    labels: str
        Name of the variable used as the labels in the training.
    n_types: int
        The number of types to divide the data in.

    Returns
    -------
    A DataFrame with an additional event_type column.
    """

    event_type_quantiles = np.linspace(0, 1, n_types + 1)
    event_types_bins = mstats.mquantiles(dtf[labels].values, event_type_quantiles)
    event_types = list()
    for this_value in dtf[labels].values:
        this_event_type = np.searchsorted(event_types_bins, this_value)
        if this_event_type < 1:
            this_event_type = 1
        if this_event_type > n_types:
            this_event_type = n_types

        event_types.append(this_event_type)

    dtf.loc[:, 'event_type'] = event_types

    return dtf


def define_regressors():
    """
    Define regressors to train the data with.
    All possible regressors should be added here.
    Regressors can be simple ones or pipelines that include standardisation or anything else.
    The parameters for the regressors are hard coded since they are expected to more or less
    stay constant once tuned.
    TODO: Include a feature selection method in the pipeline?
          That way it can be done automatically separately in each energy bin.
          (see https://scikit-learn.org/stable/modules/feature_selection.html).

    Returns
    -------
    A dictionary of regressors to train.
    """

    regressors = dict()

    regressors['random_forest'] = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=0,
        n_jobs=8
    )
    regressors['MLP_relu'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPRegressor(
            hidden_layer_sizes=(100, 50),
            solver='adam',
            max_iter=20000,
            activation='relu',
            tol=1e-5,
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
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    regressors['MLP_uniform'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='uniform', random_state=0),
        MLPRegressor(
            hidden_layer_sizes=(80, 45),
            solver='adam',
            max_iter=20000,
            activation='tanh',
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    regressors['MLP_tanh'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPRegressor(
            hidden_layer_sizes=(36, 6),
            solver='adam',
            max_iter=20000,
            activation='tanh',
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    regressors['MLP_lbfgs'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPRegressor(
            hidden_layer_sizes=(36, 6),
            solver='lbfgs',
            max_iter=20000,
            activation='logistic',
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    regressors['BDT'] = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=5, random_state=0),
        n_estimators=50, random_state=0
    )
    regressors['BDT_small'] = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=30, random_state=0),
        n_estimators=30, random_state=0
    )
    regressors['linear_regression'] = LinearRegression(n_jobs=4)
    regressors['ridge'] = Ridge(alpha=1.0)
    regressors['SVR'] = SVR(C=10.0, epsilon=0.2)
    regressors['linear_SVR'] = make_pipeline(
        preprocessing.StandardScaler(),
        LinearSVR(random_state=0, tol=1e-5, C=10.0, epsilon=0.2, max_iter=100000)
    )
    regressors['SGD'] = make_pipeline(
        preprocessing.StandardScaler(),
        SGDRegressor(loss='epsilon_insensitive', max_iter=20000, tol=1e-5)
    )

    return regressors


def define_classifiers():
    """
    Define classifiers to train the data with.
    All possible classifiers should be added here.
    Classifiers can be simple ones or pipelines that include standardisation or anything else.
    The parameters for the classifiers are hard coded since they are expected to more or less
    stay constant once tuned.
    TODO: Include a feature selection method in the pipeline?
          That way it can be done automatically separately in each energy bin.
          (see https://scikit-learn.org/stable/modules/feature_selection.html).

    Returns
    -------
    A dictionary of classifiers to train.
    """

    classifiers = dict()

    classifiers['random_forest_classifier'] = RandomForestClassifier(
        n_estimators=100,
        random_state=0,
        n_jobs=8
    )
    classifiers['MLP_classifier'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPClassifier(
            hidden_layer_sizes=(36, 6),
            solver='adam',
            max_iter=20000,
            activation='tanh',
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    classifiers['MLP_relu_classifier'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPClassifier(
            hidden_layer_sizes=(100, 50),
            solver='adam',
            max_iter=20000,
            activation='relu',
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    classifiers['MLP_logistic_classifier'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='normal', random_state=0),
        MLPClassifier(
            hidden_layer_sizes=(80, 45),
            solver='adam',
            max_iter=20000,
            activation='logistic',
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    classifiers['MLP_uniform_classifier'] = make_pipeline(
        preprocessing.QuantileTransformer(output_distribution='uniform', random_state=0),
        MLPClassifier(
            hidden_layer_sizes=(80, 45),
            solver='adam',
            max_iter=20000,
            activation='tanh',
            tol=1e-5,
            # early_stopping=True,
            random_state=0
        )
    )
    classifiers['BDT_classifier'] = AdaBoostClassifier(
        n_estimators=100, random_state=0
    )
    classifiers['ridge_classifier'] = RidgeClassifier()
    classifiers['ridgeCV_classifier'] = RidgeClassifierCV(
        alphas=[1e-3, 1e-2, 1e-1, 1],
        normalize=True
    )
    classifiers['SVC_classifier'] = SVC(gamma=2, C=1)
    classifiers['SGD_classifier'] = make_pipeline(
        preprocessing.StandardScaler(),
        SGDClassifier(loss='epsilon_insensitive', max_iter=20000, tol=1e-5)
    )
    classifiers['Gaussian_process_classifier'] = GaussianProcessClassifier(1.0 * RBF(1.0))
    classifiers['bagging_svc_classifier'] = BaggingClassifier(
        base_estimator=SVC(),
        n_estimators=100,
        random_state=0
    )
    classifiers['bagging_dt_classifier'] = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=100,
        random_state=0
    )
    classifiers['oneVsRest_classifier'] = OneVsRestClassifier(SVC(), n_jobs=8)
    classifiers['gradient_boosting_classifier'] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=0
    )

    return classifiers


def train_models(dtf_e_train, models_to_train):
    """
    Train all the models in models, using the data in dtf_e_train.
    The models are trained per energy range in dtf_e_train.

    Parameters
    ----------
    dtf_e_train: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to train with.
        The keys of the dict are the energy ranges of the data.
        Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    models_to_train: a nested dict of models:
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            'model':dict of sklearn models (as returned from define_regressors/classifiers()).
            'train_features': list of variable names to train with.
            'labels': Name of the variable used as the labels in the training.


    Returns
    -------
    A nested dictionary trained models, train_features and labels:
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values 3rd dict
        3rd dict:
            'model': trained model for this energy range
            'train_features': list of variable names to train with.
            'labels': Name of the variable used as the labels in the training.
    """

    models = dict()
    for this_model_name, this_model in models_to_train.items():
        models[this_model_name] = dict()
        for this_e_range in dtf_e_train.keys():
            print('Training {} in the energy range - {}'.format(this_model_name, this_e_range))
            x_train = dtf_e_train[this_e_range][this_model['train_features']].values
            y_train = dtf_e_train[this_e_range][this_model['labels']].values

            models[this_model_name][this_e_range] = dict()
            models[this_model_name][this_e_range]['train_features'] = this_model['train_features']
            models[this_model_name][this_e_range]['labels'] = this_model['labels']
            models[this_model_name][this_e_range]['test_data_suffix'] = this_model[
                'test_data_suffix'
            ]
            models[this_model_name][this_e_range]['model'] = copy.deepcopy(
                this_model['model'].fit(x_train, y_train)
            )

    return models


def save_models(trained_models, train_size=None):
    """
    Save the trained models to disk.
    The path for the models is in models/'model_name',
    or models/'model_name'_train'train_size' in case the train size is specified.
    All models are saved per energy range for each model in trained_models.

    Parameters
    ----------
    trained_models: a nested dict of trained sklearn model per energy range.
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values 3rd dict
        3rd dict:
            'model': trained model for this energy range.
            'train_features': list of variable names trained with.
            'labels': name of the variable used as the labels in the training.
            'test_data_suffix': suffix of the test dataset saved to disk.
    train_size: percentage of the data used for training the model, so it can be included in the name.
    """

    for model_name, this_model in trained_models.items():
        if train_size is not None:
            model_name = model_name + '_train' + str(train_size)
        Path('models').joinpath(model_name).mkdir(parents=True, exist_ok=True)
        for this_e_range, model_now in this_model.items():
            e_range_name = this_e_range.replace(' < ', '-').replace(' ', '_')

            model_file_name = Path('models').joinpath(
                model_name,
                '{}.joblib'.format(e_range_name)
            )
            dump(model_now, model_file_name, compress=3)

    return


def save_test_dtf(dtf_e_test, suffix='default'):
    """
    Save the test data to disk, so it can be loaded together with load_models().
    The path for the test data is in models/test_data.

    Parameters
    ----------
    dtf_e_test: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to test with.
        The keys of the dict are the energy ranges of the data.
        Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    suffix: str
        The suffix to add to the file name
    """

    Path('models').joinpath('test_data').mkdir(parents=True, exist_ok=True)

    if suffix != '':
        if not suffix.startswith('_'):
            suffix = '_{}'.format(suffix)

    test_data_file_name = Path('models').joinpath('test_data').joinpath(
        'dtf_e_test{}.joblib'.format(suffix)
    )
    dump(dtf_e_test, test_data_file_name, compress=3)

    return


def save_scores(scores):
    """
    Save the scores of the trained models to disk.
    The path for the scores is in scores/'model name'.

    Parameters
    ----------
    scores: a dict of scores per energy range per trained sklearn model.
        dict:
            keys=model names, values=list of scores
    """

    Path('scores').mkdir(parents=True, exist_ok=True)
    for model_name, these_scores in scores.items():
        file_name = Path('scores').joinpath('{}.joblib'.format(model_name))
        dump(these_scores, file_name, compress=3)

    return


def load_test_dtf(suffix='default'):
    """
    Load the test data together with load_models().
    The path for the test data is in models/test_data.

    Parameters
    ----------
    suffix: str
        The suffix added to the file name (the nominal is dtf_e_test_default.joblib)

    Returns
    -------
    dtf_e_test: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data to test with.
        The keys of the dict are the energy ranges of the data.
        Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    """

    if suffix != '':
        if not suffix.startswith('_'):
            suffix = '_{}'.format(suffix)

    test_data_file_name = Path('models').joinpath('test_data').joinpath(
        'dtf_e_test{}.joblib'.format(suffix)
    )

    return load(test_data_file_name)


def load_multi_test_dtfs(suffix=['default']):
    """
    Load the test data together with load_models().
    The path for the test data is in models/test_data.

    Parameters
    ----------
    suffix: list of str
        The suffix added to the file name (the nominal is dtf_e_test_default.joblib).

    Returns
    -------
    dtf_e_test: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            dict of pandas DataFrames
            Each entry in the dict is a DataFrame containing the data to test with.
            The keys of the dict are the energy ranges of the data.
            Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    """

    dtf_e_test = dict()
    for this_data_name in suffix:
        dtf_e_test[this_data_name] = load_test_dtf(this_data_name)

    return dtf_e_test


def load_models(model_names=list()):
    """
    Read the trained models from disk.
    The path for the models is in models/'model name'.
    All models are saved per energy range for each model in trained_models.

    Parameters
    ----------
    model_names: list of str
        A list of model names to load from disk

    Returns
    -------
    trained_models: a nested dict of trained sklearn model per energy range.
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values 3rd dict
        3rd dict:
            'model': trained model for this energy range.
            'train_features': list of variable names trained with.
            'labels': name of the variable used as the labels in the training.
            'test_data_suffix': suffix of the test dataset saved to disk.
    """

    trained_models = defaultdict(dict)

    for model_name in model_names:
        print('Loading the {} model'.format(model_name))
        models_dir = Path('models').joinpath(model_name)
        for this_file in sorted(models_dir.iterdir(), key=os.path.getmtime):

            if this_file.is_file():
                e_range_name = this_file.stem.replace('-', ' < ').replace('_', ' ')

                Path('models').joinpath(model_name, '{}.joblib'.format(e_range_name))
                trained_models[model_name][e_range_name] = load(this_file)

    return trained_models


def add_predict_column(dtf_e_test, trained_models):
    """
    Add a column to the test dataset with predicted label values.
    Instead of returning the dataset as is divided into energy bins,
    the energy binned datasets is combined into one dataset.

    Parameters
    ----------
    dtf_e_test: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            dict of pandas DataFrames
            Each entry in the dict is a DataFrame containing the data to test with.
            The keys of the dict are the energy ranges of the data.
            Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    trained_models: a nested dict of trained sklearn model per energy range.
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values 3rd dict
        3rd dict:
            'model': trained model for this energy range.
            'train_features': list of variable names trained with.
            'labels': name of the variable used as the labels in the training.
            'test_data_suffix': suffix of the test dataset saved to disk.

    Returns
    -------
    dtf_test_squashed: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            The pandas DataFrame containing the test data and an additional
            column with prediction values. This DataFrame is a combination of the
            energy binned DataFrames given as input
    """

    dtf_test_squashed = dict()
    list_of_dtfs = list()

    for model_name, model in trained_models.items():

        print('Calculating the predictions for the {} model'.format(model_name))

        for this_e_range, this_model in model.items():

            # To keep lines short
            dtf_this_e = dtf_e_test[this_model['test_data_suffix']][this_e_range]
            list_of_dtfs.append(dtf_this_e)

            X_test = dtf_this_e[this_model['train_features']].values
            # Check if any value is inf (found one on a proton file...).
            # If true, change it to a big negative or positive value.
            if np.any(np.isinf(X_test)):
                # Remove positive infs
                X_test[X_test > 999999] = 999999
                # Remove negative infs
                X_test[X_test < -999999] = -999999
            if np.any(np.isnan(X_test)):
                # Remove nans
                X_test[np.isnan(X_test)] = 99999

            y_pred = this_model['model'].predict(X_test)

            dtf_this_e.loc[:, 'y_pred'] = y_pred

        # ignore_index needs to be true, so that the df.loc we run later on does not find
        # several entries for a given index.
        dtf_test_squashed[this_model['test_data_suffix']] = pd.concat(list_of_dtfs)

    return dtf_test_squashed


def partition_event_types(dtf_test, labels, log_e_bins, n_types=3, type_bins='equal statistics',
                          return_partition=False, event_type_bins=None):
    """
    Divide the events into n_types event types in each energy bin.
    The bins defining the types are calculated from the predicted label values,
    assumed to be included already in dtf_test.
    Two lists of types are returned per model and per energy range, one true and one predicted.

    Parameters
    ----------
    dtf_test: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            Pandas DataFrames containing the test data and a column with the predicted values.
    labels: string
        The name of the label column used to train with.
    log_e_bins: array-like
        A list of energy bins in which to divide the data into types.
        The bins are assumed to be the log values of the energy in TeV.
    n_types: int (default=3)
        The number of types to divide the data in.
    type_bins: list of floats or str
        A list defining the bin sizes of each type,
        e.g., [0, 0.2, 0.8, 1] would divide the reconstructed labels dataset (angular error)
        into three bins, best 20%, middle 60% and worst 20%.
        The list must be n_types + 1 long and the first and last values must be zero and one.
        The default is equal statistics bins, given as the default string.
    return_partition: Bool
        If true, a dictionary containing the partition values used for each model
        and each energy bin will be returned.
    event_type_bins: a nested dict of partition values per trained model and energy range
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values=partition values array
    Returns
    -------
    event_types: nested dict
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values=3rd dict
        3rd dict:
            keys=true or reco, values=event type
    """

    event_types = dict()

    if type_bins == 'equal statistics':
        type_bins = np.linspace(0, 1, n_types + 1)
    elif not isinstance(type_bins, list):
        raise ValueError('type_bins must be a list of floats or equal statistics')
    elif len(type_bins) != n_types + 1:
        raise ValueError('type_bins must be n_types + 1 long')
    elif type_bins[0] != 0 or type_bins[-1] != 1:
        raise ValueError('the first and last values of type_bins must be zero and one')
    else:
        pass

    if return_partition:
        event_type_bins = dict()

    for model_name, this_dtf in dtf_test.items():

        event_types[model_name] = dict()
        if return_partition:
            event_type_bins[model_name] = dict()

        print('Calculating event types for the {} model'.format(model_name))

        dtf_e_test = bin_data_in_energy(this_dtf, log_e_reco_bins=log_e_bins)

        for this_e_range, dtf_this_e in dtf_e_test.items():

            event_types[model_name][this_e_range] = defaultdict(list)

            event_types_bins = mstats.mquantiles(
                dtf_this_e['y_pred'],
                type_bins
            )

            # If return_partition is True, then store the event type bins into the container.
            if return_partition:
                event_type_bins[model_name][this_e_range] = event_types_bins
            # If return_partition is False and an event_type_bins container was provided,
            # then use the values from the container.

            if not return_partition and event_type_bins is not None:
                event_types_bins = event_type_bins[model_name][this_e_range]

            for this_value in dtf_this_e['y_pred']:
                this_event_type = np.searchsorted(event_types_bins, this_value)
                if this_event_type < 1:
                    this_event_type = 1
                if this_event_type > n_types:
                    this_event_type = n_types
                event_types[model_name][this_e_range]['reco'].append(this_event_type)

            for this_value in dtf_this_e[labels].values:
                this_event_type = np.searchsorted(event_types_bins, this_value)
                if this_event_type < 1:
                    this_event_type = 1
                if this_event_type > n_types:
                    this_event_type = n_types
                event_types[model_name][this_e_range]['true'].append(this_event_type)

        # When adding the new column, set default value to -2.
        # If there are -2 values at the end, it means those indexes weren't filled here,
        # i.e. those events are not in the defined energy/offset bins.
        this_dtf['event_type'] = -2
        for energy_key in dtf_e_test.keys():
            this_dtf.loc[dtf_e_test[energy_key].index.values, 'event_type'] = (
                event_types[model_name][energy_key]['reco'])

    if return_partition:
        return event_types, event_type_bins
    else:
        return event_types


def partition_event_types_2(dtf_test, labels, log_e_bins, offset_bins=None, n_types=3, type_bins='equal statistics',
                            return_partition=False, event_type_bins=None):
    """
    Divide the events into n_types event types in each energy bin.
    The bins defining the types are calculated from the predicted label values,
    assumed to be included already in dtf_test.
    Two lists of types are returned per model and per energy range, one true and one predicted.

    Parameters
    ----------
    dtf_test: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            Pandas DataFrames containing the test data and a column with the predicted values.
    labels: string
        The name of the label column used to train with.
    log_e_bins: array-like
        A list of energy bins in which to divide the data into types.
        The bins are assumed to be the log values of the energy in TeV.
    offset_bins: array-like
        A list of camera-offset bins (in degrees) in which to divide the data into types.
    n_types: int (default=3)
        The number of types to divide the data in.
    type_bins: list of floats or str
        A list defining the bin sizes of each type,
        e.g., [0, 0.2, 0.8, 1] would divide the reconstructed labels dataset (angular error)
        into three bins, best 20%, middle 60% and worst 20%.
        The list must be n_types + 1 long and the first and last values must be zero and one.
        The default is equal statistics bins, given as the default string.
    return_partition: Bool
        If true, a dictionary containing the partition values used for each model
        and each energy bin will be returned.
    event_type_bins: a nested dict of partition values per trained model and energy range
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values=partition values array
    Returns
    -------
    event_types: nested dict
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values=3rd dict
        3rd dict:
            keys=true or reco, values=event type
    """

    event_types = dict()

    if offset_bins is None:
        offset_bins = [0, 10]

    if type_bins == 'equal statistics':
        type_bins = np.linspace(0, 1, n_types + 1)
    elif not isinstance(type_bins, list):
        raise ValueError('type_bins must be a list of floats or equal statistics')
    elif len(type_bins) != n_types + 1:
        raise ValueError('type_bins must be n_types + 1 long')
    elif type_bins[0] != 0 or type_bins[-1] != 1:
        raise ValueError('the first and last values of type_bins must be zero and one')
    else:
        pass

    if return_partition:
        event_type_bins = dict()

    for model_name, this_dtf in dtf_test.items():
        event_types[model_name] = dict()
        if return_partition:
            event_type_bins[model_name] = dict()

        print('Calculating event types for the {} model'.format(model_name))

        dtf_binned_test = bin_data_in_energy_and_offset(this_dtf, log_e_reco_bins=log_e_bins, offset_bins=offset_bins)

        for this_e_range in dtf_binned_test.keys():
            event_types[model_name][this_e_range] = dict()
            if return_partition:
                event_type_bins[model_name][this_e_range] = dict()
            for this_offset_range, dtf_this_e_offset in dtf_binned_test[this_e_range].items():

                event_types[model_name][this_e_range][this_offset_range] = defaultdict(list)
                event_types[model_name][this_e_range][this_offset_range] = defaultdict(list)

                event_types_bins = mstats.mquantiles(
                    dtf_this_e_offset['y_pred'],
                    type_bins
                )

                # If return_partition is True, then store the event type bins into the container.
                if return_partition:
                    event_type_bins[model_name][this_e_range][this_offset_range] = event_types_bins
                # If return_partition is False and an event_type_bins container was provided,
                # then use the values from the container.
                if not return_partition and event_type_bins is not None:
                    event_types_bins = event_type_bins[model_name][this_e_range][this_offset_range]

                for this_value in dtf_this_e_offset['y_pred']:
                    this_event_type = np.searchsorted(event_types_bins, this_value)
                    if this_event_type < 1:
                        this_event_type = 1
                    if this_event_type > n_types:
                        this_event_type = n_types
                    event_types[model_name][this_e_range][this_offset_range]['reco'].append(this_event_type)

                for this_value in dtf_this_e_offset[labels].values:
                    this_event_type = np.searchsorted(event_types_bins, this_value)
                    if this_event_type < 1:
                        this_event_type = 1
                    if this_event_type > n_types:
                        this_event_type = n_types
                    event_types[model_name][this_e_range][this_offset_range]['true'].append(this_event_type)

        # When adding the new column, set default value to -2.
        # If there are -2 values at the end, it means those indexes weren't filled here,
        # i.e. those events are not in the defined energy/offset bins.
        this_dtf['event_type'] = -2
        # Fill 'event_type' column with the right event types, using the 'reco' dict.
        for energy_key in dtf_binned_test.keys():
            for offset_key in dtf_binned_test[energy_key].keys():
                this_dtf.loc[dtf_binned_test[energy_key][offset_key].index.values, 'event_type'] = (
                    event_types[model_name][energy_key][offset_key]['reco'])

    if return_partition:
        return event_types, event_type_bins
    else:
        return event_types


def predicted_event_types(dtf_e_test, trained_models, n_types=2):
    """
    Get the true and predicted event types for n_types event types.
    Two lists of types are returned per model and per energy range, one true and one predicted.
    This function is meant to be used only for the classification case.

    Parameters
    ----------
    dtf_e_test: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            dict of pandas DataFrames
            Each entry in the dict is a DataFrame containing the data to test with.
            The keys of the dict are the energy ranges of the data.
            Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    trained_models: a nested dict of trained sklearn model per energy range.
            1st dict:
                keys=model names, values=2nd dict
            2nd dict:
                keys=energy ranges, values 3rd dict
            3rd dict:
                'model': trained model for this energy range,
                'train_features': list of variable names trained with.
                'labels': name of the variable used as the labels in the training.
                'test_data_suffix': suffix of the test dataset saved to disk.
    n_types: int (default=2)
            The number of types used in the training.

    Returns
    -------
    event_types: nested dict
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values=3rd dict
        3rd dict:
            keys=true or reco, values=event type
    """

    event_types = dict()

    for model_name, model in trained_models.items():

        event_types[model_name] = dict()

        for this_e_range, this_model in model.items():
            event_types[model_name][this_e_range] = defaultdict(list)
            event_types[model_name][this_e_range] = defaultdict(list)

            # To keep lines short
            dtf_this_e = dtf_e_test[this_model['test_data_suffix']][this_e_range]

            event_types[model_name][this_e_range]['true'] = dtf_this_e[
                'event_type_{:d}'.format(n_types)
            ]

            X_test = dtf_this_e[this_model['train_features']].values
            event_types[model_name][this_e_range]['reco'] = this_model['model'].predict(X_test)

    return event_types


def add_event_types_column(dtf_e, labels, n_types=[2, 3, 4]):
    """
    Divide the events into n_types event types.
    The bins defining the types are calculated from the label values.
    The data will be divided to n number of types with equivalent number of events in each type.
    A column with the type will be added to the DataFrame per entry in the n_types list.

    Parameters
    ----------
    dtf_e: dict of pandas DataFrames
        Each entry in the dict is a DataFrame containing the data.
        The keys of the dict are the energy ranges of the data.
    labels: str
        The variable to use as a basis on which to divide the data.
    n_types: list of ints (default=[2, 3, 4])
        The data will be divided to n number of types
        with equivalent number of events in each type.
        A column with the type will be added to the DataFrame per entry in the n_types list.

    Returns
    -------
    dtf_e: dict of pandas DataFrames
        The same DataFrame as the input but with added columns for event types,
        one column per n_types entry. The column names are event_type_n.
    """

    pd.options.mode.chained_assignment = None

    for this_n_type in n_types:

        for this_e_range, this_dtf in dtf_e.items():

            event_types = list()

            event_types_bins = mstats.mquantiles(
                this_dtf[labels].values,
                np.linspace(0, 1, this_n_type + 1)
            )

            for this_value in this_dtf[labels].values:
                this_event_type = np.searchsorted(event_types_bins, this_value)
                if this_event_type < 1:
                    this_event_type = 1
                if this_event_type > this_n_type:
                    this_event_type = this_n_type
                event_types.append(this_event_type)

            this_dtf.loc[:, 'event_type_{:d}'.format(this_n_type)] = event_types

    return dtf_e


def extract_unique_dataset_names(trained_models):
    """
    Extract all test datasets names necessary for the given trained models.

    Parameters
    ----------
    trained_models: a nested dict of trained sklearn model per energy range.
            1st dict:
                keys=model names, values=2nd dict
            2nd dict:
                keys=energy ranges, values 3rd dict
            3rd dict:
                'model': trained model for this energy range.
                'train_features': list of variable names trained with.
                'labels': name of the variable used as the labels in the training.
                'test_data_suffix': suffix of the test dataset saved to disk.

    Returns
    -------
    dataset_names: set
        Set of unique data set names
    """

    dataset_names = set()
    for model in trained_models.values():
        for this_model in model.values():
            dataset_names.add(this_model['test_data_suffix'])

    return dataset_names


def plot_pearson_correlation(dtf, title):
    """
    Calculate the Pearson correlation between all variables in this DataFrame.

    Parameters
    ----------
    dtf: pandas DataFrame
        The DataFrame containing the data.
    title: str
        A title to add to the olot (will be added to 'Pearson correlation')

    Returns
    -------
    A pyplot instance with the Pearson correlation plot.
    """

    plt.subplots(figsize=[16, 16])
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
    plt.title('Pearson correlations {}'.format(title))
    plt.tight_layout()

    return plt


def plot_test_vs_predict(dtf_e_test, trained_models, trained_model_name):
    """
    Plot true values vs. the predictions of the model for all energy bins.

    Parameters
    ----------
    dtf_e_test: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            dict of pandas DataFrames
            Each entry in the dict is a DataFrame containing the data to test with.
            The keys of the dict are the energy ranges of the data.
            Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    trained_models: a nested dict of one trained sklearn model per energy range.
        1st dict:
            keys=energy ranges, values=2nd dict
        2nd dict:
            'model': trained model for this energy range
            'train_features': list of variable names trained with.
            'labels': Name of the variable used as the labels in the training.
    trained_model_name: str
        Name of the model trained.

    Returns
    -------
    A pyplot instance with the test vs. prediction plot.
    """

    nrows = 5
    ncols = 4

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[14, 18])

    for i_plot, (this_e_range, this_model) in enumerate(trained_models.items()):
        # To keep lines short
        dtf_this_e = dtf_e_test[this_model['test_data_suffix']][this_e_range]

        X_test = dtf_this_e[this_model['train_features']].values
        y_test = dtf_this_e[this_model['labels']].values

        if np.any(np.isinf(X_test)):
            # Remove positive infs
            X_test[X_test > 999999] = 999999
            # Remove negative infs
            X_test[X_test < -999999] = -999999

        y_pred = this_model['model'].predict(X_test)

        ax = axs[int(np.floor(i_plot / ncols)), i_plot % ncols]

        ax.hist2d(y_pred, y_test, bins=(50, 50), cmap=plt.cm.jet)
        ax.plot(
            [min(y_test), max(y_test)], [min(y_test), max(y_test)],
            linestyle='--',
            lw=2,
            color='white'
        )
        ax.set_xlim(np.quantile(y_pred, [0.01, 0.99]))
        ax.set_ylim(np.quantile(y_test, [0.01, 0.99]))
        ax.set_title(this_e_range)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    axs[nrows - 1, ncols - 1].axis('off')
    axs[nrows - 1, ncols - 1].text(
        0.5,
        0.5,
        trained_model_name,
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=18,
        transform=axs[nrows - 1, ncols - 1].transAxes
    )
    plt.tight_layout()

    return plt


def plot_matrix(dtf, train_features, labels, n_types=2, plot_events=20000):
    """
    Plot a matrix of each variable in train_features against another (not all combinations).
    The data is divided to n_types bins of equal statistics based on the labels.
    Each type is plotted in a different colour.
    This function produces mutliple plots, where in each plot a maximum of 5 variables are plotted.
    Unlike in most cases in this code, dtf is the DataFrame itself,
    not a dict of energy ranges. This function should be called per energy bin.

    Parameters
    ----------
    dtf: pandas DataFrames
        A DataFrame to add event types to.
    train_features: list
        List of variable names trained with.
    labels: str
        Name of the variable used as the labels in the training.
    n_types: int (default=2)
        The number of types to divide the data in.
    plot_events: int (default=20000)
        For efficiency, limit the number of events that will be used for the plots

    Returns
    -------
    A list of seaborn.PairGrid instances, each with one matrix plot.
    """

    setStyle()

    # Check if event_type column already present within dtf:
    if "event_type" not in dtf.columns:
        dtf = add_event_type_column(dtf, labels, n_types)

    # Mask out the events without a clear event type
    dtf = dtf[dtf['event_type'] > 0]
    type_colors = {
        1: "#ba2c54",
        2: "#5B90DC",
        3: '#FFAB44',
        4: '#0C9FB3'
    }

    vars_to_plot = np.array_split(
        [labels] + train_features,
        round(len([labels] + train_features) / 5)
    )
    grid_plots = list()
    for these_vars in vars_to_plot:
        grid_plots.append(
            sns.pairplot(
                dtf.sample(n=plot_events),
                vars=these_vars,
                hue='event_type',
                palette=type_colors,
                corner=True
            )
        )
    return grid_plots


def plot_score_comparison(dtf_e_test, trained_models):
    """
    Plot the score of the model as a function of energy.
    #TODO add a similar function that plots from saved scores instead of calculating every time.

    Parameters
    ----------
    dtf_e_test: a nested dict of test datasets per trained model
        1st dict:
            keys=test_data_suffix, values=2nd dict
        2nd dict:
            dict of pandas DataFrames
            Each entry in the dict is a DataFrame containing the data to test with.
            The keys of the dict are the energy ranges of the data.
            Each DataFrame is assumed to contain all 'train_features' and 'labels'.
    trained_models: a nested dict of trained sklearn model per energy range.
        1st dict:
            keys=model names, values=2nd dict
        2nd dict:
            keys=energy ranges, values 3rd dict
        3rd dict:
            'model': dict of trained models for this energy range.
            'train_features': list of variable names trained with.
            'labels': name of the variable used as the labels in the training.
            'test_data_suffix': suffix of the test dataset saved to disk.

    Returns
    -------
    A pyplot instance with the scores plot.
    """

    setStyle()

    fig, ax = plt.subplots(figsize=(8, 6))

    scores = defaultdict(dict)
    energy_bins = extract_energy_bins_centers(trained_models[next(iter(trained_models))].keys())

    for this_model_name, trained_model in trained_models.items():

        print('Calculating scores for {}'.format(this_model_name))

        scores_this_model = list()
        for this_e_range, this_model in trained_model.items():

            # To keep lines short
            dtf_this_e = dtf_e_test[this_model['test_data_suffix']][this_e_range]

            X_test = dtf_this_e[this_model['train_features']].values
            y_test = dtf_this_e[this_model['labels']].values

            if np.any(np.isinf(X_test)):
                # Remove positive infs
                X_test[X_test > 999999] = 999999
                # Remove negative infs
                X_test[X_test < -999999] = -999999

            y_pred = this_model['model'].predict(X_test)

            scores_this_model.append(this_model['model'].score(X_test, y_test))

        scores[this_model_name]['scores'] = scores_this_model
        scores[this_model_name]['energy'] = energy_bins
        ax.plot(
            scores[this_model_name]['energy'],
            scores[this_model_name]['scores'],
            label=this_model_name
        )

    ax.set_xlabel('E [TeV]')
    ax.set_ylabel('score')
    ax.set_xscale('log')
    ax.set_ylim([0, 1])
    ax.legend()
    plt.tight_layout()

    return plt, scores


def plot_confusion_matrix(event_types, trained_model_name, n_types=2):
    """
    Plot the confusion matrix of the model for all energy bins.

    Parameters
    ----------
    event_types: nested dict
        1st dict:
            keys=energy ranges, values=2nd dict
        2nd dict:
            keys=true or reco, values=event type
    trained_model_name: str
        Name of the model used to obtain the reconstructed event types
    n_types: int (default=2)
        The number of types the data was divided in.

    Returns
    -------
    A pyplot instance with the confusion matrix plot.
    """

    # setStyle()

    nrows = 5
    ncols = 4

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[14, 18])

    for i_plot, this_e_range in enumerate(event_types.keys()):
        ax = axs[int(np.floor(i_plot / ncols)), i_plot % ncols]

        cm = confusion_matrix(
            event_types[this_e_range]['true'],
            event_types[this_e_range]['reco'],
            normalize='true',
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt='.1%',
            ax=ax,
            cmap='Blues',
            cbar=False,
            xticklabels=['{}'.format(tick) for tick in np.arange(1, n_types + 1, 1)],
            yticklabels=['{}'.format(tick) for tick in np.arange(1, n_types + 1, 1)]
        )
        ax.set_xlabel('Prediction')
        ax.set_ylabel('True')
        ax.set_title(this_e_range)

    axs[nrows - 1, ncols - 1].axis('off')
    axs[nrows - 1, ncols - 1].text(
        0.5,
        0.5,
        trained_model_name,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=18,
        transform=axs[nrows - 1, ncols - 1].transAxes
    )
    plt.tight_layout()

    return plt


def plot_1d_confusion_matrix(event_types, trained_model_name, n_types=2):
    """
    Plot a one-dimensional confusion matrix of the model for all energy bins.

    Parameters
    ----------
    event_types: nested dict
        1st dict:
            keys=energy ranges, values=2nd dict
        2nd dict:
            keys=true or reco, values=event type
    trained_model_name: str
        Name of the model used to obtain the reconstructed event types
    n_types: int (default=2)
        The number of types the data was divided in.

    Returns
    -------
    A pyplot instance with the one-dimensional confusion matrix plot.
    """

    # setStyle()

    nrows = 5
    ncols = 4

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[14, 10])

    for i_plot, this_e_range in enumerate(event_types.keys()):

        ax = axs[int(np.floor(i_plot / ncols)), i_plot % ncols]

        pred_error = np.abs(
            np.array(event_types[this_e_range]['true']) - np.array(event_types[this_e_range]['reco'])
        )
        frac_pred_error = list()
        for i_type in range(n_types):
            frac_pred_error.append(np.sum(pred_error == i_type) / len(pred_error))

        df = pd.DataFrame(
            {'Prediction accuracy': frac_pred_error},
            index=['correct'] + ['{} off'.format(off) for off in range(1, n_types)]
        )

        sns.heatmap(
            df.T,
            annot=True,
            fmt='.1%',
            ax=ax,
            cmap='Blues',
            square=True,
            cbar=False,
        )
        ax.set_yticklabels(ax.get_yticklabels(), va='center')
        ax.set_title(
            'Score(F1) = {:.2f}\n{}'.format(
                f1_score(
                    event_types[this_e_range]['true'],
                    event_types[this_e_range]['reco'],
                    average='macro'
                ), this_e_range
            )
        )

    axs[nrows - 1, ncols - 1].axis('off')
    axs[nrows - 1, ncols - 1].text(
        0.5,
        0.5,
        trained_model_name,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=18,
        transform=axs[nrows - 1, ncols - 1].transAxes
    )
    plt.tight_layout()

    return plt
