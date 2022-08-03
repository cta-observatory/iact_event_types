import argparse
from pathlib import Path
from event_types import event_types

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'An example script how plot some control plots.'
        )
    )

    args = parser.parse_args()

    start_from_DL2 = False
    if start_from_DL2:
        # Prod5
        dl2_file_name = (
            '/lustre/fs22/group/cta/users/maierg/analysis/AnalysisData/'
            'prod5-Paranal-20deg-sq08-LL/EffectiveAreas/'
            'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210921-V3/BDT.DL2.50h-V3.g20210921/'
            'gamma_onSource.S.BL-4LSTs25MSTs70SSTs-MSTF_ID0.eff-0.root'
        )
        dtf = event_types.extract_df_from_dl2(dl2_file_name)
    else:
        dtf = event_types.load_dtf('gamma_onSource.N.D25-4LSTs09MSTs-MSTN_ID0.eff-0')

    dtf_e = event_types.bin_data_in_energy(dtf)

    labels, train_features = event_types.nominal_labels_train_features()

    Path('plots/matrices').mkdir(parents=True, exist_ok=True)
    Path('plots/pearson').mkdir(parents=True, exist_ok=True)

    for this_e_range, this_dtf in dtf_e.items():

        e_range_name = this_e_range.replace(' < ', '-').replace(' ', '_')

        plt = event_types.plot_pearson_correlation(this_dtf, this_e_range)
        plt.savefig('plots/pearson/pearson_{}.pdf'.format(e_range_name))

        grids = event_types.plot_matrix(
            this_dtf,
            train_features,
            labels,
            n_types=2
        )

        for i_grid, this_grid in enumerate(grids):
            this_grid.tight_layout()
            this_grid.savefig('plots/matrices/matrix_{}_{}.png'.format(e_range_name, i_grid + 1))
