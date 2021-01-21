import argparse
from pathlib import Path
import event_classes

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'An example script how plot some control plots.'
        )
    )

    args = parser.parse_args()

    dl2_file_name = '/lustre/fs21/group/cta/users/maierg/analysis/AnalysisData/uploadDL2/Paranal_20deg/gamma_onSource.S.3HB9-FD_ID0.eff-0.root'
    dtf = event_classes.extract_df_from_dl2(dl2_file_name)
    dtf_e = event_classes.bin_data_in_energy(dtf)

    labels, train_features = event_classes.nominal_labels_train_features()

    Path('plots/matrices').mkdir(parents=True, exist_ok=True)
    Path('plots/pearson').mkdir(parents=True, exist_ok=True)

    for this_e_range, this_dtf in dtf_e.items():

        e_range_name = this_e_range.replace(' < ', '-').replace(' ', '_')

        plt = event_classes.plot_pearson_correlation(this_dtf, this_e_range)
        plt.savefig('plots/pearson/pearson_{}.pdf'.format(e_range_name))

        grids = event_classes.plot_matrix(
            this_dtf,
            train_features,
            labels,
            n_types=2
        )

        for i_grid, this_grid in enumerate(grids):
            this_grid.tight_layout()
            this_grid.savefig('plots/matrices/matrix_{}_{}.png'.format(e_range_name, i_grid + 1))
