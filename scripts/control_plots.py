import argparse
from pathlib import Path

from event_types import event_types

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=("An example script how plot some control plots."))

    args = parser.parse_args()

    start_from_DL2 = False
    bins_off_axis_angle = [0, 1, 2, 3, 4, 5]
    layout_desc = "N.D25-4LSTs09MSTs-MSTN"  # LaPalma
    # layout_desc = 'S-M6C8aj-14MSTs37SSTs-MSTF'  # Paranal
    if start_from_DL2:
        dl2_file_names = [
            "../../data/LongFiles/North/"
            f"gamma_cone.{layout_desc}_ID0.eff-{i}.root" for i in bins_off_axis_angle
        ]
        dtf = event_types.extract_df_from_multiple_dl2(dl2_file_names)
    else:
        dtf = event_types.load_dtf(
            [f"gamma_cone.{layout_desc}_ID0.eff-{i}" for i in bins_off_axis_angle]
        )

    dtf_e = event_types.bin_data_in_energy(dtf)

    labels, train_features = event_types.nominal_labels_train_features()

    Path("plots/matrices").mkdir(parents=True, exist_ok=True)
    Path("plots/pearson").mkdir(parents=True, exist_ok=True)

    for this_e_range, this_dtf in dtf_e.items():
        # Select the dtf for the first entry in the dictionary (there is only one entry for offset angle)
        this_dtf = this_dic[list(this_dic.keys())[0]]

        e_range_name = this_e_range.replace(" < ", "-").replace(" ", "_")

        plt = event_types.plot_pearson_correlation(this_dtf, this_e_range)
        plt.savefig("plots/pearson/pearson_{}.pdf".format(e_range_name))

        grids = event_types.plot_matrix(this_dtf, train_features, labels, n_types=2)

        for i_grid, this_grid in enumerate(grids):
            this_grid.tight_layout()
            this_grid.savefig("plots/matrices/matrix_{}_{}.png".format(e_range_name, i_grid + 1))
