import argparse
from event_types import event_types

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Read the data from the DL2 root files and save them in pickle format'
            'for fast reading.'
        )
    )

    args = parser.parse_args()

    dl2_file_path = (
        # Paranal
        # '/lustre/fs22/group/cta/users/maierg/analysis/AnalysisData/'
        # 'prod5-Paranal-20deg-sq10-LL-DL2plus/EffectiveAreas/'
        # 'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210610-V3/BDT.50h-V3.g20210610/'

        # La Palma
        '/lustre/fs22/group/cta/users/maierg/analysis/AnalysisData/'
        'prod5b-LaPalma-20deg-sq10-LL-DL2plus/EffectiveAreas/'
        'EffectiveArea-50h-ID0-NIM2LST2MST2SST2SCMST2-g20210610-V3/BDT.50h-V3.g20210610/'
    )
    particles = ['gamma_onSource', 'gamma_cone', 'electron', 'proton']
    bins_off_axis_angle = [1, 2, 3, 4, 5]
    layout_desc = 'N.D25-4LSTs09MSTs-MSTN'  # LaPalma
    # layout_desc = 'S-LM6C5ax-4LSTs14MSTs40SSTs-MSTF'  # Paranal

    dl2_file_list = list()
    for particle in particles:
        for off_axis_bin in bins_off_axis_angle:
            if off_axis_bin > 0 and particle == 'gamma_onSource':
                continue
            dl2_file_list.append(f'{particle}.{layout_desc}_ID0.eff-{off_axis_bin}.root')

    for filename in dl2_file_list:
        dtf = event_types.extract_df_from_dl2('{}/{}'.format(dl2_file_path, filename))
        event_types.save_dtf(dtf, filename.replace('.root', ''))
