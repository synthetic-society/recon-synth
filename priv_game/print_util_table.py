import pandas as pd
from math import ceil, log10

def print_utility(df, metric, synth_models):
    for data_name in ['acs', 'fire']:
        print(data_name)
        print('--------')
        df_data = df[df['data_name'] == data_name]
        for m in [10, 100, 1000, 10000, 100000, 1000000]:
            print(f'$10^{int(log10(m))}$', end="")
            for synth_model in synth_models:
                rel_mean = df_data[(df_data['synth_model'] == synth_model) & (df_data['synth_size'] == m)][metric].to_numpy()[0]
                rel_std = df_data[(df_data['synth_model'] == synth_model) & (df_data['synth_size'] == m)][f'{metric}_std'].to_numpy()[0]

                rel_mean_str = f'{rel_mean:.2f}' # normal rounding operation
                rel_std_str = f'{ceil(rel_std * 100) / 100:.2f}' # round up std dev

                print(f' & ${rel_mean_str} \pm {rel_std_str}$', end="")
            print(' \\\\')
        print()

if __name__ == '__main__':
    df = pd.read_csv('results/v5/results_utility.csv')
    util = 'rel_mean' # avg_tvd or rel_mean
    dp = True 
    nondp_name = 'BayNet_3parents' # BayNet_3parents
    dp_name = 'PrivBayes_3parents' # PrivBayes_3parents

    if not dp:
        synth_models = ['NonPrivate', 'BayNet_3parents', 'RAP_2Kiters', 'CTGAN', 'IndHist']
        print_utility(df, util, synth_models)
        print()
    else:
        synth_models = [f'{dp_name}_1eps', f'{dp_name}_10eps', f'{dp_name}_100eps', nondp_name]
        print_utility(df, util, synth_models)