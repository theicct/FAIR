import pandas as pd
import os
import matplotlib.pyplot as plt

# Set pandas display options so that we can see all columns when debugging
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


def collect_data(batch_size):

    # Read all input data
    all = []
    for filename in os.listdir(f'{batch_size}'):
        print(filename)
        if filename.endswith('.csv'):
            filepath = os.path.join(f'{batch_size}', filename)
            df = pd.read_csv(filepath)
            all.append(df)

    all = pd.concat(all, ignore_index=True)

    # Drop 'Lower limit', 'Upper limit', and 'Mean' Scenarios
    all = all[~all['scenario'].isin(['Lower limit', 'Upper limit', 'Mean', 'ssp119', 'ssp119_BAU'])]

    ssp119_zero_tra = all[all['scenario'] == 'ssp119_zero_tra'].copy().drop_duplicates() # there will be a zero scenario for each run
    # Ensure no duplicate years in ssp119_zero_tra
    assert len(ssp119_zero_tra['timebounds'].unique()) == len(ssp119_zero_tra), "Duplicate years in ssp119_zero_tra"
    all = all[all['scenario'] != 'ssp119_zero_tra'].copy()
    all = all.merge(ssp119_zero_tra[['timebounds', 'temp']].rename(columns={'temp': 'temp_zero'}), on='timebounds', how='left')
    all['Attributable temp'] = all['temp'] - all['temp_zero']
    all = all.drop(columns=['temp_zero'])

    # Calculate a mean and upper/lower 95th percentile
    avg = all.groupby(["timebounds"], as_index=False, dropna=False).agg({"Attributable temp": "mean"})
    avg = avg.rename(columns={'Attributable temp': 'Mean'})

    quantiles = all.groupby('timebounds')['Attributable temp'].quantile([0.025, 0.975]).unstack()
    quantiles.columns = ['lower_95th', 'upper_95th']

    avg_and_quantiles = avg.merge(quantiles, on='timebounds', how='left')
    # Convert to long with 'Scenario'
    avg_and_quantiles_long = avg_and_quantiles.melt(id_vars=['timebounds'], var_name='scenario', value_name='temp')

    # Concatenate
    # all = pd.concat([avg_and_quantiles_long, all], ignore_index=True)

    # avg_and_quantiles = avg_and_quantiles.drop(columns=['lower_95th', 'upper_95th'])

    # Merge the two dataframes
    # all = all.merge(avg_and_quantiles, on='timebounds', how='left')

    return all, avg_and_quantiles



def main():
    """Read all input data from the 'sens' directory, combine them, and save the results."""
    batch_size = 10

    all, avg_and_quantiles = collect_data(batch_size)

    all.to_csv(f'all_sensitivity_runs_{batch_size}.csv', index=False)
    avg_and_quantiles.to_csv(f'avg_sensitivity_runs_{batch_size}.csv', index=False)


if __name__ == "__main__":
        main()
