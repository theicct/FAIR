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
    all = all[~all['scenario'].isin(['Lower limit', 'Upper limit', 'Mean', 'ssp119'])]

    # Calculate a mean and upper/lower 95th percentile
    avg = all.groupby(["timebounds"], as_index=False, dropna=False).agg({"temp": "mean"})
    avg = avg.rename(columns={'temp': 'Mean'})

    quantiles = all.groupby('timebounds')['temp'].quantile([0.025, 0.975]).unstack()
    quantiles.columns = ['lower_95th', 'upper_95th']

    avg_and_quantiles = avg.merge(quantiles, on='timebounds', how='left')
    # Convert to long with 'Scenario'
    avg_and_quantiles_long = avg_and_quantiles.melt(id_vars=['timebounds'], var_name='scenario', value_name='temp')

    # Concatenate
    all = pd.concat([avg_and_quantiles_long, all], ignore_index=True)

    # avg_and_quantiles = avg_and_quantiles.drop(columns=['lower_95th', 'upper_95th'])

    # Merge the two dataframes
    all = all.merge(avg_and_quantiles, on='timebounds', how='left')
    # Subtract the mean from all scenarios
    all['temp_anomaly'] = all['temp'] - all['Mean']

    return all


def make_plot(all):
    """Make a plot of the results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mean = all[all['scenario'] == 'Mean'].copy()
    lower = all[all['scenario'] == 'lower_95th'].copy()[['timebounds', 'CAT']]
    upper = all[all['scenario'] == 'upper_95th'].copy()[['timebounds', 'CAT']]
    # Merge the mean and quantiles
    mean = mean.merge(lower, on='timebounds', how='left', suffixes=('', '_lower'))
    mean = mean.merge(upper, on='timebounds', how='left', suffixes=('', '_upper'))

    # Plot the mean and 95th percentiles
    ax.plot(mean['timebounds'], mean['CAT'], label='Mean', color='red')
    ax.fill_between(mean['timebounds'], mean['CAT_lower'],
                    mean['CAT_upper'],
                    color='blue', alpha=0.2, label='95th Percentile')

    # Annotate the 2050 values
    mean_2050 = mean[mean['timebounds'] == 2050].copy()

    # Calculate the change from the mean and percent differences
    mean_2050['CAT_change_lower'] = mean_2050['CAT_lower'] - mean_2050['CAT']
    mean_2050['CAT_change_lower_percent'] = (mean_2050['CAT_change_lower'] / mean_2050['CAT'].mean()) * 100

    mean_2050['CAT_change_upper'] = mean_2050['CAT_upper'] - mean_2050['CAT']
    mean_2050['CAT_change_upper_percent'] = (mean_2050['CAT_change_upper'] / mean_2050['CAT'].mean()) * 100

    # Annotate the 2025 values
    mean_2025 = mean[mean['timebounds'] == 2025].copy()

    # Calculate the change from the mean and percent differences
    mean_2025['CAT_change_lower'] = mean_2025['CAT_lower'] - mean_2025['CAT']
    mean_2025['CAT_change_lower_percent'] = (mean_2025['CAT_change_lower'] / mean_2025['CAT'].mean()) * 100

    mean_2025['CAT_change_upper'] = mean_2025['CAT_upper'] - mean_2025['CAT']
    mean_2025['CAT_change_upper_percent'] = (mean_2025['CAT_change_upper'] / mean_2025['CAT'].mean()) * 100

    ax.annotate(f"{mean_2050['CAT'].values[0]:.4f}°C",
                xy=(2050, mean_2050['CAT'].values[0]),
                xytext=(2052, mean_2050['CAT'].values[0]),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10)
    ax.annotate(f"{mean_2050['CAT_lower'].values[0]:.4f}°C",
                xy=(2050, mean_2050['CAT_lower'].values[0]),
                xytext=(2052, mean_2050['CAT_lower'].values[0]),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10)
    ax.annotate(f"{mean_2050['CAT_upper'].values[0]:.4f}°C",
                xy=(2050, mean_2050['CAT_upper'].values[0]),
                xytext=(2052, mean_2050['CAT_upper'].values[0]),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10)

    ax.annotate(f"{mean_2025['CAT'].values[0]:.4f}°C",
                xy=(2025, mean_2025['CAT'].values[0]),
                xytext=(2027, mean_2025['CAT'].values[0]),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10)
    ax.annotate(f"{mean_2025['CAT_lower'].values[0]:.4f}°C",
                xy=(2025, mean_2025['CAT_lower'].values[0]),
                xytext=(2027, mean_2025['CAT_lower'].values[0]),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10)
    ax.annotate(f"{mean_2025['CAT_upper'].values[0]:.4f}°C",
                xy=(2025, mean_2025['CAT_upper'].values[0]),
                xytext=(2027, mean_2025['CAT_upper'].values[0]+0.01),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10)

    ax.annotate(f"({mean_2050['CAT_change_lower'].values[0]:.4f}°C, \n({round(mean_2050['CAT_change_lower_percent'].values[0])}%)",
                xy=(2050, mean_2050['CAT_lower'].values[0]),
                xytext=(2052, mean_2050['CAT_lower'].values[0]-0.004),
                fontsize=10)

    ax.annotate(f"(+{mean_2050['CAT_change_upper'].values[0]:.4f}°C, \n+{round(mean_2050['CAT_change_upper_percent'].values[0])}%)",
                xy=(2050, mean_2050['CAT_upper'].values[0]),
                xytext=(2052, mean_2050['CAT_upper'].values[0]-0.004),
                fontsize=10)

    ax.annotate(f"({mean_2025['CAT_change_lower'].values[0]:.4f}°C, \n{round(mean_2025['CAT_change_lower_percent'].values[0])}%)",
                xy=(2025, mean_2025['CAT_lower'].values[0]),
                xytext=(2027, mean_2025['CAT_lower'].values[0]-0.004),
                fontsize=10)
    ax.annotate(f"(+{mean_2025['CAT_change_upper'].values[0]:.4f}°C, \n+{round(mean_2025['CAT_change_upper_percent'].values[0])}%)",
                xy=(2025, mean_2025['CAT_upper'].values[0]),
                xytext=(2028, mean_2025['CAT_upper'].values[0]+0.006),
                fontsize=10)

    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly (°C)')
    ax.set_title("Contrail-Attributable Global Warming Projections with Monte Carlo Simulations")
    ax.legend()

    # Save as PDF and PNG
    plt.savefig('figures/contrail_attribution.png', dpi=300)
    plt.savefig('figures/contrail_attribution.pdf')


def main():
    """Read all input data from the 'sens' directory, combine them, and save the results."""
    batch_size = 100

    # all = collect_data(batch_size)

    # all.to_csv(f'../all_sensitivity_runs_{batch_size}.csv')
    all = pd.read_csv(f'../all_sensitivity_runs_{batch_size}.csv')
    # Filter to 2020-2050
    all = all[(all['timebounds'] >= 2020) & (all['timebounds'] <= 2050)]

    # Drop lower_95th_x, upper_95th_x, lower_95th_y, upper_95th_y
    all = all.drop(columns=['lower_95th_x', 'upper_95th_x', 'lower_95th_y', 'upper_95th_y', 'Unnamed: 0', 'Mean'])

    zero = pd.read_csv('../temperature_all.csv')
    zero = zero[(zero['timebounds'] >= 2020) & (zero['timebounds'] <= 2050)]
    # Add the 'ssp119_contrails_Aviation_striving' scenario to the sensitivity runs
    zero = zero[zero['scenario'] == 'ssp119_contrails_Aviation_striving'].copy()
    zero['scenario'] = 'ssp119_contrails_Aviation_striving'
    zero = zero[['timebounds', 'scenario', 'temp_zero']].rename(columns={'temp_zero': 'temp'})
    zero['scenario'] = 'Zero contrails'

    all = pd.concat([zero, all], ignore_index=True)

    zero = zero.rename(columns={'temp': 'temp_zero'}).drop(columns='scenario')
    all = all.merge(zero, on=['timebounds'], how='left')
    all['CAT'] = all['temp'] - all['temp_zero']

    make_plot(all)

    # Rename 'CAT' to 'Contrail_Attributable_Temperature_Anomaly_C'
    all = all.rename(columns={
        'CAT': 'contrail_attributable_temperature_anomaly_C',
        'temp_anomaly': 'temperature_anomaly_from_mean_C',
    }).drop(columns='temp_zero')

    # Save the results
    all.to_csv(f'../all_sensitivity_runs_{batch_size}_with_intervals.csv', index=False)


if __name__ == "__main__":
        main()
