import numpy as np
import pandas as pd

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

# Set pandas display options so that we can see all columns when debugging
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


def set_up_fair(scenarios):

    # Set up FAIR model --------------------------------------------------------------------------------------------------
    f = FAIR(ch4_method='thornhill2021')
    f.define_time(1750, 2100, 1)
    f.define_scenarios(scenarios)

    # Define a series of climate configurations --------------------------------------------------------------------------
    df = pd.read_csv("../tests/test_data/4xCO2_cummins_ebm3.csv")
    models = df['model'].unique()
    configs = []

    for imodel, model in enumerate(models):
        for run in df.loc[df['model']==model, 'run']:
            configs.append(f"{model}_{run}")
    f.define_configs(configs)

    # Set up FaIR species configs ----------------------------------------------------------------------------------------

    species, properties = read_properties()

    # Make exception for contrails
    properties['Contrails'] = {'type': 'contrails', 'input_mode': 'forcing', 'greenhouse_gas': False,
                               'aerosol_chemistry_from_emissions': False, 'aerosol_chemistry_from_concentration': False}
    # Adjust O3 so that NOx from aviation will force it
    # properties['Ozone'] = {'type': 'ozone', 'input_mode': 'calculated', 'greenhouse_gas': True,
    #                        'aerosol_chemistry_from_emissions': True, 'aerosol_chemistry_from_concentration': True}
    # properties['CH4'] = {'type': 'ch4', 'input_mode': 'emissions', 'greenhouse_gas': True,
    #                        'aerosol_chemistry_from_emissions': True, 'aerosol_chemistry_from_concentration': True}

    # Save a human-readable version of the properties for reference
    properties_df = pd.DataFrame(properties).transpose().reset_index()
    properties_df = properties_df.rename(columns={'index': 'Variable'})
    properties_df.to_csv('properties/properties.csv', index=False)

    f.define_species(species, properties)

    f.allocate()

    f.fill_species_configs()
    # fill(f.species_configs['ozone_radiative_efficiency_array']
    # fill(f.species_configs['forcing_reference_concentration'], 30, specie='Ozone')
    fill(f.species_configs['unperturbed_lifetime'], 10.8537568, specie='CH4')
    fill(f.species_configs['baseline_emissions'], 19.01978312, specie='CH4')
    fill(f.species_configs['baseline_emissions'], 0.08602230754, specie='N2O')

    # Set up other model parameters to align with CMIP6 -------------------------------------------------------------------

    df_volcanic = pd.read_csv('../tests/test_data/volcanic_ERF_monthly_175001-201912.csv', index_col='year')
    df_volcanic[1750:].head()

    f.fill_from_rcmip()

    # overwrite volcanic
    volcanic_forcing = np.zeros(351)
    volcanic_forcing[:271] = df_volcanic[1749:].groupby(np.ceil(df_volcanic[1749:].index) // 1).mean().squeeze().values
    fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")  # sometimes need to expand the array

    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)


    df = pd.read_csv("../tests/test_data/4xCO2_cummins_ebm3.csv")
    models = df['model'].unique()

    seed = 1355763

    for config in configs:
        model, run = config.split('_')
        condition = (df['model']==model) & (df['run']==run)
        fill(f.climate_configs['ocean_heat_capacity'], df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
        fill(f.climate_configs['ocean_heat_transfer'], df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
        fill(f.climate_configs['deep_ocean_efficacy'], df.loc[condition, 'epsilon'].values[0], config=config)
        fill(f.climate_configs['gamma_autocorrelation'], df.loc[condition, 'gamma'].values[0], config=config)
        fill(f.climate_configs['sigma_eta'], df.loc[condition, 'sigma_eta'].values[0], config=config)
        fill(f.climate_configs['sigma_xi'], df.loc[condition, 'sigma_xi'].values[0], config=config)
        fill(f.climate_configs['stochastic_run'], True, config=config)
        fill(f.climate_configs['use_seed'], True, config=config)
        fill(f.climate_configs['seed'], seed, config=config)

        seed = seed + 399

    return f


def add_zero_scenarios(df):

    # zero = df[df['scenario']=='ssp119_zero_tra'].copy()[['timebounds', 'temp']]
    #
    # # Merge
    # df = df.merge(zero, on=['timebounds'], how='left', suffixes=['', '_zero'])
    #
    # # Calculate the transportation attributable emissions by subtracting the zero_scenarios from the striving_scenarios
    # df['Transportation-Attributable Temperature'] = df['temp'] - df['temp_zero']

    zero_scenarios = df[df['scenario'].str.endswith('_zero')].copy()[['timebounds', 'scenario', 'temp']]
    striving_scenarios = df[~(df['scenario'].str.endswith('_zero'))].copy()

    # Replace '_zero' with '_striving' in the scenario names
    zero_scenarios['scenario'] = zero_scenarios['scenario'].str.replace('_zero', '_striving')

    # Merge the zero_scenarios with the striving_scenarios
    striving_scenarios = striving_scenarios.merge(zero_scenarios, on=['timebounds', 'scenario'], how='left', suffixes=['', '_zero'])

    # Calculate the transportation attributable emissions by subtracting the zero_scenarios from the striving_scenarios
    striving_scenarios['BAU_temp_attribution'] = striving_scenarios['temp_ssp119'] - striving_scenarios['temp_zero']
    striving_scenarios['Striving_temp_attribution'] = striving_scenarios['temp'] - striving_scenarios['temp_zero']

    return striving_scenarios


def clean_temp_output(f):
    """
    Takes the output from the FaIR model and averages it across all configurations.
    """
    temp = f.temperature.to_dataframe('temp').reset_index()

    # Average across all configs and set to layer = 0 for surface
    temp = temp[temp.pop('layer') == 0]

    # Average across all configs
    df_avg = temp.groupby(["timebounds", "scenario"], as_index=False, dropna=False).agg({"temp": "mean"})
    print(df_avg['scenario'].unique())

    # Add an 'Avoided' column as the difference from ssp119
    ssp119 = df_avg[df_avg['scenario'] == 'ssp119'].copy()
    df_avg = df_avg.merge(ssp119, on='timebounds', suffixes=('', '_ssp119'))
    df_avg['Avoided'] = df_avg['temp_ssp119'] - df_avg['temp']

    # Add a column to indicate SLCP or Kyoto
    # If scneario name contains 'CO2', 'CH4', or 'N2O', label 'Kyoto; otherwise 'SLCP'
    # Label 'Pollutant' based on scenario
    df_avg['Pollutant'] = np.where(df_avg['scenario'].str.contains('CO2'), 'CO2', 'non-CO2')
    # df_avg['Pollutant'] = np.where(df_avg['scenario'].str.contains('CO2|CH4|N2O'), 'Kyoto', 'SLCP')

    df_avg['Actual Pollutant'] = df_avg['scenario'].str.split('_', expand=True)[1]

    # Remove the Pollutant values for 'ssp119', 'ssp119_striving', and 'ssp119_zero_tra'
    df_avg.loc[df_avg['scenario'].isin(['ssp119', 'ssp119_striving', 'ssp119_zero_tra']), 'Pollutant'] = np.nan
    df_avg.loc[df_avg['scenario'].isin(['ssp119', 'ssp119_striving', 'ssp119_zero_tra']), 'Actual Pollutant'] = np.nan

    # Assign 'Sector' based on scenario
    df_avg['Sector'] = np.select(
        [
            df_avg['scenario'].str.contains('Aviation'),
            df_avg['scenario'].str.contains('off-road'),
            df_avg['scenario'].str.contains('on-road'),
            df_avg['scenario'].str.contains('Marine')
        ],
        ['Aviation', 'off-road', 'on-road', 'Marine'],
        default='Other'  # Default value if no condition is met
    )

    df_avg = add_zero_scenarios(df_avg)

    # # Create a summed version for SLCPs and Kyoto
    # all_slcps_striving = df_avg[df_avg['Pollutant']=='SLCP'].groupby(['timebounds'], as_index=False).agg({'Avoided': 'sum'})
    # all_slcps_striving['scenario'] = 'Striving_all'
    # all_slcps_striving['Pollutant'] = 'SLCP'
    # all_kyoto_striving = df_avg[df_avg['Pollutant']=='Kyoto'].groupby(['timebounds'], as_index=False).agg({'Avoided': 'sum'})
    # all_kyoto_striving['scenario'] = 'Striving_all'
    # all_kyoto_striving['Pollutant'] = 'Kyoto'
    #
    # # Copy ZeroTra and Striving scenarios to the two Pollutants
    # ztra1 = df_avg[df_avg['scenario']=='ssp119_zero_tra'].copy()
    # ztra1['Pollutant'] = 'Kyoto'
    # ztra2 = ztra1.copy()
    # ztra1['Pollutant'] = 'SLCP'
    #
    # df_avg = pd.concat([df_avg, ztra1, ztra2], ignore_index=True)
    #
    # all_subs = pd.concat([all_slcps_striving, all_kyoto_striving], ignore_index=True)
    # # Calculate the 'temp' column as ssp119 + the sum of the Avoided column
    # all_subs = all_subs.merge(ssp119, on='timebounds', suffixes=('', '_ssp119'))
    # all_subs['temp'] = all_subs['temp'] - all_subs['Avoided']
    #
    # # # Concatenate the summed versions to the original dataframe
    # df_avg = pd.concat([df_avg,  all_subs], ignore_index=True)

    return df_avg


def main():
    """Set up FaIR, run it, and save results."""
    # Read input data
    forcing = pd.read_csv('../inputs/final/rcmip-radiative-forcing-annual-means-v5-1-0.csv')
    ems = pd.read_csv('../inputs/final/rcmip-emissions-annual-means-v5-1-0.csv')
    conc = pd.read_csv('../inputs/final/rcmip-concentrations-annual-means-v5-1-0.csv')

    # Define scenarios
    scenarios = list(set(pd.concat([forcing['Scenario'], ems['Scenario'], conc['Scenario']]).unique()))

    # Set up FaIR
    f = set_up_fair(scenarios)

    # Run the model
    f.run()

    # Retrieve the temperature results in a clean format
    temp = clean_temp_output(f)

    # Save the results
    temp.to_csv('Results/temperature_all.csv', index=False)


if __name__ == "__main__":
        main()
