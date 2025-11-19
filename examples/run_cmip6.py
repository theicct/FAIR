import numpy as np
import pandas as pd
import time

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

# Set pandas display options so that we can see all columns when debugging
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Define user parameters consistent with time bounds and baseline scenario names
MIN_YEAR = 1940
MAX_YEAR = 2050
BASE_SCEN = 'Historical Trends'

# Define output paths. NOTE: "lever", if included in the TEMP_OUT filename, is a special keyword that triggers lever
# attribution postprocessing (renormalization). If running a lever-based scenario, ensure that "lever" is in the filename
# and INTERMEDIATE_OUT is additionally defined
TEMP_OUT = 'Results/temperature_Sep17_2025.csv'
# INTERMEDIATE_OUT = 'Results/temperature_Sep17_2025_no_normalization.csv'
CLEAN_OUT = 'Results/temperature_Sep17_2025_summary.csv'

if 'lever' in TEMP_OUT or 'Lever' in TEMP_OUT:
    POSTPROCESS_LEVERS = True
else:
    POSTPROCESS_LEVERS = False

print(f"Running a lever attribution scenario (with renormalization): {POSTPROCESS_LEVERS}")

# Internal input paths
FORCING_IN = '../inputs/final/rcmip-radiative-forcing-annual-means-v5-1-0.csv'
EMS_IN = '../inputs/final/rcmip-emissions-annual-means-v5-1-0.csv'
CONC_IN = '../inputs/final/rcmip-concentrations-annual-means-v5-1-0.csv'

def set_up_fair(scenarios):
    print("Setting up FaIR model...")

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
    # properties['Contrails'] = {'type': 'contrails', 'input_mode': 'forcing', 'greenhouse_gas': False,
    #                            'aerosol_chemistry_from_emissions': False, 'aerosol_chemistry_from_concentration': False}
    properties['Contrails']['input_mode'] = 'forcing'
    properties['Stratospheric water vapour']['input_mode'] = 'forcing'

    # Save a human-readable version of the properties for reference
    properties_df = pd.DataFrame(properties).transpose().reset_index()
    properties_df = properties_df.rename(columns={'index': 'Variable'})
    # properties_df.to_csv('diagnostic/properties.csv', index=False)

    f.define_species(species, properties)

    f.allocate()

    f.fill_species_configs()
    # fill(f.species_configs['ozone_radiative_efficiency_array']
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


def add_zero_scenarios(df, scenarios):

    # Divide scenarios into total and pollutant specific
    # Add 'ssp119_' prefix
    scenarios = [f'ssp119_{s}' for s in scenarios]
    # Set the 'Actual Pollutant' value for these scenarios to 'All'
    df.loc[df['scenario'].isin(scenarios), 'Actual Pollutant'] = 'All'
    df.loc[df['scenario'].isin(['ssp119_zero_tra']), 'Actual Pollutant'] = 'All'

    zero_scenarios = df[df['scenario'].str.contains('_zero')].copy()[['timebounds', 'Actual Pollutant', 'temp']]
    # striving_scenarios = df[~(df['scenario'].str.contains('_zero'))].copy()

    # Merge the zero_scenarios with the striving_scenarios
    striving_scenarios = df.merge(zero_scenarios, on=['timebounds', 'Actual Pollutant'],
                                                  how='left', suffixes=['', '_zero'])

    # Calculate the transportation attributable emissions by subtracting the zero_scenarios from the striving_scenarios
    striving_scenarios['BAU_temp_attribution'] = striving_scenarios['temp_ssp119'] - striving_scenarios['temp_zero']
    striving_scenarios['Striving_temp_attribution'] = striving_scenarios['temp'] - striving_scenarios['temp_zero']

    return striving_scenarios


def clean_temp_output(f, vizcon_scenarios):
    """
    Takes the output from the FaIR model and averages it across all configurations.
    """
    print("Postprocessing outputs...")
    temp = f.temperature.to_dataframe('temp').reset_index()

    # forc = f.forcing.to_dataframe('forcing').reset_index()
    # forc = forc[(forc['timebounds'] >= MIN_YEAR) & (forc['timebounds'] <= MAX_YEAR)]
    # forc = forc.groupby(['timebounds', 'scenario', 'specie']).mean(numeric_only=True)
    # forc.to_csv('diagnostic/out_forcing.csv')
    #
    # ems = f.emissions.to_dataframe('ems').reset_index()
    # ems = ems[(ems['timepoints'] >= MIN_YEAR) & (ems['timepoints'] <= MAX_YEAR)]
    # ems = ems.groupby(['timepoints', 'scenario', 'specie']).mean(numeric_only=True)
    # ems.to_csv('diagnostic/out_ems.csv')

    # Average across all configs and set to layer = 0 for surface
    temp = temp[temp.pop('layer') == 0]

    # Filter to MIN_YEAR-MAX_YEAR
    temp = temp[(temp['timebounds'] >= MIN_YEAR) & (temp['timebounds'] <= MAX_YEAR)]

    # Average across all configs
    df_avg = temp.groupby(["timebounds", "scenario"], as_index=False, dropna=False).agg({"temp": "mean"})

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

    df_avg = add_zero_scenarios(df_avg, vizcon_scenarios)

    # Add 'Vizcon Scenario' Column
    for scen in vizcon_scenarios:
        df_avg.loc[df_avg['scenario'].str.contains(scen), 'Vizcon Scenario'] = scen

    # Rename BAU scenario to Baseline
    df_avg.loc[df_avg['Vizcon Scenario'] == 'BAU', 'Vizcon Scenario'] = BASE_SCEN

    df_avg = df_avg.rename(columns={'Striving_temp_attribution': 'Attributable Warming', 'temp': 'Global Temperature Anomaly'})

    return df_avg


def adjust_taxation_scenario(temp):
    """
    Ensure that the taxation scenario is relative to Baseline.
    """
    no_pricing = temp[temp['Vizcon Scenario'].isin(['FB without pricing', 'FB without pricing'])].copy()
    all_levers = temp[temp['Vizcon Scenario']=='FB all levers'].copy()
    bau = temp[temp['Vizcon Scenario'] == BASE_SCEN].copy()
    all_other = temp[~(temp['Vizcon Scenario'].isin(['FB without pricing', 'FB without pricing']))].copy()

    no_pricing = no_pricing.merge(all_levers[[
                                'timebounds', 'Pollutant Group',
                                'Actual Pollutant', 'Attributable Warming',
                                'Global Temperature Anomaly']],
                                  on=['timebounds', 'Pollutant Group', 'Actual Pollutant'], how='left',
                                  suffixes=('', '_all_levers'))

    no_pricing = no_pricing.merge(bau[[
                                'timebounds', 'Pollutant Group',
                                'Actual Pollutant', 'Attributable Warming',
                                'Global Temperature Anomaly']],
                                  on=['timebounds', 'Pollutant Group', 'Actual Pollutant'], how='left',
                                  suffixes=('', '_bau'))

    no_pricing['Attributable Warming'] = (no_pricing['Attributable Warming_all_levers']
                                          - no_pricing['Attributable Warming']
                                          + no_pricing['Attributable Warming_bau'])
    no_pricing['Global Temperature Anomaly'] = (no_pricing['Global Temperature Anomaly_all_levers']
                                                - no_pricing['Global Temperature Anomaly']
                                                + no_pricing['Global Temperature Anomaly_bau'])

    temp = pd.concat([all_other, no_pricing[all_other.columns]], ignore_index=True)

    return temp


def test_normalization(temp):
    """
    Verify that normalized avoided warming sums correctly
    """
    individual_levers_mask = (~temp['Vizcon Scenario'].isin(['FB all levers', BASE_SCEN, 'zero'])) & (
                temp['Actual Pollutant'] != 'All')
    normalized_totals = temp[individual_levers_mask].groupby(
        ['timebounds', 'Actual Pollutant', 'Pollutant Group']
    )['Normalized Avoided Warming'].sum().reset_index()

    all_levers_totals = temp[temp['Vizcon Scenario'] == 'FB all levers'][
        ['timebounds', 'Actual Pollutant', 'Pollutant Group', 'Avoided Warming']
    ]

    comparison = normalized_totals.merge(
        all_levers_totals,
        on=['timebounds', 'Actual Pollutant', 'Pollutant Group']
    )

    assert np.allclose(
        comparison['Normalized Avoided Warming'],
        comparison['Avoided Warming'],
        rtol=1e-10
    ), "Normalized avoided warming totals don't match 'FB all levers' totals by pollutant"


def normalize_lever_attribution(temp):
    """
    Add separate columns that preserve the share of the total attributable warming for each lever, but ensure
    that the totals match the 'all levers' scenario.
    """
    original_cols = temp.columns.tolist()

    # Calculate avoided warming vs baseline
    bau = temp[temp['Vizcon Scenario'] == BASE_SCEN].copy()
    bau = bau[['timebounds', 'Actual Pollutant', 'Pollutant Group', 'Attributable Warming']].rename(
        columns={'Attributable Warming': 'Attributable Warming_bau'})

    temp = temp.merge(bau, on=['timebounds', 'Actual Pollutant', 'Pollutant Group'], how='left')
    temp['Avoided Warming'] = temp['Attributable Warming'] - temp['Attributable Warming_bau']

    # Get all levers total for each pollutant
    all_levers = temp[temp['Vizcon Scenario'] == 'FB all levers'].copy()[
        ['timebounds', 'Actual Pollutant', 'Pollutant Group', 'Avoided Warming']
    ].rename(columns={'Avoided Warming': 'Avoided Warming_all_levers'})

    # Calculate shares for individual levers (excluding baseline and all levers)
    individual_levers = temp[
        (~temp['Vizcon Scenario'].isin(['FB all levers', BASE_SCEN, 'zero'])) &
        (temp['Actual Pollutant'] != 'All')
        ].copy()

    # Calculate total avoided warming across all individual levers for each pollutant
    lever_totals = individual_levers.groupby([
        'timebounds', 'Actual Pollutant', 'Pollutant Group'
    ])['Avoided Warming'].sum().reset_index().rename(
        columns={'Avoided Warming': 'Total_Individual_Levers'}
    )

    # Merge back to get shares
    temp = temp.merge(lever_totals, on=['timebounds', 'Actual Pollutant', 'Pollutant Group'], how='left')
    temp['Share of Avoided Warming'] = temp['Avoided Warming'] / temp['Total_Individual_Levers']

    # Set the share to 1 for 'FB all levers' scenario
    temp.loc[temp['Vizcon Scenario'] == 'FB all levers', 'Share of Avoided Warming'] = 1.0
    temp.loc[temp['Vizcon Scenario'] == 'zero', 'Share of Avoided Warming'] = np.nan

    # Merge all levers totals and calculate normalized values
    temp = temp.merge(all_levers, on=['timebounds', 'Actual Pollutant', 'Pollutant Group'], how='left')
    temp['Normalized Avoided Warming'] = temp['Share of Avoided Warming'] * temp['Avoided Warming_all_levers']

    temp['Normalized Attributable Warming'] = temp['Attributable Warming_bau'] + temp['Normalized Avoided Warming']

    test_normalization(temp)

    return temp[original_cols + ['Normalized Attributable Warming', 'Avoided Warming', 'Normalized Avoided Warming']]


def postprocess_levers(temp):
    """
    Recalculate the pricing scenario by taking the difference from 'All levers', then adding to BASE_SCEN.
    """
    if not POSTPROCESS_LEVERS:
        return temp
    print("Renormalizing lever attribution scenarios and adjusting taxation scenario...")
    temp = adjust_taxation_scenario(temp)

    # Normalize the lever attribution ------------------------------------------------------------------------------
    temp = normalize_lever_attribution(temp)


    return temp


def main():
    """Set up FaIR, run it, and save results."""

    st = time.time()
    # Read input data
    forcing = pd.read_csv(FORCING_IN)
    ems = pd.read_csv(EMS_IN)
    conc = pd.read_csv(CONC_IN)

    # Define scenarios
    scenarios = list(set(pd.concat([forcing['Scenario'], ems['Scenario'], conc['Scenario']]).unique()))

    # Replace '_upper' with ' upper' and '_lower' with ' lower'
    scenarios = [s.replace('_upper', ' upper').replace('_lower', ' lower') for s in scenarios]

    vizcon_scenarios =  np.unique([s.split('_')[-1] if '_' in s else s for s in scenarios])
    # Drop 'ssp119', 'tra', 'zero'
    vizcon_scenarios = [name for name in vizcon_scenarios if name not in ['ssp119', 'tra']]
    vizcon_scenarios = list(map(str, vizcon_scenarios))

    # Set up FaIR
    f = set_up_fair(scenarios)

    # Run the model
    print("Running FaIR...")
    f.run()

    # Retrieve the temperature results in a clean format
    temp = clean_temp_output(f, vizcon_scenarios)

    # Rename 'Pollutant' to 'Pollutant Group'
    temp = temp.rename(columns={'Pollutant': 'Pollutant Group'})
    temp = temp[['timebounds', 'scenario', 'Vizcon Scenario', 'Pollutant Group',
                 'Actual Pollutant', 'Attributable Warming', 'Global Temperature Anomaly']]
    if POSTPROCESS_LEVERS:
        # Save an intermediate output without lever renormalization
        temp.to_csv(INTERMEDIATE_OUT, index=False)
        print(f"Saved intermediate results to {INTERMEDIATE_OUT}")
    # # Remove any scenarios with 'lower' or 'upper' in the scenario name
    # temp = temp[~temp['scenario'].str.contains('lower', case=False, regex=True)]
    # temp = temp[~temp['scenario'].str.contains('upper', case=False, regex=True)]

    temp = postprocess_levers(temp)

    # Save the results
    temp.to_csv(TEMP_OUT, index=False)

    scenario_totals = temp[temp['Actual Pollutant'] == 'All'].copy()
    scenario_totals = scenario_totals[['timebounds', 'Vizcon Scenario', 'Attributable Warming']]
    # Pivot 'Vizcon Scenario' to columns
    scenario_totals = scenario_totals.pivot(index='timebounds', columns='Vizcon Scenario',
                                            values='Attributable Warming').reset_index()
    scenario_totals.to_csv(CLEAN_OUT, index=False)

    print(f"Ran in {(time.time() - st)/60:.2f} minutes")


if __name__ == "__main__":
        main()
