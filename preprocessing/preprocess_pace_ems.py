"""
Convert PACE emissions outputs into inputs for FaIR.
"""

import pandas as pd
import numpy as np

# set wide display
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

FNAME_IN = 'PACE/summary_sept16_historical_and_slcp_uq.csv'
EMS_OUT = 'final/PACE_inventory_long.csv' # If SENSITIVITY_FNAME is not None, this will be replaced to include "_sens"

# Optional: If a sensitivity runs file is provided, then only the baseline scenario will be included and
# all sensitivity runs will be added from the provided file.
SENSITIVITY_FNAME = 'PACE/pace_baseline_erf_projection_samples.csv'

# Use a separate filename to indicate that the output includes sensitivity runs, which changes the handling in later scripts
if SENSITIVITY_FNAME is not None:
    EMS_OUT = 'final/PACE_inventory_long_sens.csv'
    BASE_YR_SENS = 2023
    END_YR_SENS = 2050

CONTRAILS_VAR_NAME = 'ConERF'
CONTRAILS_UNITS = 'W/m2'
BASE_SCEN = 'Historical Trends'

SA_EARTH_m2 = 5.1e14  # m2
SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60  # seconds in a year

def clean_pace_output(df):
    """
    Assigns metadata and reshapes PACE emissions output for FaIR input.
    """
    # Assigns sector and scope metadata
    df['Sector'] = 'Aviation'
    df['Source'] = 'WTW'

    # Replace baseline scenario with 'BAU' for standardized handling in later scripts
    df['Scenario'] = df['Scenario'].replace(BASE_SCEN, 'BAU')

    # Rename 'nvPM' to 'BC'
    df = df.rename(columns={'nvPM_mass': 'BC', CONTRAILS_VAR_NAME: 'contrails', 'CY': 'Year', 'SO2': 'SOx'})

    CLIMATE_FORCERS = ['CO2', 'H2OERF', 'SOx', 'NOxERF', 'CH4', 'N2O', 'BC', 'contrails']
    ID_COLS = ['Scenario', 'Sector', 'Year', 'Source']

    df_long = df[ID_COLS + CLIMATE_FORCERS].melt(
        id_vars=ID_COLS,
        var_name='Species',
        value_name='ems'
    )

    # Set Unit to GJ for EF and Mt for all others based on PACE output units
    df_long['Units'] = 'Mt'
    df_long.loc[df_long['Species'] == 'contrails', 'Units'] = CONTRAILS_UNITS

    return df_long


def convert_units(df_long):
    """
    Fair expects the following units for each species

    {'BC': 'Mt BC/yr', 'C2F6': 'kt C2F6/yr', 'C3F8': 'kt C3F8/yr', 'C4F10': 'kt C4F10/yr', 'C5F12': 'kt C5F12/yr',
     'C6F14': 'kt C6F14/yr', 'C7F16': 'kt C7F16/yr', 'C8F18': 'kt C8F18/yr', 'CCl4': 'kt CCl4/yr', 'CF4': 'kt CF4/yr',
      'CFC-11': 'kt CFC11/yr', 'CFC-113': 'kt CFC113/yr', 'CFC-114': 'kt CFC114/yr', 'CFC-115': 'kt CFC115/yr',
       'CFC-12': 'kt CFC12/yr', 'CH2Cl2': 'kt CH2Cl2/yr', 'CH3Br': 'kt CH3Br/yr', 'CH3CCl3': 'kt CH3CCl3/yr',
        'CH3Cl': 'kt CH3Cl/yr', 'CH4': 'Mt CH4/yr', 'CHCl3': 'kt CHCl3/yr', 'CO': 'Mt CO/yr', 'CO2': 'Gt CO2/yr',
         'CO2 AFOLU': 'Gt CO2/yr', 'CO2 FFI': 'Gt CO2/yr', 'HCFC-141b': 'kt HCFC141b/yr', 'HCFC-142b': 'kt HCFC142b/yr',
          'HCFC-22': 'kt HCFC22/yr', 'HFC-125': 'kt HFC125/yr', 'HFC-134a': 'kt HFC134a/yr', 'HFC-143a': 'kt HFC143a/yr',
           'HFC-152a': 'kt HFC152a/yr', 'HFC-227ea': 'kt HFC227ea/yr', 'HFC-23': 'kt HFC23/yr', 'HFC-236fa': 'kt HFC236fa/yr',
           'HFC-245fa': 'kt HFC245fa/yr', 'HFC-32': 'kt HFC32/yr', 'HFC-365mfc': 'kt HFC365mfc/yr', 'HFC-4310mee': 'kt HFC4310mee/yr',
            'Halon-1202': 'kt Halon1202/yr', 'Halon-1211': 'kt Halon1211/yr', 'Halon-1301': 'kt Halon1301/yr',
             'Halon-2402': 'kt Halon2402/yr', 'N2O': 'Mt N2O/yr', 'NF3': 'kt NF3/yr', 'NH3': 'Mt NH3/yr',
              'NOx': 'Mt NO2/yr', 'NOx aviation': 'Mt NO2/yr', 'OC': 'Mt OC/yr', 'SF6': 'kt SF6/yr',
               'SO2F2': 'kt SO2F2/yr', 'Sulfur': 'Mt SO2/yr', 'VOC': 'Mt VOC/yr', 'c-C4F8': 'kt cC4F8/yr'}

    Of relevance:
        'CO2 FFI': 'Gt CO2/yr'
        'BC': 'Mt BC/yr'
        'CH4': 'Mt CH4/yr'
        'NOx aviation': 'Mt NO2/yr'
        'Sulfur': 'Mt SO2/yr'
        'N2O': 'Mt N2O/yr'

    For contrails, we need W/m2
    """
    # Convert where units are 't' (tonnes) to Tg (teragrams)
    df_long.loc[df_long['Units'] == 't', 'ems'] = df_long['ems'] / 1e6
    df_long.loc[df_long['Units'] == 't', 'Units'] = 'Mt'

    # Relabel Tg as Mt, since they are equivalent
    df_long.loc[df_long['Units'] == 'Tg', 'Units'] = 'Mt'

    df_long.loc[df_long['Units'] == 'kt', 'ems'] = df_long['ems'] * 0.001
    df_long.loc[df_long['Units'] == 'kt', 'Units'] = 'Mt'

    # For CO2, convert from Mt to Gt
    # Assert CO2 units are Mt
    assert df_long.loc[df_long['Species'] == 'CO2', 'Units'].unique() == ['Mt']
    df_long.loc[df_long['Species'] == 'CO2', 'ems'] = df_long['ems'] * 0.001
    df_long.loc[df_long['Species'] == 'CO2', 'Units'] = 'Gt'

    if 'contrails' in df_long['Species'].unique():
        if CONTRAILS_UNITS == 'GJ':
            assert df_long.loc[df_long['Species'] == 'contrails', 'Units'].unique() == ['GJ']
            df_long.loc[df_long['Species'] == 'contrails', 'ems'] = df_long['ems'] * 1e9
            df_long.loc[df_long['Species'] == 'contrails', 'Units'] = 'J'

            df_long.loc[df_long['Species'] == 'contrails', 'ems'] = 0.42 * df_long['ems'] / (SA_EARTH_m2 * SECONDS_IN_YEAR)
            df_long.loc[df_long['Species'] == 'contrails', 'Units'] = 'W/m2'

        elif CONTRAILS_UNITS == 'mW/m2':
            df_long.loc[df_long['Species'] == 'contrails', 'ems'] = df_long['ems'] / 1000
            df_long.loc[df_long['Species'] == 'contrails', 'Units'] = 'W/m2'

    # Convert Stratospheric water vapor and stratospheric NOx emission into radiative forcing
    if 'H2O' in df_long['Species'].unique():
        # 0.0052 ± 0.0026 mW m−2 (Tg (H2O) yr−1)−1
        df_long.loc[df_long['Species'] == 'H2O', 'ems'] *= 0.0052 / 1000
        df_long.loc[df_long['Species'] == 'H2O', 'Units'] = 'W/m2'

    if 'NOx' in df_long['Species'].unique():
        # Two conversions are needed:
        # 1. Convert from NOx to N using molecular weight conversion (*14/46)
        # 2. Convert from mass emissions to radiative forcing (3.6 mW m−2 (Tg (N) yr−1)−1)
        df_long.loc[df_long['Species'] == 'NOx', 'ems'] *= 14/46 * 3.6 / 1000
        df_long.loc[df_long['Species'] == 'NOx', 'Units'] = 'W/m2'

    # Rename 'NOXERF' to 'NOx' and 'H2OERF' to 'H2O'
    df_long.loc[df_long['Species'] == 'NOxERF', 'Species'] = 'NOx'
    df_long.loc[df_long['Species'] == 'H2OERF', 'Species'] = 'H2O'

    return df_long


def clean_sensitivity_runs(df):
    """
    Sensitivity output were provided in the wide format with years as rows and scenarios as columns.

    Though the year rows were not labeled, they covered 2023 to 2050.
    """
    assert all(['Sample' in col for col in df.columns])

    # Add a 'Year' column 2023-2050
    years = np.arange(BASE_YR_SENS, END_YR_SENS+1)

    assert len(years) == df.shape[0]

    df['Year'] = years

    # Melt and add scenario column
    df = df.melt(id_vars=['Year'], var_name='Scenario', value_name='ems')

    # Add metadata
    df['Sector'] = 'Aviation'
    df['Species'] = 'contrails'
    df['Units'] = 'mW/m2'
    df['Source'] = 'WTW'

    # Convert units to W/m2
    df['ems'] = df['ems'] * 1e-3
    df['Units'] = 'W/m2'

    return df


def main():
    """
    Execute the script.
    """
    df = pd.read_csv(FNAME_IN)

    df_long = clean_pace_output(df)

    df_long = convert_units(df_long)

    # Standardize to lowercase species names
    df_long['Species'] = df_long['Species'].str.lower()
    # Remove all '_' from Scenario names since underscores are a key delimiter in later scripts
    df_long['Scenario'] = df_long['Scenario'].str.replace('_', ' ', regex=False)

    if SENSITIVITY_FNAME is not None:
        # Add the BAU scenario to the mc_simulations
        bau_contrails = df_long[(df_long['Species'] == 'contrails') & (df_long['Scenario'] == 'BAU')].copy()
        # Add in contrails from the Monte Carlo simulations
        sens_simulations = pd.read_csv(SENSITIVITY_FNAME)
        sens_simulations = clean_sensitivity_runs(sens_simulations)

        # Concatenate the two dataframes
        df_long = pd.concat([bau_contrails, sens_simulations], ignore_index=True)

    df_long.to_csv(EMS_OUT, index=False)


if __name__ == "__main__":
    main()
