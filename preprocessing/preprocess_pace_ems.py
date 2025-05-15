"""
Prepare PACE emissions outputs for inputs into FaIR.
"""

import pandas as pd

# set wide display
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

# FNAME_IN = 'PACE/pace_vizcon_summary_apr23.csv'
FNAME_IN = 'PACE/summary_with_historical_data_May14.csv'

CONTRAILS_VAR_NAME = 'ConERF'
CONTRAILS_UNITS = 'W/m2'

SA_EARTH_m2 = 5.1e14  # m2
SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60  # seconds in a year


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

    return df_long


def build_scenarios(df):
    """
    Create an individual scenario for each pollutant, Sector, Source, and current 'Scenario' name.
    """
    # Also only filter to GHGs (co2, ch4, and n2o)
    df = df.loc[df['Species'].isin(['co2', 'ch4', 'n2o'])].reset_index(drop=True)

    for i, row in enumerate(df.iterrows()):
        row = row[1]
        # Create a new scenario name
        new_scenario = f"{row.Scenario}_{row.Sector}_{row.Source}_{row.Species}"
        df.loc[i, 'Scenario'] = new_scenario

    print(df['Scenario'].unique())
    return df


def run():
    """
    The base inventory file includes Marine and aviation currently.

    TODO: add on-road, off-road, and calculate WTT emissions for sectors missing them.
    """
    df = pd.read_csv(FNAME_IN)
    # df = df[df['CY']>=2023].copy()

    # Drop where 'Scenario' is nan
    df = df.dropna(subset=['Scenario'])
    df['Sector'] = 'Aviation'
    df['Source'] = 'WTW'

    # Replace 'Baseline' with 'BAU'
    df['Scenario'] = df['Scenario'].replace('Baseline', 'BAU')

    # Rename 'nvPM' to 'BC'
    df = df.rename(columns={'nvPM_mass': 'BC', CONTRAILS_VAR_NAME: 'contrails', 'CY': 'Year', 'SO2': 'SOx'})

    CLIMATE_FORCERS = ['CO2', 'H2O', 'SOx', 'NOx', 'CH4', 'N2O', 'BC', 'contrails']
    ID_COLS = ['Scenario', 'Sector', 'Year', 'Source']

    df_long = df[ID_COLS + CLIMATE_FORCERS].melt(
        id_vars=ID_COLS,
        var_name='Species',
        value_name='ems'
    )

    # Set Unit to GJ for EF and Mt for all others
    df_long['Units'] = 'Mt'
    df_long.loc[df_long['Species'] == 'contrails', 'Units'] = CONTRAILS_UNITS

    df_long = convert_units(df_long)

    df_long['Species'] = df_long['Species'].str.lower()

    df_long.to_csv('final/PACE_inventory_long.csv', index=False)


def main():
    run()


if __name__ == "__main__":
    main()
