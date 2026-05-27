import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# set wide display
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

# Define upstream emissions intensities

jetA_co2_wtt_ei = np.nan # gCO2/MJ
jetA_ch4_wtt_ei = 0.1 # g CH4/MJ
jetA_n2o_wtt_ei = 0.0001739 # gN2O/MJ
jetA_bc_wtt_ei = 0.0002235 # gBC per MJ


diesel_co2_wtt_ei = 12.12427654 # gCO2/MJ
diesel_ch4_wtt_ei = 0.103806325 # gCH4/MJ
diesel_n2o_wtt_ei = 0.000221792 # gN2O/MJ
diesel_bc_wtt_ei = 0.000158163 # gBC/MJ


# Define other constants
jetA_MJ_per_kg = 43.02 # MJ/kg
jetA_MJ_per_Mt = jetA_MJ_per_kg * 1e9 # convert from kg to Mt

diesel_MJ_per_kg = 45.5 # MJ/kg
diesel_MJ_per_Mt = diesel_MJ_per_kg * 1e9 # convert from kg to Mt
# diesel_MtCO2_per_MJ = 7.35e-8 #MtCO2/MJ

PACE_IN = "PACE_inventory_long.csv"

def calculate_aviation_wtt(df_long):
    """
    From the CO2 values (or fuel consumption), we can calculate the WTT emissions for aviation
    using the upstream emission factors from the GHS.
    """
    aviation = df_long.loc[(df_long['Sector'] == 'Aviation') & (df_long['Species'] == 'fuel')].copy()
    other_df = df_long.loc[~((df_long['Sector'] == 'Aviation') & (df_long['Species'] == 'fuel'))].copy()

    aviation = aviation.drop(columns='Species')
    aviation['Source'] = 'WTT'

    aviation['MJ'] = aviation.pop('ems') * jetA_MJ_per_Mt

    aviation['co2'] = aviation['MJ'] * jetA_co2_wtt_ei  # gCO2
    aviation['ch4'] = aviation['MJ'] * jetA_ch4_wtt_ei # gCH4
    aviation['n2o'] = aviation['MJ'] * jetA_n2o_wtt_ei # gN2O
    aviation['bc'] = aviation['MJ'] * jetA_bc_wtt_ei # gBC

    # Convert all pollutants to Mt except convert CO2 to Gt
    aviation['co2'] = aviation['co2'] * 1e-9
    aviation['ch4'] = aviation['ch4'] * 1e-6
    aviation['n2o'] = aviation['n2o'] * 1e-6
    aviation['bc'] = aviation['bc'] * 1e-6

    # Melt the dataframe back to long
    aviation = aviation.melt(
        id_vars=['Scenario', 'Sector', 'Source', 'Units', 'Year'],
        var_name='Species',
        value_name='ems'
    )

    # Rename units
    aviation.loc[aviation['Species'] == 'co2', 'Units'] = 'Gt'
    aviation.loc[aviation['Species'] != 'co2', 'Units'] = 'Mt'

    df_long = pd.concat([other_df, aviation])

    return df_long

def interp_off_road(off_road_ems):

    # Define the full range of years
    years = np.arange(1990, 2051).astype(str)  # Keep as strings to match column names

    # Ensure all expected years exist in the DataFrame
    existing_years = off_road_ems.columns[5:]  # Extract year columns
    missing_years = sorted(set(years) - set(existing_years))
    for year in missing_years:
        off_road_ems[year] = np.nan  # Add missing years with NaNs

    # Sort columns to maintain proper order
    year_columns = sorted(off_road_ems.columns[5:], key=int)
    off_road_ems = off_road_ems[['Scenario', 'Sector', 'Sub-Sector', 'Species', 'Units'] + year_columns]

    # Convert year columns to numeric (ensures correct dtype for interpolation)
    off_road_ems[year_columns] = off_road_ems[year_columns].apply(pd.to_numeric, errors='coerce')

    # Interpolate across years
    def interpolate_row(row):
        x = np.array(year_columns, dtype=int)  # Convert column names to integers
        y = row[year_columns].values.astype(float)  # Convert values to float
        mask = ~np.isnan(y)  # Find valid data points

        if mask.sum() > 1:  # Ensure at least two valid points for interpolation
            f = interp1d(x[mask], y[mask], kind='linear', fill_value='extrapolate')
            row[year_columns] = f(x)  # Apply interpolation

        return row

    # Apply interpolation row-wise
    off_road_ems = off_road_ems.apply(interpolate_row, axis=1)

    # Convert to long format
    off_road_ems = off_road_ems.melt(
        id_vars=['Scenario', 'Sector', 'Sub-Sector', 'Species', 'Units'],
        var_name='Year',
        value_name='ems'
    )

    # Set the 'Year' column to int
    off_road_ems['Year'] = off_road_ems['Year'].astype(int)

    return off_road_ems


def remove_domestic_aviation(off_road_ems, other_ems):
    """
    TRA_OT_OTHER includes domestic civil aviation. This function calculates domestic aviation emissions for a base
    year based on the aviation emissions in the BAU scenario, calculates the share of the emissions (for each
    pollutant) in the TRA_OT_OTHER subsector that comes from domestic aviation and removes this % for all time.
    """
    base_yr = 2022

    # Calculate the share of emissions in the TRA_OT_OTHER subsector that comes from domestic aviation
    off_road_bau = off_road_ems[(off_road_ems['Sub-Sector'] == 'TRA_OT_OTHER') & (off_road_ems['Year'] == base_yr)].copy()

    # Set the 'Year' column to int
    other_ems['Year'] = other_ems['Year'].astype(int)

    aviation_bau = other_ems[(other_ems['Sector'] == 'Aviation') & (other_ems['Scenario'] == 'BAU') & (other_ems['Year'] == base_yr)].copy()
    # Filter to relevant pollutants (co2, ch4, bc)
    aviation_bau = aviation_bau[aviation_bau['Species'].isin(['co2', 'ch4', 'bc'])].copy()
    # Assume only TTW emissions
    # aviation_bau = aviation_bau[aviation_bau['Source'] == 'TTW'].copy()
    # Calculate the domestic aviation emissions by multiplying by 40%
    aviation_bau['ems'] = aviation_bau['ems'] * 0.4

    # Convert to the same units as off_road_ems
    for spec in ['co2', 'bc']:
        off_road_unit = off_road_ems.loc[off_road_ems['Species'] == spec, 'Units'].unique()[0]
        aviation_unit = aviation_bau.loc[aviation_bau['Species'] == spec, 'Units'].unique()[0]
        if (off_road_unit == 'Mt') and (aviation_unit == 'Tg'):
            # Same unit, just change the name
            aviation_bau.loc[aviation_bau['Species'] == spec, 'Units'] = 'Mt'
        if (off_road_unit == 'kt') and (aviation_unit == 'Tg'):
            aviation_bau.loc[aviation_bau['Species'] == spec, 'ems'] *= 1e3  # Convert to kt
            aviation_bau.loc[aviation_bau['Species'] == spec, 'Units'] = 'kt'

    # Calculate the share of emissions in the TRA_OT_OTHER subsector that comes from domestic aviation
    off_road_bau = off_road_bau.merge(aviation_bau[['Species', 'Units', 'Year', 'ems']], on=['Species', 'Units', 'Year'],
                       how='left', suffixes=('', '_domestic_aviation')).fillna(0)
    off_road_bau['domestic_aviation_share'] = off_road_bau['ems_domestic_aviation'] / off_road_bau['ems']

    print(off_road_bau)
    off_road_bau = off_road_bau[['Sub-Sector', 'Species', 'domestic_aviation_share']]

    off_road_ems = off_road_ems.merge(off_road_bau, on=['Sub-Sector', 'Species'], how='left').fillna(0)
    # Remove the domestic aviation share from the TRA_OT_OTHER subsector
    off_road_ems['ems'] *= 1 - off_road_ems.pop('domestic_aviation_share')

    return off_road_ems

def interp_reductions(reductions):
    # Define full year range
    full_years = np.arange(1990, 2051)

    # Function to interpolate reductions
    def interpolate_group(g):
        # Store group metadata
        metadata = g[['Sub-Sector', 'Species']].iloc[0]

        # Interpolation
        f = interp1d(g['Year'], g['Reduction'], kind='linear', fill_value='extrapolate')
        interpolated = pd.DataFrame({'Year': full_years, 'Reduction': f(full_years)})

        # Add metadata back
        for col, value in metadata.items():
            interpolated[col] = value

        return interpolated

    # Apply interpolation, keeping grouping columns
    reductions = (
        reductions.groupby(['Sub-Sector', 'Species'], group_keys=False)
        .apply(interpolate_group)
        .reset_index(drop=True)
    )

    return reductions


def calc_off_road(off_road_ems, other_ems):
    """
    Use the BAU scenario for off-road to calculate the emissions for the Striving scenario using the
    following percentage reductions:

    For TRA_OT_OTHER:
        CO2: 2035: 20%, 2050: 50%
        BC: 2035: 20%, 2050: 50%

    For all other sub-sectors:
        CO2: 2035: 40%, 2050: 100%
        BC: 2035: 60%, 2050: 100%
    """
    # Drop the shipping sector ('TRA_OT_SHIP_NAT'), this is covered by the marine sector
    off_road_ems = off_road_ems[(off_road_ems['Sub-Sector'] != 'TRA_OT_SHIP_NAT')].copy()

    # Convert to long format and interpolate intermediate years
    off_road_ems = interp_off_road(off_road_ems)

    off_road_ems = remove_domestic_aviation(off_road_ems, other_ems)
    other_ems = other_ems[other_ems['Sector'] != 'off-road'].copy()

    off_road_ems.to_csv('diagnostic/off-road_no_SHP_AVI.csv', index=False)

    striving = off_road_ems.copy()
    striving['Scenario'] = 'Striving'

    reductions = pd.read_csv('off-road/Striving_reductions.csv')

    # interpolate these values to get the reductions for each year
    reductions = interp_reductions(reductions)

    # Calculate the percentage reductions
    # Ensure 'Year' is int for both dataframes
    reductions['Year'] = reductions['Year'].astype(int)
    striving['Year'] = striving['Year'].astype(int)
    striving = striving.merge(reductions, on=['Year', 'Sub-Sector', 'Species'], how='left')

    # Assert no NaN reductions
    assert not striving['Reduction'].isna().any(), "NaN values found in reductions"

    striving['ems'] = striving['ems'] * (1 - striving.pop('Reduction'))

    off_road_ems = pd.concat([off_road_ems, striving], ignore_index=True)

    off_road_ems['Source'] = 'TTW'

    # Export a version of just construction vehicles
    off_road_export = calculate_off_road_diesel_wtt(off_road_ems)
    # off_road_export = convert_units(off_road_export)
    # off_road_export['Sector'] = off_road_export.pop('Sub-Sector')
    off_road_export.to_csv('off-road/off-road_sub_sectors.csv', index=False)

    # Now that we've removed shipping and aviation from off-road, we can aggregate
    off_road_ems = off_road_ems.groupby(['Scenario', 'Sector', 'Source', 'Species', 'Units', 'Year']).sum(numeric_only=True).reset_index()

    # Filter to 2020 and later
    off_road_ems = off_road_ems[off_road_ems['Year'] >= 2020].copy()

    off_road_ems = pd.concat([off_road_ems, other_ems], ignore_index=True)

    return off_road_ems


def calculate_on_road_diesel_wtt(df_long):
    """
    From the CO2 values (or fuel consumption), we can calculate the WTT emissions for aviation
    using the upstream emission factors from the GHS.
    """
    energy = pd.read_csv('on-road/on_road_PJ_2deg_gap.csv')

    # Melt the dataframe back to long
    energy = energy.melt(
        id_vars=['Scenario', 'Year'],
        var_name='Fuel',
        value_name='PJ'
    )

    # Only use energy for diesel
    # energy = energy.loc[energy['Fuel'] == 'diesel'].copy()

    energy = energy.groupby(['Scenario', 'Year']).sum(numeric_only=True).reset_index() # PJ

    energy['MJ'] = energy.pop('PJ') * 1e9 # MJ
    print(energy)

    # Calculate the WTT emissions assuming everything is diesel
    # energy['co2'] = energy['MJ'] * diesel_co2_wtt_ei * 1e-15 # TgCO2
    energy['ch4'] = energy['MJ'] * diesel_ch4_wtt_ei * 1e-9 # ktCH4
    energy['n2o'] = energy['MJ'] * diesel_n2o_wtt_ei * 1e-9 # ktN2O
    energy['bc'] = energy.pop('MJ') * diesel_bc_wtt_ei * 1e-9 # ktBC

    # Melt the dataframe back to long
    energy = energy.melt(
        id_vars=['Scenario', 'Year'],
        var_name='Species',
        value_name='ems'
    )

    energy['Sector'] = 'on-road'
    energy['Source'] = 'WTT'
    energy['Units'] = 'kt'

    # Ensure 'Year' is type int for both dataframes
    df_long['Year'] = df_long['Year'].astype(int)
    energy['Year'] = energy['Year'].astype(int)

    energy = energy[df_long.columns]

    df_long = pd.concat([df_long, energy]).reset_index(drop=True)

    # assert no duplicate ID columns (Scenario, Sector, Source, Species, Year)
    assert df_long.drop_duplicates(subset=['Scenario', 'Sector', 'Source', 'Species', 'Year']).shape == df_long.shape

    return df_long


def calculate_off_road_diesel_wtt(df_long):
    """
    From the CO2 values (or fuel consumption), we can calculate the WTT emissions
    using the upstream emission factors from the GHS.
    """
    non_road = df_long.loc[(df_long['Sector'].isin(['off-road'])) & (df_long['Species'] == 'co2')].copy()

    non_road = non_road.drop(columns='Species')
    non_road['Source'] = 'WTT'

    non_road['MJ'] = non_road.pop('ems') * diesel_MJ_per_Mt

    # Convert all pollutants to Mt except convert CO2 to Gt
    non_road['co2'] = non_road['MJ'] * diesel_co2_wtt_ei * 1e-12 # Mt CO2
    non_road['ch4'] = non_road['MJ'] * diesel_ch4_wtt_ei * 1e-9 # kt CH4
    non_road['n2o'] = non_road['MJ'] * diesel_n2o_wtt_ei * 1e-9 # kt N2O
    non_road['bc'] = non_road.pop('MJ') * diesel_bc_wtt_ei * 1e-9 # kt BC

    # Melt the dataframe back to long
    if 'Sub-Sector' in non_road.columns:
        id_cols = ['Scenario', 'Sector', 'Sub-Sector', 'Source', 'Units', 'Year']
    else:
        id_cols = ['Scenario', 'Sector', 'Source', 'Units', 'Year']
    non_road = non_road.melt(
        id_vars=id_cols,
        var_name='Species',
        value_name='ems'
    )

    # Rename units
    non_road.loc[non_road['Species'] == 'co2', 'Units'] = 'Mt'
    non_road.loc[non_road['Species'] != 'co2', 'Units'] = 'kt'

    df_long = pd.concat([df_long, non_road]).reset_index(drop=True)

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
    # assert df_long.loc[df_long['Species'] == 'co2', 'Units'].unique() == ['Mt']
    # Where Species is 'co2' and Units are 'Mt', convert to Gt
    df_long.loc[(df_long['Species'] == 'co2') & (df_long['Units'] == 'Mt'), 'ems'] = df_long['ems'] * 0.001
    df_long.loc[df_long['Species'] == 'co2', 'Units'] = 'Gt'

    # Contrails are in units of mW/m2, convert to W/m2
    ct_unit = list(df_long.loc[df_long['Species'] == 'contrails', 'Units'].unique())
    if ('contrails' in df_long['Species'].unique()) & (ct_unit == ['mW/m2']):
        assert df_long.loc[df_long['Species'] == 'contrails', 'Units'].unique() == ['mW/m2']
        df_long.loc[df_long['Species'] == 'contrails', 'ems'] = df_long['ems'] * 0.001
        df_long.loc[df_long['Species'] == 'contrails', 'Units'] = 'W/m2'

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


def add_pace_to_totals(df, pace):
    """
    Add PACE emissions to the total inventory emissions.
    """
    # Filter to Historical Trends and Full Breakthrough scenarios
    pace = pace[pace['Scenario'].isin(['BAU', 'Full Breakthrough'])].copy()
    # Rename to "BAU" and "Striving"
    pace['Scenario'] = pace['Scenario'].replace({'BAU': 'BAU', 'Full Breakthrough': 'Striving'})
    # Ensure Filter to Year 2020 and later
    pace = pace[pace['Year'] >= 2020].copy()

    # Remove existing Aviation from df
    df = df[~((df['Sector'] == 'Aviation'))].copy()
    df = pd.concat([df, pace], ignore_index=True)

    return df


def run():
    """
    The base inventory file includes Marine and aviation currently.

    TODO: add on-road, off-road, and calculate WTT emissions for sectors missing them.
    """
    df = pd.read_csv('SLCP_inventory.csv')
    pace = pd.read_csv(PACE_IN)

    # Drop where 'Scenario' is nan
    df = df.dropna(subset=['Scenario'])

    df_long = df.melt(
        id_vars=['Scenario', 'Sector', 'Source', 'Species', 'Units'],
        var_name='Year',
        value_name='ems'
    )
    df_long = add_pace_to_totals(df_long, pace)

    off_road = pd.read_csv('off-road/off-road_BAU_IIASA.csv')
    df_long = calc_off_road(off_road, df_long)

    # Calculate WTT emissions for non-road
    df_long = calculate_on_road_diesel_wtt(df_long)

    df_long = calculate_off_road_diesel_wtt(df_long)

    df_long = convert_units(df_long)

    # Calculate WTT emissions for aviation
    # df_long = calculate_aviation_wtt(df_long)

    df_long.to_csv('final/SLCP_inventory_long.csv', index=False)

    # df_long = build_scenarios(df_long)
    # df_long.to_csv('diagnostic/GHG_SLCP_inventory_long.csv', index=False)

    # Produce a test dataset that aggregates across sectors for FaIR
    aviation = df_long[(df_long['Sector'] == 'Aviation')]
    aviation['ems'] = aviation['ems'] * 0.8

    print(df_long)

    # Check for duplicat ID columns
    print(df_long.drop_duplicates(subset=['Scenario', 'Sector', 'Source', 'Species', 'Year']).shape)

    # Pivot the year back wide
    df_wide = df_long.pivot_table(
        index=['Scenario', 'Sector', 'Source', 'Species', 'Units'],
        columns='Year',
        values='ems'
    ).reset_index()

    # df_wide = df_wide.drop(columns=['Sector', 'Source'])

    print(df_long['ems'].unique())
    print(df_wide)

    df_wide.to_csv('diagnostic/Test_full.csv', index=False)


def main():
    run()


if __name__ == "__main__":
    main()
