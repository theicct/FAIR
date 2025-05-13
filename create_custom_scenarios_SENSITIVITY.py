"""Convert ICCT SLCP emissions to FaIR format and combine with CMIP6 emissions data."""

import pandas as pd
import time

# Set wide display
pd.set_option('display.width', 10000000000)
pd.set_option('display.max_columns', 30)

BASE_YR = 2023
END_YR = 2050
EMS_IN = 'preprocessing/final/sens_inventory_long.csv'
STRIVING_SCEN = 'Striving'


def clean_slcp_ems(slcp_ems):
    # Convert all 'Species' names to lowercase
    slcp_ems['Species'] = slcp_ems['Species'].str.lower()

    # Pivot 'Scenario' column wide
    slcp_ems = slcp_ems.pivot_table(index=['Year', 'Species', 'Units', 'Source', 'Sector'], columns='Scenario', values='ems').reset_index()
    # Filter out slcp_ems water vapour
    slcp_ems = slcp_ems[slcp_ems['Species'] != 'h2o']

    # Rename 'ems' species columns to match expected names
    slcp_ems_conversion = {'bc':'BC', 'ch4':'CH4', 'co2':'CO2 FFI', 'h2o':'Stratospheric water vapour',
                               'n2o':'N2O', 'nox':'NOx', 'sox':'Sulfur', 'so2':'Sulfur'}
    slcp_ems = slcp_ems.replace(slcp_ems_conversion)


    return slcp_ems


def define_species_mapping():
    species = ['CO2 FFI', 'CO2 AFOLU', 'CO2', 'CH4', 'N2O', 'Sulfur', 'BC', 'OC', 'NH3', 'NOx', 'VOC', 'CO', 'CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115', 'HCFC-22', 'HCFC-141b', 'HCFC-142b', 'CCl4', 'CHCl3', 'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br', 'Halon-1202', 'Halon-1211', 'Halon-1301', 'Halon-2402', 'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12', 'C6F14', 'C7F16', 'C8F18', 'NF3', 'SF6', 'SO2F2', 'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa', 'HFC-32', 'HFC-365mfc', 'HFC-4310mee', 'NOx aviation', 'Solar', 'Volcanic', 'Aerosol-radiation interactions', 'Aerosol-cloud interactions', 'Ozone', 'Contrails', 'Light absorbing particles on snow and ice', 'Stratospheric water vapour', 'Land use', 'Equivalent effective stratospheric chlorine']

    species_to_rcmip = {specie: specie.replace("-", "") for specie in species}
    species_to_rcmip["CO2 FFI"] = "CO2|MAGICC Fossil and Industrial"
    species_to_rcmip["CO2 AFOLU"] = "CO2|MAGICC AFOLU"
    species_to_rcmip["NOx aviation"] = "NOx|MAGICC Fossil and Industrial|Aircraft"
    species_to_rcmip["Aerosol-radiation interactions"] = (
        "Aerosols-radiation interactions"
    )
    species_to_rcmip["Aerosol-cloud interactions"] = "Aerosols-radiation interactions"
    species_to_rcmip["Contrails"] = "Contrails and Contrail-induced Cirrus"
    species_to_rcmip["Light absorbing particles on snow and ice"] = "BC on Snow"
    species_to_rcmip["Stratospheric water vapour"] = "CH4 Oxidation Stratospheric H2O"
    species_to_rcmip["Land use"] = "Albedo Change"

    return species_to_rcmip


def clean_inputs(ems, conc, forc, species_to_rcmip):
    # Filter the inputs to only the full species values
    ems = ems[ems['Variable'].str.endswith(tuple(species_to_rcmip.values()))]
    conc = conc[conc['Variable'].str.endswith(tuple(species_to_rcmip.values()))]
    forc = forc[forc['Variable'].str.endswith(tuple(species_to_rcmip.values()))]

    properties = pd.read_csv('/Users/j.benoit/Documents/GitHub/FAIR/examples/properties/properties.csv')
    # Only keep the species for which 'input_mode' is 'emissions'
    for specie in properties['Variable']:
        if properties.loc[properties['Variable'] == specie, 'input_mode'].values[0] != 'emissions':
            ems = ems[~ems['Variable'].str.endswith(specie)]
        if properties.loc[properties['Variable'] == specie, 'input_mode'].values[0] != 'concentration':
            conc = conc[~conc['Variable'].str.endswith(specie)]
        if properties.loc[properties['Variable'] == specie, 'input_mode'].values[0] != 'forcing':
            forc = forc[~forc['Variable'].str.endswith(specie)]

    def linearly_interp(ems):
        # Identify the year columns dynamically
        year_cols = ems.columns[ems.columns.str.match(r'^\d{4}$')]

        # Perform linear interpolation across the year columns
        ems[year_cols] = ems[year_cols].interpolate(axis=1)

        # Optionally, fill any remaining NaNs at the beginning or end
        ems[year_cols] = ems[year_cols].bfill(axis=1).ffill(axis=1)

        return ems

    # Linearly interpolate missing values
    ems = linearly_interp(ems)
    forc = linearly_interp(forc)
    conc = linearly_interp(conc)

    # Filter to only Region = 'World' for ecah species type
    ems = ems[ems['Region'] == 'World']
    forc = forc[forc['Region'] == 'World']
    conc = conc[conc['Region'] == 'World']

    # Remove all scenarios that don't start with ssp119
    ems = ems[ems['Scenario']=='ssp119']
    forc = forc[forc['Scenario']=='ssp119']
    conc = conc[conc['Scenario']=='ssp119']

    return ems, conc, forc


def adjust_contrail_forc(slcp_ems, forc, SSPs, scenarios):
    ct_var = 'Effective Radiative Forcing|Anthropogenic|Other|Contrails and Contrail-induced Cirrus'

    # Use ICCT estimates for contrail BAU and striving under SSP119
    for scen in scenarios:
        for yr in range(BASE_YR, END_YR+1):
            striving = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == 'contrails')][scen].values[0]

            # Adjust the BAU for contrail scenarios
            if STRIVING_SCEN is not None:
                # Set the striving scenario
                forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == scen), str(yr)] = striving

    return forc


def align_inputs_with_fair(ems, conc, forc, slcp_ems, species_to_rcmip, SSPs):

    # Copy the ssp119 scenario to ssp119_striving

    id_cols = ['Year',
       'Species',
       'Units',
       'Source',
       'Sector']
    scenarios = [col for col in slcp_ems.columns if col not in id_cols]


    SSP = 'ssp119'

    ems_list = [ems]
    conc_list = [conc]
    forc_list = [forc]

    for scen in scenarios:
        ems_scen = ems.loc[ems['Scenario'] == SSP].copy()
        ems_scen['Scenario'] = scen
        ems_list.append(ems_scen)

        conc_scen = conc.loc[conc['Scenario'] == SSP].copy()
        conc_scen['Scenario'] = scen
        conc_list.append(conc_scen)

        forc_scen = forc.loc[forc['Scenario'] == SSP].copy()
        forc_scen['Scenario'] = scen
        forc_list.append(forc_scen)

    ems = pd.concat(ems_list, ignore_index=True)
    conc = pd.concat(conc_list, ignore_index=True)
    forc = pd.concat(forc_list, ignore_index=True)

    forc = adjust_contrail_forc(slcp_ems, forc, SSPs, scenarios)

    # drop all duplicates in case we ran a scenario twice
    ems = ems.drop_duplicates()
    forc = forc.drop_duplicates()
    conc = conc.drop_duplicates()

    return ems, conc, forc


def main():
    st = time.time()

    SSPs = ['ssp119']
    slcp_ems = pd.read_csv(EMS_IN)

    ems_orig = pd.read_csv('CMIP6/rcmip-emissions-annual-means-v5-1-0_original.csv')
    conc_orig = pd.read_csv('CMIP6/rcmip-concentrations-annual-means-v5-1-0_original.csv')
    forc_orig = pd.read_csv('CMIP6/rcmip-radiative-forcing-annual-means-v5-1-0_original.csv')


    slcp_ems = clean_slcp_ems(slcp_ems)
    species_to_rcmip = define_species_mapping()

    # Clean input data
    ems_orig, conc_orig, forc_orig = clean_inputs(ems_orig, conc_orig, forc_orig, species_to_rcmip)

    id_cols = ['Year', 'Species', 'Units', 'Source', 'Sector']
    scenarios = [scen for scen in slcp_ems.columns if scen not in id_cols]

    batch_size = 100
    # Ensure 'Results/sens/{batch_size}' exists
    import os
    if not os.path.exists(f'CMIP6/batches/{batch_size}'):
        os.makedirs(f'CMIP6/batches/{batch_size}')
    num_scenarios = len(scenarios)
    num_batches = round(num_scenarios/batch_size)
    # Run in batches
    for i in range(0, num_scenarios, batch_size):
        st = time.time()

        scenario_batch = scenarios[i:i + batch_size]

        # Filter to batch
        slcp_ems_batch = slcp_ems[id_cols + scenario_batch].copy()

        # Align inputs with FaIR
        ems, conc, forc = align_inputs_with_fair(ems_orig.copy(), conc_orig.copy(), forc_orig.copy(), slcp_ems_batch, species_to_rcmip, SSPs)

        forc.to_csv(f'inputs/final/batches/{batch_size}/rcmip-radiative-forcing-annual-means-v5-1-0_{round(i/batch_size)}.csv', index=False)
        ems.to_csv(f'inputs/final/batches/{batch_size}/rcmip-emissions-annual-means-v5-1-0_{round(i/batch_size)}.csv', index=False)
        conc.to_csv(f'inputs/final/batches/{batch_size}/rcmip-concentrations-annual-means-v5-1-0_{round(i/batch_size)}.csv', index=False)

        print(f'Ran batch {i/batch_size} of {num_batches} ({i/batch_size/num_batches*100:.2f}%) in {time.time() - st:.2f} seconds')


    # Melt year columns into a single column
    # ems_long = ems.melt(id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Mip_Era', 'Activity_Id'], var_name='Year', value_name='ems')
    # ems_long.to_csv('inputs/final/rcmip-emissions-annual-means-v5-1-0_long.csv', index=False)
    # forc_long = forc.melt(id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Mip_Era', 'Activity_Id'], var_name='Year', value_name='ems')
    # forc_long.to_csv('inputs/final/rcmip-forcing-annual-means-v5-1-0_long.csv', index=False)

    print("Time taken to run the script: ", time.time() - st)

if __name__ == "__main__":
    main()
