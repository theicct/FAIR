"""Convert ICCT SLCP emissions to FaIR format and combine with CMIP6 emissions data."""

import pandas as pd

# Set wide display
pd.set_option('display.width', 1000000)
pd.set_option('display.max_columns', 100000)

BASE_YR = 2023
END_YR = 2050
EMS_IN = 'preprocessing/final/SLCP_inventory_long.csv'
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
    ems = ems[ems['Scenario'].str.startswith('ssp119')]
    forc = forc[forc['Scenario'].str.startswith('ssp119')]
    conc = conc[conc['Scenario'].str.startswith('ssp119')]

    return ems, conc, forc


def adjust_contrail_forc(slcp_ems, forc, SSPs):
    ct_var = 'Effective Radiative Forcing|Anthropogenic|Other|Contrails and Contrail-induced Cirrus'

    # Use ICCT estimates for contrail BAU and striving under SSP119
    for SSP in SSPs:
        for yr in range(BASE_YR, END_YR+1):
            bau = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == 'contrails')]['BAU'].values[0]
            if STRIVING_SCEN is not None:
                striving = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == 'contrails')][STRIVING_SCEN].values[0]

            for scen in forc['Scenario'].unique():
                # Set default to BAU
                forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == scen), str(yr)] = bau

            # Adjust the BAU for contrail scenarios
            if STRIVING_SCEN is not None:
                # Set the striving scenario
                forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_striving'), str(yr)] = striving
                forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_contrails_Aviation_striving'), str(
                    yr)] = striving

            forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_zero_tra'), str(yr)] = 0

            # Add zero contrails scenario
            forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_contrails_Aviation_zero'), str(
                yr)] = 0

    return forc


def align_inputs_with_fair(ems, conc, forc, slcp_ems, species_to_rcmip, SSPs):

    # Copy the ssp119 scenario to ssp119_striving
    for SSP in SSPs:
        ems = pd.concat([ems, ems[ems['Scenario'] == SSP].replace(SSP, f'{SSP}_striving')])
        forc = pd.concat([forc, forc[forc['Scenario'] == SSP].replace(SSP, f'{SSP}_striving')])
        conc = pd.concat([conc, conc[conc['Scenario'] == SSP].replace(SSP, f'{SSP}_striving')])

        ems = pd.concat([ems, ems[ems['Scenario'] == SSP].replace(SSP, f'{SSP}_zero_tra')])
        forc = pd.concat([forc, forc[forc['Scenario'] == SSP].replace(SSP, f'{SSP}_zero_tra')])
        conc = pd.concat([conc, conc[conc['Scenario'] == SSP].replace(SSP, f'{SSP}_zero_tra')])

        for specie in slcp_ems['Species'].unique():
            for sector in slcp_ems['Sector'].unique():
                # for source in slcp_ems['Source'].unique():
                if slcp_ems[(slcp_ems['Species'] == specie) & (slcp_ems['Sector'] == sector)].empty:
                    continue

                ems = pd.concat([ems, ems[(ems['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_striving')])
                conc = pd.concat([conc, conc[(conc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_striving')])
                forc = pd.concat([forc, forc[(forc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_striving')])

                # Add zero scenarios
                ems = pd.concat([ems, ems[(ems['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_zero')])
                conc = pd.concat([conc, conc[(conc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_zero')])
                forc = pd.concat([forc, forc[(forc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_zero')])

    forc = adjust_contrail_forc(slcp_ems, forc, SSPs)

    # Use ICCT estimates for anthropogenic emissions as well
    for specie, specie_rcmip_name in species_to_rcmip.items():
    # for specie, specie_rcmip_name in dict({'N2O': 'N2O'}).items():
        for SSP in SSPs:
            if specie not in slcp_ems['Species'].values:
                continue
            print(specie)
            icct_units = slcp_ems[(slcp_ems['Species'] == specie)]['Units'].values[0]
            expected_units = ems.loc[(ems["Scenario"] == SSP) & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) & (
                        ems["Region"] == "World")]['Unit'].values[0]
            expected_units = str(expected_units).split(' ')[0]
            for yr in range(BASE_YR, END_YR+1):

                ems_yr = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == specie)].copy()

                assert expected_units == 'Mt' or expected_units == 'kt', f"Expected units are not Mt or kt, but {expected_units}"

                conversion_factors = {
                    ('Gt', 'Mt'): 1e3,
                    ('Gt', 'kt'): 1e6,
                    ('Mt', 'kt'): 1e3,
                    ('kt', 'Mt'): 1e-3
                }

                factor = conversion_factors.get((icct_units, expected_units))

                id_cols = ['Scenario', 'Year',  'Species', 'Units', 'Source', 'Sector']
                scenarios = [col for col in ems_yr.columns if col not in id_cols]

                if factor:
                    ems_yr[scenarios] *= factor

                for scen in scenarios:
                    ems_yr['Avoided'] = ems_yr[scen] - ems_yr['BAU']
                    scen_lower = scen.lower()

                    ems.loc[(ems["Scenario"] == f'{SSP}_{scen_lower}') & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                            (ems["Region"] == "World"), str(yr)] += (ems_yr['Avoided']).sum()
                ems.loc[(ems["Scenario"] == f'{SSP}_zero_tra') & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                        (ems["Region"] == "World"), str(yr)] += -1*ems_yr['BAU'].sum()

                for sector in slcp_ems['Sector'].unique():
                    # for source in slcp_ems['Source'].unique():
                    if slcp_ems[(slcp_ems['Species'] == specie) & (slcp_ems['Sector'] == sector)].empty:
                        continue
                    for scen in scenarios:
                        scen_lower = scen.lower()
                        scenario_name = f'{SSP}_{specie}_{sector}_{scen_lower}'
                        avoided = ems_yr[(ems_yr['Sector'] == sector)]['Avoided'].values.sum()
                        ems.loc[(ems["Scenario"] == scenario_name) & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                                (ems["Region"] == "World"), str(yr)] += avoided

                    # Add a zero_out_scenario
                    zero_scenario_name = f'{SSP}_{specie}_{sector}_zero'
                    sector_bau = ems_yr[(ems_yr['Sector'] == sector)]['BAU'].values.sum()
                    ems.loc[(ems["Scenario"] == zero_scenario_name) & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                            (ems["Region"] == "World"), str(yr)] += -1 * sector_bau

                    if (yr == 2050) & (specie=='Contrails'):
                        pass

    # drop all duplicates in case we ran a scenario twice
    ems = ems.drop_duplicates()
    forc = forc.drop_duplicates()
    conc = conc.drop_duplicates()

    return ems, conc, forc


def main():
    SSPs = ['ssp119']
    slcp_ems = pd.read_csv(EMS_IN)

    ems = pd.read_csv('CMIP6/rcmip-emissions-annual-means-v5-1-0_original.csv')
    conc = pd.read_csv('CMIP6/rcmip-concentrations-annual-means-v5-1-0_original.csv')
    forc = pd.read_csv('CMIP6/rcmip-radiative-forcing-annual-means-v5-1-0_original.csv')

    slcp_ems = clean_slcp_ems(slcp_ems)
    species_to_rcmip = define_species_mapping()

    # Clean input data
    ems, conc, forc = clean_inputs(ems, conc, forc, species_to_rcmip)

    # Align inputs with FaIR
    ems, conc, forc = align_inputs_with_fair(ems, conc, forc, slcp_ems, species_to_rcmip, SSPs)

    forc.to_csv('CMIP6/rcmip-radiative-forcing-annual-means-v5-1-0.csv', index=False)
    ems.to_csv('CMIP6/rcmip-emissions-annual-means-v5-1-0.csv', index=False)
    conc.to_csv('CMIP6/rcmip-concentrations-annual-means-v5-1-0.csv', index=False)

    # Melt year columns into a single column
    ems_long = ems.melt(id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Mip_Era', 'Activity_Id'], var_name='Year', value_name='ems')
    ems_long.to_csv('CMIP6/rcmip-emissions-annual-means-v5-1-0_long.csv', index=False)
    forc_long = forc.melt(id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Mip_Era', 'Activity_Id'], var_name='Year', value_name='ems')
    forc_long.to_csv('CMIP6/rcmip-forcing-annual-means-v5-1-0_long.csv', index=False)

if __name__ == "__main__":
    main()
