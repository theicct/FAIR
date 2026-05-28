"""Convert ICCT SLCP emissions to FaIR format and combine with CMIP6 emissions data."""

import os
import time
import pandas as pd

# Set wide display
pd.set_option('display.width', 1000000)
pd.set_option('display.max_columns', 100000)

# User inputs — set in examples/run_all.py when running the full pipeline, or override here for standalone use
BASE_YR       = int(os.environ.get('FAIR_BASE_YR',       1940))
END_YR        = int(os.environ.get('FAIR_END_YR',        2050))
BASELINE_SCEN = os.environ.get(    'FAIR_BASELINE_SCEN', 'BAU')
BATCH_SIZE    = int(os.environ.get('FAIR_BATCH_SIZE',    100))
EMS_IN        = os.environ.get(    'FAIR_EMS_IN',        'preprocessing/final/PACE_inventory_long.csv')
# For a sensitivity run, set EMS_IN to 'preprocessing/final/PACE_inventory_long_sens.csv'

RUNNING_SENS = EMS_IN.endswith('_sens.csv')
NOX_VAR = 'NOx aviation'
SSPs = ['ssp119']  # Currently hardcoded to only handle ssp119

# Internal inputs
CMIP_EMISSIONS = 'CMIP6/rcmip-emissions-annual-means-v5-1-0_original.csv'
CMIP_CONCENTRATIONS = 'CMIP6/rcmip-concentrations-annual-means-v5-1-0_original.csv'
CMIP_FORCING = 'CMIP6/rcmip-radiative-forcing-annual-means-v5-1-0_original.csv'

EMISSIONS_OUT = 'inputs/final/rcmip-emissions-annual-means-v5-1-0.csv'
CONCENTRATION_OUT = 'inputs/final/rcmip-concentrations-annual-means-v5-1-0.csv'
FORCING_OUT = 'inputs/final/rcmip-radiative-forcing-annual-means-v5-1-0.csv'

EMISSIONS_OUT_LONG = 'inputs/final/rcmip-emissions-annual-means-v5-1-0_long.csv'
FORCING_OUT_LONG = 'inputs/final/rcmip-forcing-annual-means-v5-1-0_long.csv'


def clean_slcp_ems(slcp_ems):
    """
    Clean ICCT emissions and standardize with FaIR conventions.
    """
    scenarios = slcp_ems['Scenario'].unique()

    # Convert all 'Species' names to lowercase
    slcp_ems['Species'] = slcp_ems['Species'].str.lower()

    # Pivot 'Scenario' column wide
    slcp_ems = slcp_ems.pivot_table(index=['Year', 'Species', 'Units', 'Source', 'Sector'], columns='Scenario', values='ems').reset_index()

    # Rename 'ems' species columns to match expected names
    slcp_ems_conversion = {'bc':'BC', 'ch4':'CH4', 'co2':'CO2 FFI', 'h2o':'Stratospheric water vapour',
                               'n2o':'N2O', 'nox':NOX_VAR, 'sox':'Sulfur', 'so2':'Sulfur'}
    slcp_ems = slcp_ems.replace(slcp_ems_conversion)

    return slcp_ems, scenarios


def define_species_mapping():
    """
    # Define a naming mapping between shorthand species names and RCMIP FaIR species names.
    """
    # TODO: Create a constants.py file to hold long lists like this
    species = ['CO2 FFI', 'CO2 AFOLU', 'CO2', 'CH4', 'N2O', 'Sulfur', 'BC', 'OC', 'NH3', 'NOx', 'VOC', 'CO', 'CFC-11',
               'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115', 'HCFC-22', 'HCFC-141b', 'HCFC-142b', 'CCl4', 'CHCl3',
               'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br', 'Halon-1202', 'Halon-1211', 'Halon-1301', 'Halon-2402',
               'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12', 'C6F14', 'C7F16', 'C8F18', 'NF3', 'SF6', 'SO2F2',
               'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa', 'HFC-32',
               'HFC-365mfc', 'HFC-4310mee', 'NOx aviation', 'Solar', 'Volcanic', 'Aerosol-radiation interactions',
               'Aerosol-cloud interactions', 'Ozone', 'Contrails', 'Light absorbing particles on snow and ice',
               'Stratospheric water vapour', 'Land use', 'Equivalent effective stratospheric chlorine']

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
    """
    Filter the CMIP6 emissions, concentrations, and forcing data to only include relevant data and linearly
    interpolate missing values.
    """
    # Filter the inputs to only the full species values
    ems = ems[ems['Variable'].str.endswith(tuple(species_to_rcmip.values()))]
    conc = conc[conc['Variable'].str.endswith(tuple(species_to_rcmip.values()))]
    forc = forc[forc['Variable'].str.endswith(tuple(species_to_rcmip.values()))]

    properties = pd.read_csv('examples/properties/properties.csv')
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


def handle_sensitivity_input(ems_orig, conc_orig, forc_orig, slcp_ems, species_to_rcmip, scenarios):
    """
    If a sensitivity run is being executed, create separate input files for batches of sensitivity runs.
    """
    id_cols = ['Year', 'Species', 'Units', 'Source', 'Sector']

    # Ensure 'Results/sens/{BATCH_SIZE}' exists
    if not os.path.exists(f'inputs/final/batches/{BATCH_SIZE}'):
        os.makedirs(f'inputs/final/batches/{BATCH_SIZE}')
    num_scenarios = len(scenarios)
    num_batches = round(num_scenarios/BATCH_SIZE)
    # Run in batches
    for i in range(0, num_scenarios, BATCH_SIZE):
        st = time.time()

        scenario_batch = list(scenarios[i:i + BATCH_SIZE])

        # Filter to batch
        slcp_ems_batch = slcp_ems[id_cols + scenario_batch].copy()

        # Align inputs with FaIR
        ems, conc, forc = align_inputs_with_fair(ems_orig.copy(), conc_orig.copy(), forc_orig.copy(), slcp_ems_batch,
                                                 species_to_rcmip, scenario_batch)

        # Store each batched input in a separate directory
        forc.to_csv(f'inputs/final/batches/{BATCH_SIZE}/rcmip-radiative-forcing-annual-means-v5-1-0_{round(i/BATCH_SIZE)}.csv', index=False)
        ems.to_csv(f'inputs/final/batches/{BATCH_SIZE}/rcmip-emissions-annual-means-v5-1-0_{round(i/BATCH_SIZE)}.csv', index=False)
        conc.to_csv(f'inputs/final/batches/{BATCH_SIZE}/rcmip-concentrations-annual-means-v5-1-0_{round(i/BATCH_SIZE)}.csv', index=False)

        print(f'Ran batch {i/BATCH_SIZE} of {num_batches} ({i/BATCH_SIZE/num_batches*100:.2f}%) in {time.time() - st:.2f} seconds')


def align_inputs_with_fair(ems, conc, forc, slcp_ems, species_to_rcmip, scenarios):
    """
    Align the inputs with FaIR by creating new scenarios and adjusting the emissions.
    """
    ems, conc, forc = create_new_scenarios(ems, conc, forc, scenarios, slcp_ems)

    forc = adjust_forc(slcp_ems, forc, scenarios)

    ems = modify_emissions(ems, slcp_ems, species_to_rcmip, scenarios)

    # Ensure no duplicates
    assert ems.duplicated().sum() == 0, "There are duplicates in the emissions data"
    assert forc.duplicated().sum() == 0, "There are duplicates in the forcing data"
    assert conc.duplicated().sum() == 0, "There are duplicates in the concentrations data"

    return ems, conc, forc


def adjust_forc(slcp_ems, forc, scenarios):
    """
    Modify the radiative forcing data to reflect the ICCT estimates for aviation. Includes contrails, stratospheric H2O,
    and stratospheric NOx.
    """
    ct_var = 'Effective Radiative Forcing|Anthropogenic|Other|Contrails and Contrail-induced Cirrus'
    h2o_var = 'Effective Radiative Forcing|Anthropogenic|Other|CH4 Oxidation Stratospheric H2O'
    SSP = 'ssp119'

    # Use ICCT estimates for contrail BAU and striving under SSP119. Set as contrail total forcing since
    # contrails come from no other source
    # Contrails ---------------------------------------------------------------------------------------------------
    contrail_years = sorted(slcp_ems[slcp_ems['Species'] == 'contrails']['Year'].unique())
    for yr in contrail_years:
        # Adjust the BAU for intervention scenarios
        for scen in scenarios:
            ct_val = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == 'contrails')][scen].values[0]

            forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_{scen}'), str(yr)] = ct_val
            forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_contrails_Aviation_{scen}'), str(
                yr)] = ct_val

        # Set to 0 in zero-out scenarios, because contrails come from no other source
        forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_zero_tra'), str(yr)] = 0
        forc.loc[(forc['Variable'] == ct_var) & (forc['Scenario'] == f'{SSP}_contrails_Aviation_zero'), str(
            yr)] = 0

    # We only consider contrails for sensitivity runs to save on compute
    if RUNNING_SENS:
        return forc

    # Stratospheric H2O ----------------------------------------------------------------------------------------------
    h2o_years = sorted(slcp_ems[slcp_ems['Species'] == 'Stratospheric water vapour']['Year'].unique())
    for yr in h2o_years:
        bau_h2o = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == 'Stratospheric water vapour')][BASELINE_SCEN].values[0]

        # Adjust the BAU for contrail scenarios
        for striving_scen in [s for s in scenarios if s!=BASELINE_SCEN]:
            striving_h2o = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == 'Stratospheric water vapour')][striving_scen].values[0]

            forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_{striving_scen}'), str(yr)] += -1*(bau_h2o - striving_h2o)
            forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_Stratospheric water vapour_Aviation_{striving_scen}'), str(
                yr)] = forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_{striving_scen}'), str(yr)]

        forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_zero_tra'), str(yr)] += -1*bau_h2o

        # Add zero contrails scenario
        forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_Stratospheric water vapour_Aviation_zero'), str(
            yr)] += -1*bau_h2o

    # Run NOx as an additional perturbation to H2O in the model because there is no valid NOx aviation forcing input
    # NOx ---------------------------------------------------------------------------------------------------------
    nox_years = sorted(slcp_ems[slcp_ems['Species'] == NOX_VAR]['Year'].unique())
    for yr in nox_years:
        bau_h2o = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == NOX_VAR)][BASELINE_SCEN].values[0]

        # Adjust the BAU for contrail scenarios
        for striving_scen in [s for s in scenarios if s!=BASELINE_SCEN]:
            striving_h2o = slcp_ems[(slcp_ems['Year'] == yr) & (slcp_ems['Species'] == NOX_VAR)][striving_scen].values[0]

            forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_{striving_scen}'), str(yr)] += -1*(bau_h2o - striving_h2o)
            forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_{NOX_VAR}_Aviation_{striving_scen}'), str(
                yr)] += -1*(bau_h2o - striving_h2o)

        forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_zero_tra'), str(yr)] += -1*bau_h2o

        # Add zero contrails scenario
        forc.loc[(forc['Variable'] == h2o_var) & (forc['Scenario'] == f'{SSP}_{NOX_VAR}_Aviation_zero'), str(
            yr)] += -1*bau_h2o

    return forc


def create_new_scenarios(ems, conc, forc, scenarios, slcp_ems):
    """
    Define a scenario for each pollutant, Sector, Source, and current 'Scenario' name.
    """
    SSP = 'ssp119'
    # Copy the ssp119 scenario to ssp119_striving
    for scenario in scenarios:
        ems = pd.concat([ems, ems[ems['Scenario'] == SSP].replace(SSP, f'{SSP}_{scenario}')])
        forc = pd.concat([forc, forc[forc['Scenario'] == SSP].replace(SSP, f'{SSP}_{scenario}')])
        conc = pd.concat([conc, conc[conc['Scenario'] == SSP].replace(SSP, f'{SSP}_{scenario}')])

    ems = pd.concat([ems, ems[ems['Scenario'] == SSP].replace(SSP, f'{SSP}_zero_tra')])
    forc = pd.concat([forc, forc[forc['Scenario'] == SSP].replace(SSP, f'{SSP}_zero_tra')])
    conc = pd.concat([conc, conc[conc['Scenario'] == SSP].replace(SSP, f'{SSP}_zero_tra')])

    # If running a contrail sensitivity run, skip species-specific scenarios (only contrails are modified)
    if not RUNNING_SENS:
        species_list = slcp_ems['Species'].unique()

        for specie in species_list:
            for sector in slcp_ems['Sector'].unique():
                # for source in slcp_ems['Source'].unique():
                if slcp_ems[(slcp_ems['Species'] == specie) & (slcp_ems['Sector'] == sector)].empty:
                    continue

                for scenario in scenarios:
                    ems = pd.concat([ems, ems[(ems['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_{scenario}')])
                    conc = pd.concat([conc, conc[(conc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_{scenario}')])
                    forc = pd.concat([forc, forc[(forc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_{scenario}')])

                # Add zero scenarios
                ems = pd.concat([ems, ems[(ems['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_zero')])
                conc = pd.concat([conc, conc[(conc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_zero')])
                forc = pd.concat([forc, forc[(forc['Scenario'] == SSP)].replace(SSP, f'{SSP}_{specie}_{sector}_zero')])

    return ems, conc, forc


def modify_emissions(ems, slcp_ems, species_to_rcmip, scenarios):
    """Fill in ems with the values from slcp_ems"""
    slcp_ems = convert_units_ems(slcp_ems, ems, species_to_rcmip)

    ems = perturb_baseline_ems(ems, slcp_ems, species_to_rcmip, scenarios)

    return ems

def perturb_baseline_ems(ems, slcp_ems, species_to_rcmip, scenarios):
    """
    Modify the emissions data to reflect the ICCT estimates for mitigation scenarios.

    ems is wide by year. slcp_ems is long. For scenarios, subtract the difference avoided emissions. Currently, all
    ems represent the baseline emissions.
    """
    for specie, specie_rcmip_name in species_to_rcmip.items():
        if (specie not in slcp_ems['Species'].values) | (specie == 'Stratospheric water vapour'):
            continue

        # Calculate avoided emissions
        ems_spec = slcp_ems[(slcp_ems['Species'] == specie)].copy()
        for scen in scenarios:
            ems_spec[f'Avoided_{scen}'] = ems_spec[scen] - ems_spec[BASELINE_SCEN]

            for yr in ems_spec['Year'].unique():
                ems_yr = ems_spec[ems_spec['Year'] == yr].copy()

                # Add the avoided emissions to the ems dataframe
                ems.loc[(ems["Scenario"] == f'ssp119_{scen}') & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                        (ems["Region"] == "World"), str(yr)] += ems_yr[f'Avoided_{scen}'].values
                ems.loc[(ems["Scenario"] == f'ssp119_{specie}_Aviation_{scen}') & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                        (ems["Region"] == "World"), str(yr)] += ems_yr[f'Avoided_{scen}'].values
                if scen == BASELINE_SCEN:
                    ems.loc[(ems["Scenario"] == f'ssp119_{specie}_Aviation_zero') & (
                        ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                            (ems["Region"] == "World"), str(yr)] += -1*ems_yr[BASELINE_SCEN].values
                    ems.loc[(ems["Scenario"] == f'ssp119_zero_tra') & (
                        ems["Variable"].str.endswith("|" + specie_rcmip_name)) &
                            (ems["Region"] == "World"), str(yr)] += -1*ems_yr[BASELINE_SCEN].values

    return ems


def convert_units_ems(slcp_ems, ems, species_to_rcmip):
    SSP = 'ssp119'
    id_cols = ['Scenario', 'Year', 'Species', 'Units', 'Source', 'Sector']
    scenarios = [col for col in slcp_ems.columns if col not in id_cols]
    conversion_factors = {
        ('Gt', 'Mt'): 1e3,
        ('Gt', 'kt'): 1e6,
        ('Mt', 'kt'): 1e3,
        ('kt', 'Mt'): 1e-3,
        ('Mt', 'Mt'): 1,
    }

    # Modify the baseline emissions to reflect the ICCT estimates
    for specie, specie_rcmip_name in species_to_rcmip.items():
        if ((specie not in slcp_ems['Species'].values) | (specie == 'Stratospheric water vapour')
                | (specie == NOX_VAR)):
            continue
        print(specie)
        icct_units = slcp_ems[(slcp_ems['Species'] == specie)]['Units'].values[0]
        expected_units = ems.loc[(ems["Scenario"] == SSP) & (ems["Variable"].str.endswith("|" + specie_rcmip_name)) & (
                    ems["Region"] == "World")]['Unit'].values[0]
        expected_units = str(expected_units).split(' ')[0]
        assert expected_units == 'Mt' or expected_units == 'kt', f"Expected units are not Mt or kt, but {expected_units}"

        factor = conversion_factors.get((icct_units, expected_units))

        print(f"Converting {specie} from {icct_units} to {expected_units} with factor {factor}")

        if factor:
            slcp_ems.loc[(slcp_ems['Species'] == specie), scenarios] *= factor
            slcp_ems.loc[(slcp_ems['Species'] == specie), 'Units'] = expected_units

    return slcp_ems


def main():
    """
    Execute the script.
    """
    st = time.time()
    slcp_ems = pd.read_csv(EMS_IN) # Custom ICCT emissions/forcing inputs

    # CMIP6 default inputs
    ems = pd.read_csv(CMIP_EMISSIONS) # Emissions
    conc = pd.read_csv(CMIP_CONCENTRATIONS) # Concentrations
    forc = pd.read_csv(CMIP_FORCING) # Radiative forcing

    # Clean inputs
    species_to_rcmip = define_species_mapping()
    slcp_ems, scenarios = clean_slcp_ems(slcp_ems)
    ems, conc, forc = clean_inputs(ems, conc, forc, species_to_rcmip)

    if RUNNING_SENS:
        # Fully handles adjustments and export for sensitivity runs
        handle_sensitivity_input(ems, conc, forc, slcp_ems, species_to_rcmip, scenarios)
    else:
        # Align inputs with FaIR
        ems, conc, forc = align_inputs_with_fair(ems, conc, forc, slcp_ems, species_to_rcmip, scenarios)

        # Export FaIR-ready final inputs
        ems.to_csv(EMISSIONS_OUT, index=False)
        conc.to_csv(CONCENTRATION_OUT, index=False)
        forc.to_csv(FORCING_OUT, index=False)

        # Create a long version along year for validation
        ems_long = ems.melt(id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Mip_Era', 'Activity_Id'], var_name='Year', value_name='ems')
        ems_long.to_csv(EMISSIONS_OUT_LONG, index=False)
        forc_long = forc.melt(id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Mip_Era', 'Activity_Id'], var_name='Year', value_name='ems')
        forc_long.to_csv(FORCING_OUT_LONG, index=False)

    print(f"Script executed in {time.time() - st:.2f} seconds.")


if __name__ == "__main__":
    main()
