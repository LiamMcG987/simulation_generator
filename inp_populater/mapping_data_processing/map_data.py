import os
import numpy as np
import pandas as pd
import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as Rotate
from ..mapping_data_processing.check_data import check_orientation
from ..mapping_data_processing.parse_data import append_profile
from ..mapping_data_logging.data_logging import UpdateLog
from ..mapping_data_processing.scaling_data import global_mass_scaling, local_mass_scaling_lmg
from ..mapping_data_processing.scaling_data import calc_bottle_mass
from ..mapping_data_processing.parse_data_3dx import InstanceThreeDx


# New
# Regression fitting
def get_column_name(map_from, exp_heading, alt_heading):
    heading = exp_heading
    for col in map_from.columns:
        if col.lower() == alt_heading:
            heading = col
    return heading


def fit_modulus_regression(inputs):
    map_from_coords = inputs['map_from_coords']
    modulus_map_shape = inputs['modulus_fit_shape']

    sr_col = get_column_name(map_from_coords, 'eqsr', ['eqsr_sr', 'sr'])
    rel_mod_col = get_column_name(map_from_coords, 'rel_modulus', 'rel_mod')
    rel_sr_col = get_column_name(map_from_coords, 'rel_stretch_ratio', 'rel_sr')
    mod_fit = fit_mod_sr(np.asarray(map_from_coords[rel_mod_col].dropna()),
                         np.asarray(map_from_coords[rel_sr_col].dropna()),
                         modulus_map_shape)
    map_from_coords['modulus'] = get_map_from_mod_values(np.asarray(map_from_coords[sr_col]), mod_fit)
    return map_from_coords


def get_map_from_mod_values(stretch_ratios, poly):
    modulus = poly(stretch_ratios)
    return modulus


def fit_mod_sr(modulus, stretch_ratios, poly_deg):
    poly_shape = {
        'linear': 1,
        'poly': 2
    }
    degree = poly_shape.get(poly_deg.lower())
    if degree is None:
        raise TypeError("Cannot access specified modulus fit.")

    coefs = np.polyfit(stretch_ratios, modulus, degree)
    poly = np.poly1d(coefs)
    return poly


# End of regression fitting


# Mapping
def rotate_sbm_data(data, rot_axis, rot_angle):
    """ Rotates sbm bottle data to align it with the mapping bottle. """

    # Get coordinates
    if isinstance(data, pd.DataFrame):
        coords = data[['x', 'y', 'z']].to_numpy()
    else:
        coords = data
    # Generate rotation vector
    rot_vector = np.radians(rot_angle) * rot_axis
    rot = Rotate.from_rotvec(rot_vector)
    # Rotate xyz data
    rot_coords = rot.apply(coords)
    if isinstance(data, pd.DataFrame):
        data[['x', 'y', 'z']] = rot_coords
    else:
        data = rot_coords
    return data


def user_check():
    if click.confirm('Continue mapping?', default=True):
        click.echo('Mapping running.')
    else:
        raise SystemExit('Mapping process cancelled.')


def check_mapping(mapped_coords, cs, inputs, colors_map='jet', angles=None):
    """ Checks/verifies mapping algorithm. """

    angles = angles if angles is not None else [0, 180, 270]
    x, y = mapped_coords['x'].to_numpy(), mapped_coords['y'].to_numpy()
    z, cs_vals = mapped_coords['z'].to_numpy(), mapped_coords[str(cs)].to_numpy()
    fig = plt.figure()
    cm = plt.get_cmap(colors_map)
    cNorm = mpl.colors.Normalize(vmin=min(cs_vals), vmax=max(cs_vals))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.set_array(cs_vals)

    for i, angle in enumerate(angles):
        ax = fig.add_subplot(1, len(angles), i + 1, projection='3d')
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs_vals))
        ax.view_init(10, angle)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.set_title('{} map, {}{}'.format(str(cs).capitalize(), str(angle), "\u00b0"))
        ax.set_axis_off()
        ax.autoscale(enable=True)

    map_from_file_name = os.path.splitext(os.path.split(inputs['map_from_file'])[-1])[0].replace(' ', '')
    map_to_file_name = os.path.splitext(os.path.split(inputs['map_to_file'])[-1])[0].replace(' ', '')
    plt.suptitle('{} mapped to {}\n{} Map'.format(map_from_file_name, map_to_file_name,
                                                  str(cs).capitalize()),
                 weight='bold')
    plt.tight_layout()
    plt.savefig('{}_to_{}_{}_map.png'.format(map_from_file_name, map_to_file_name,
                                             str(cs).capitalize()))
    plt.show()


def visualise_profiles(dataset, variable, overlay=None):
    overlay = overlay if overlay is not None else False

    fig = plt.figure()
    ax = fig.add_subplot()

    if overlay:
        y_1 = dataset[0]['z']
        y_2 = dataset[1]['z']
        profile_var_1 = dataset[0][variable]
        profile_var_2 = dataset[1][variable]

        ax.scatter(y_1, profile_var_1)
        ax.scatter(y_2, profile_var_2)
        ax.set_xlabel('y')
        ax.set_ylabel(variable)

        plt.tight_layout()
        plt.show()

    else:
        y = dataset['z']
        profile_var = dataset[variable]

        ax.scatter(y, profile_var)
        ax.set_xlabel('y')
        ax.set_ylabel(variable)

        plt.tight_layout()
        plt.show()
        # plt.savefig(f'{variable}_plot_{dataset}.png')


def map_data(map_from, map_to, inputs):
    # Initialise columns for sth and modulus
    mf_coords = np.asarray(map_from[['x', 'y', 'z']])
    mt_coords = np.asarray(map_to[['x', 'y', 'z']])
    modulus_map, sth_map = inputs['modulus_map'], inputs['sth_map']

    if modulus_map and sth_map:
        map_to_data = np.zeros((len(mt_coords), 2))
    else:
        map_to_data = np.zeros((len(mt_coords), 1))

    for node_id, node in enumerate(mt_coords):
        dists = distance.cdist(np.array([node]), mf_coords, 'euclidean')[0]
        min_idx = np.argmin(dists)
        if modulus_map and sth_map:
            map_to_data[node_id, 0] = map_from.loc[min_idx, 'sth']
            map_to_data[node_id, 1] = map_from.loc[min_idx, 'modulus']
        elif modulus_map and not sth_map:
            map_to_data[node_id, 0] = map_from.loc[min_idx, 'modulus']
        else:
            map_to_data[node_id, 0] = map_from.loc[min_idx, 'sth']
    return concat_mapped_data(map_to, map_to_data, inputs)


def concat_mapped_data(map_to_coords, mapped_data, inputs):
    modulus_map, sth_map = inputs['modulus_map'], inputs['sth_map']
    if modulus_map and sth_map:
        map_to_coords[['sth', 'modulus']] = mapped_data
    elif mapped_data[0, 0] >= inputs['modulus_limits'][0]:
        map_to_coords['modulus'] = mapped_data
    else:
        map_to_coords['sth'] = mapped_data
    return map_to_coords


def limit_mapped_values(map_to, limits):
    vals = map_to.to_numpy()
    for i, val in enumerate(vals):
        if val < limits[0]:
            vals[i] = limits[0]
        elif val > limits[1]:
            vals[i] = limits[1]
    return vals


def map_sth_elemental(connectivity_set, mapped_values):
    mapped_values_np = mapped_values.to_numpy()[:, :-1]
    col_length = np.shape(mapped_values_np)[-1] - 1
    avgs_all = []

    for elements in connectivity_set:
        node = elements[0]
        element_nodes = elements[1:]
        no_of_nodes = len(elements) - 1
        avgs = get_element_sth_return_avg(
            element_nodes, no_of_nodes, mapped_values_np, col_length)
        avgs = np.concatenate(([node], avgs))
        avgs_all.append(avgs)

    avgs_all = np.asarray(avgs_all)
    columns_elemental = ['x', 'y', 'z', 'sth']

    mapped_values_elemental_sth = pd.DataFrame(avgs_all[:, 0])
    mapped_values_elemental_sth.columns = {'nodes'}

    for i in range(len(columns_elemental)):
        mapped_values_elemental_sth[columns_elemental[i]] = avgs_all[:, i + 1]
    return mapped_values_elemental_sth


def get_element_sth_return_avg(elements, divisor, mapped_values, col_length):
    totals = np.zeros(shape=col_length)
    for node in elements:
        idx = np.where(mapped_values[:, 0] == node)
        try:
            for i in range(len(totals)):
                totals[i] += mapped_values[idx, i + 1]
        except ValueError:
            pass
    totals_avg = totals / divisor
    return totals_avg


def mapping_handler(inputs):
    map_from_coords, map_to_coords = inputs['map_from_coords'], inputs['map_to_coords']
    # visualise_profiles(map_from_coords, 'sth')
    # visualise_profiles(map_from_coords, 'modulus')
    map_to_bottle = InstanceThreeDx(inputs['map_to_file'])
    connectivity = inputs['connectivity']
    if inputs['sth_profile'] is not None:
        map_from_coords = append_profile(map_from_coords, inputs['sth_profile'])
    if inputs['mod_profile'] is not None:
        map_from_coords = append_profile(map_from_coords, inputs['mod_profile'])
    for inp in inputs['rotate']:
        rotate_sbm_data(map_from_coords, np.asarray(inp['axis']), inp['degrees'])
    map_from_coords[inputs['axis_height']] = map_from_coords[inputs['axis_height']] + inputs['shift_factor']

    # check_orientation(map_from_coords, map_to_coords, plane=('z', 'x'))
    # user_check()

    if inputs['regression_fit']:
        map_from_coords = fit_modulus_regression(inputs)

    mapped_profiles = map_data(map_from_coords, map_to_coords, inputs)
    mapped_profiles_scaled, sf = scaling_handler(map_to_bottle, map_from_coords,
                                                 mapped_profiles, inputs)

    mapped_profiles_elemental_scaled_sth = map_sth_elemental(connectivity, mapped_profiles_scaled)
    # visualise_profiles(mapped_profiles_elemental_scaled_sth, 'sth')
    # visualise_profiles(mapped_profiles_scaled, 'modulus')
    # visualise_profiles([mapped_profiles_elemental_scaled_sth, map_from_coords], 'sth', overlay=True)
    # visualise_profiles([mapped_profiles_scaled, map_from_coords], 'modulus', overlay=True)

    mapped_profiles_scaled['modulus'] = limit_mapped_values(
        mapped_profiles_scaled['modulus'],
        inputs['modulus_limits'])

    scaled_bottle_mass = 1e6 * sum(calc_bottle_mass(map_to_bottle,
                                                    mapped_profiles_scaled, inputs,
                                                    height_range=[inputs['shift_factor']]))
    print('Scaled mass of mapped bottle: {:.2f} g'.format(scaled_bottle_mass + inputs['neck_mass']))

    # check_mapping(mapped_profiles, 'modulus', inputs)
    # check_mapping(mapped_profiles_elemental_scaled_sth, 'sth', inputs)
    scaling_params = dict(list(inputs.items())[6:])
    scaling_params['scaled_mass'] = round(scaled_bottle_mass, 3)
    scaling_params['scale_factor'] = round(sf, 3)
    log = UpdateLog()
    log.update_log_map(inputs['map_from_file'], inputs['map_from_shape'],
                       inputs['map_to_file'], inputs['map_to_shape'],
                       scaling_params)
    mapped_profiles_scaled.to_csv('rkt_musk_750_mod.csv')
    mapped_profiles_elemental_scaled_sth.to_csv('rkt_musk_750_sth.csv')
    return mapped_profiles_scaled, mapped_profiles_elemental_scaled_sth


# End mapping


# Scaling
def scaling_handler(bottle, map_from, mapped_coords, inputs):
    sf = 0.00
    if inputs['scaling'] == 'global':
        mapped_coords_scaled, sf = global_mass_scaling(bottle, mapped_coords, inputs)
    elif inputs['scaling'] == 'local':
        # mapped_coords_scaled = local_mass_scaling(map_to_bottle, mapped_coords, map_from, inputs)
        mapped_coords_scaled = local_mass_scaling_lmg(inputs['map_from_file'], bottle,
                                                      mapped_coords, map_from, inputs)
    else:
        mapped_coords_scaled = mapped_coords
    return mapped_coords_scaled, sf
# End scaling
# End new
