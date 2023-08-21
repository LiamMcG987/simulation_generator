import numpy as np
from ..mapping_data_processing.return_instance_data import Instance


def get_instance_data(lines, instance_types):
    """ Reads lines and extracts data for given instances. """

    # Identify index breaks for all instances
    start_idxs, end_idxs = [], []
    for idx, line in enumerate(lines):
        if '*Instance, name=' in line:
            start_idxs.append(idx)
        elif '*End Instance' in line:
            end_idxs.append(idx)
    idx_pairs = list(zip(start_idxs, end_idxs))
    # Extract data for data types of interest
    instance_data = {}
    for instance in instance_types:
        for idx_pair in idx_pairs:
            if '{}-1'.format(instance) in lines[idx_pair[0]]:
                instance_data[instance] = \
                    Instance(
                        instance,
                        idx_pair,
                        lines[idx_pair[0]:idx_pair[1] + 1])
    return instance_data


def axis2idx(axis):
    """ Returns equivalent index for text axis input (x, y, z). """

    dirs = ['nodes', 'x', 'y', 'z']
    return dirs.index(axis)


def scale_data(df, sf, inputs):
    """ Multiply thickness by scaling factor and adjust modulus (if required). """

    if inputs['adj_sth']:
        df['sth'] = df['sth'] * sf
    if inputs['adj_mod']:
        df['mod'] = df['mod'] * 1 / sf
    return df


def nodal_mass_cal(bottle, element_masses):
    connectivity = np.asarray(bottle.ntri_connectivity)
    coords = np.asarray(bottle.coords)

    nodal_masses = []
    for coord in coords:
        node = coord[0]
        connectivity_idx = np.where(connectivity == node)[0]
        face_list = np.asarray(connectivity_idx)

        adj_element_mass = []
        for n_element in face_list:
            mass = element_masses[n_element]
            adj_element_mass.append(mass)

        nodal_mass_avg = sum(adj_element_mass) / 3
        nodal_masses.append(nodal_mass_avg)
    return np.asarray(nodal_masses)


def element_mass_calc(bottle, df, density):
    connectivity = np.asarray(bottle.ntri_connectivity)
    coords = np.asarray(bottle.coords)
    sth_map = df[['nodes', 'sth']].to_numpy()

    element_masses = []
    element_masses_w_nodes = []
    for connectivity_set in connectivity:
        nodes = [x for x in connectivity_set]

        coords_set = []
        sth_set = []
        for node in connectivity_set:
            idx = np.where(coords[:, 0] == node)[0][0]
            coords_set.append(coords[idx, 1:])
            sth_set.append(sth_map[idx, 1])

        dims = []
        for x in range(len(coords_set)):
            coord_first, coord_second = coords_set[x % len(coords_set)], coords_set[(x + 1) % len(coords_set)]
            dim = coord_first - coord_second
            dims.append(np.linalg.norm(dim))

        mass = calc_elemental_mass(dims, sth_set, density)
        element_masses.append(mass)
        element_masses_w_nodes = np.concatenate((nodes, [mass]))
    return np.asarray(element_masses), np.asarray(element_masses_w_nodes)


def calc_elemental_mass(dimensions, sth, density):
    s = sum(dimensions) / 2
    area = (s * (s - dimensions[0]) * (s - dimensions[1]) * (s - dimensions[2])) ** 0.5
    thickness = sum(sth) / 3
    volume = area * thickness
    mass = volume * density
    return mass


def calc_bottle_mass(bottle, df_bottle, inputs, height_range=None):
    """ Calculates mass of bottle body (from nodes with a height coordinate = 0). """
    height_range = height_range if not None else []

    element_masses, element_masses_nodes = element_mass_calc(bottle, df_bottle, inputs['density'])
    nodal_masses = nodal_mass_cal(bottle, element_masses)

    # Get mass of bottle without neck (check orientation first)
    if len(height_range) == 2:
        body_nodes = np.where(
            (bottle.coords[:, axis2idx(inputs['axis_height'])] <= height_range[0]) &
            (bottle.coords[:, axis2idx(inputs['axis_height'])] > height_range[1])
        )[0]
    elif len(height_range) == 1:
        body_nodes = np.where(np.asarray(bottle.coords)[:, axis2idx(inputs['axis_height'])] <= height_range[0])[0]
    else:
        return nodal_masses
    body_mass = nodal_masses[body_nodes]
    return body_mass


def global_mass_scaling(bottle, df_bottle, inputs):
    """ Global mass scaling for map-from to map-to bottle. """

    unscaled_bottle_mass = sum(calc_bottle_mass(bottle, df_bottle, inputs, height_range=[inputs['shift_factor']])) * 1e6
    body_mass = inputs['bottle_mass'] - inputs['neck_mass']
    print('Unscaled mass of mapped bottle: {:.2f} g'.format(unscaled_bottle_mass + inputs['neck_mass']))
    gsf = body_mass / unscaled_bottle_mass
    print('Global scaling factor: {:.2f}'.format(gsf))
    df_bottle = scale_data(df_bottle, gsf, inputs)
    return df_bottle, gsf


def local_mass_scaling(bottle, df_bottle, df_sbm, inputs):
    """ Global mass scaling for map-from to map-to bottle. """

    inputs['section_cuts'] = [i + inputs['shift_factor'] for i in inputs['section_cuts']]
    inputs['section_cuts'].insert(0, inputs['shift_factor'])
    inputs['section_cuts'].insert(len(inputs['section_cuts']), df_sbm[inputs['axis_height']].min())
    for idx in range(len(inputs['section_cuts']) - 1):
        # Get mass of map-to bottle section
        height_range = [inputs['section_cuts'][idx], inputs['section_cuts'][idx + 1]]
        section_mass = sum(calc_bottle_mass(bottle, df_bottle, inputs, height_range=height_range)) * 1e6
        if idx == 0:
            map_from_section_mass = inputs['section_masses'][idx] - inputs['neck_mass']
        else:
            map_from_section_mass = inputs['section_masses'][idx]
        # Compare against SBM simulation and calculate local scaling factor (lsf)
        lsf = map_from_section_mass / section_mass
        print('Local scaling factor {}: {:.2f}'.format(idx + 1, lsf))
        # Scale thickness of bottle
        df_sliced = df_bottle.loc[
            (df_bottle[(inputs['axis_height'])] <= height_range[0]) &
            (df_bottle[inputs['axis_height']] >= height_range[1])
            ]
        df_sliced = scale_data(df_sliced, lsf, inputs)
        df_bottle.update(df_sliced)
    return df_bottle


def locating_splits(dataset_map, dataset_axi, min_split, max_split):
    """ Returns scaling factor when given 3d and axi datasets, and a min and max (from specified breaks). """

    index_map = dataset_map.index[(dataset_map['y'] <= - abs(min_split)) & (dataset_map['y'] >= - abs(max_split))]
    index_axi = dataset_axi.index[(dataset_axi['y'] <= - abs(min_split)) & (dataset_axi['y'] >= - abs(max_split))]
    sf = dataset_map.loc[index_map, 'nodal_masses'].sum() / dataset_axi.loc[index_axi, 'nodal_masses'].sum()
    dataset_map.loc[index_map, 'sth'] = dataset_map.loc[index_map, 'sth_to_scale'] / sf
    return sf


def check_breaks(dataset_map, breaks):
    max_y = np.max(abs(dataset_map['y']))
    if any(abs(val) > max_y for val in breaks):
        raise IndexError('Split(s) are not in range of dataset. Max y-coord = ' + str(max_y))


def local_mass_scaling_lmg(map_from_bottle, map_to_bottle, mapped_coords, map_from, inputs):
    """
    function takes in axi and 3d dimensions, along with specified breaks in geometry.
    data checked to ensure no nodes exist above 0 in y-axis.
    """

    # axi_data = pd.DataFrame(pd.read_csv('musk_750_rb.csv'), columns=['x', 'y'])
    # mapped_coords = pd.DataFrame(mapped_dims.coords, columns=['x', 'y', 'z'])
    # mapped_coords = mapped_coords.drop(mapped_coords[mapped_coords['y'] > 0].index)
    inputs['section_cuts'] = [i + inputs['shift_factor'] for i in inputs['section_cuts']]
    check_breaks(mapped_coords, inputs['section_cuts'])

    map_from['nodal_masses'] = calc_bottle_mass(map_from_bottle, map_from, inputs['density'])
    mapped_coords['nodal_masses'] = calc_bottle_mass(map_to_bottle, mapped_coords, inputs['density'])
    mapped_coords['sth_to_scale'] = mapped_coords['sth']
    sf_values = {}

    for idx, val in enumerate(inputs['section_cuts']):
        if val == inputs['section_cuts'][0]:
            sf_values[idx + 1] = locating_splits(mapped_coords, map_from, 0, val)
        elif val == inputs['section_cuts'][-1]:
            sf_values[idx + 1] = locating_splits(mapped_coords, map_from, inputs['section_cuts'][idx - 1], val)
            # At final specified split, section above and below must be populated
            sf_values[idx + 2] = locating_splits(mapped_coords, map_from, val, mapped_coords['y'].min())
        else:
            sf_values[idx + 1] = locating_splits(mapped_coords, map_from, inputs['section_cuts'][idx - 1], val)
    return sf_values, mapped_coords
