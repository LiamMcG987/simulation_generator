import mat4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from ..mapping_data_logging.data_logging import UpdateLog
from ..mapping_data_processing.parse_data_3dx import InstanceThreeDx

geometry = {
    2: 'Axisymmetric',
    3: '3D'
}


def check_data_from(input_file):
    """Directs data dep on file extension"""
    print("Checking map_from file.")
    file_ext_handler(input_file)


def check_data_to(input_file):
    print("Checking map_to file.")
    file_ext_handler(input_file)


def file_ext_handler(input_file):
    """Function handler - directs files depending on ext"""
    file_ext = os.path.splitext(input_file)[-1]
    if file_ext == '.inp':
        coords, shape, connectivity, nset = check_data_inp(input_file)
        print_data_inp(coords, shape)
    elif file_ext in ['.csv', '.xlsx']:
        df, ndim, cols_used, cols_not_used = check_data_csv(input_file, file_ext)
        print_data_csv(ndim, cols_used, cols_not_used)
    elif file_ext == '.mat':
        nodes, x, y, sth, shape = check_data_matlab(input_file)
        print_data_matlab(x, y, shape)
    else:
        print('File extension not recognised - try .inp, .csv, .xlsx or .mat.')
    log = UpdateLog()
    log.update_log_check(input_file)


def check_data_inp(inp_deck):
    coords = False
    bottle = InstanceThreeDx(inp_deck)
    coords, shape, connectivity, nset = bottle.coords, bottle.shape, bottle.connectivity, bottle.nodes
    if not coords:
        print('Error in obtaining co-ordinates from .inp file.')
        sys.exit()
    else:
        return coords, shape, connectivity, nset


def _3dx_check(lines):
    check_end = None
    for line_i, line in enumerate(lines):
        if '*node' in line.lower():
            check_end = line_i
            break

    if check_end is None:
        raise ValueError('Unable to check platform of .inp file. Ensure correct headings and keywords are present.')
    for line in lines[:check_end]:
        if '3dexperience' in line.lower():
            return True
    return False


def locating_coords_inp(val, coord_idx, lines):
    coord_i = coord_idx.index(val)
    coords_begin_end = coord_idx[coord_i:coord_i + 2]
    coords = lines[coords_begin_end[0] + 1:coords_begin_end[1]]
    shape = len(coords[0].split(',')[1:])
    return coords, shape


def print_data_inp(coords, shape):
    print("""
    Data found from .inp file.
    Geometry: {}.
    Co-ords begin: {}.
    Co-ords end: {}.
    """.format(geometry.get(shape), coords[0], coords[-1]))


def check_data_csv(csv, ext):
    if ext == '.csv':
        df = pd.read_csv(csv)
    else:
        df = pd.read_excel(csv)
    cols = ['nodes', 'x', 'y', 'z', 'sth', 'eqsr', 'modulus', 'rel_stretch_ratio', 'rel_modulus']
    cols_present, cols_used, cols_not_used = [], [], []
    try:
        for column in df.columns:
            for target in cols:
                if target == column:
                    cols_used.append(column)
            cols_present.append(column)
        if len(list(set(cols_present) - set(cols_used))) == 0:
            cols_not_used = None
        else:
            cols_not_used = list(set(cols_present) - set(cols_used))
    except IndexError:
        print('Columns not labelled/not present.')

    df = df[cols_used]
    if 'z' not in cols_used:
        shape = 2
        df.loc[:, 'z'] = np.zeros(shape=len(df))
    else:
        shape = 3
    return df, shape, cols_used, cols_not_used


def print_data_csv(ndim, cols_used, cols_not_used):
    print("""
    Data found from .csv/.xlsx file.
    Geometry: {}.
    Columns used: {}.
    Columns not recognized: {}.
    """.format(geometry.get(ndim), cols_used, cols_not_used))


def check_data_matlab(mat):
    data = mat4py.loadmat(mat)
    nodes, x, y, sth = [], [], [], []
    shape = 2
    try:
        for idx, array in enumerate(data['UserData'][0]):
            nodes.append(array[0])
            x.append(array[1])
            y.append(array[2])
            sth.append(array[5])
    except IndexError:
        print('Columns not recognised')
    return nodes, x, y, sth, shape


def print_data_matlab(x, y, shape):
    print("""
    Data found from .mat file.
    Geometry: {}.
    Co-ords begin: {}, {}.
    Co-ords end: {}, {}.
    """.format(geometry.get(shape),
               x[0], y[0],
               x[-1], y[-1]))


def check_orientation(map_from, map_to, plane='', coi=None):
    """ Checks/verifies map-to bottles is in the correct orientation. """
    coi = coi if coi is not None else 3
    map_from = np.asarray(map_from[['x', 'y', 'z']])
    map_to = np.asarray(map_to[['x', 'y', 'z']])
    # Initialise plots
    fig = plt.figure()
    plot_data = [
        {'coords': map_from, 'color': 'green', 'alpha': 1, 'z_order': 2},
        {'coords': map_to, 'color': 'grey', 'alpha': 0.1, 'z_order': 1}
    ]
    if len(plane) == 0:
        all_coords = np.row_stack((map_from, map_to))
        for idx, d in enumerate(plot_data):
            ax = fig.add_subplot(1, len(plot_data), idx + 1, projection='3d')
            ax.scatter(d['coords'][:, 0], d['coords'][:, 1], d['coords'][:, 2], c=d['color'], alpha=0.1)
            # Set aspect ratio and viewing angle
            ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
            ax.set_box_aspect((np.ptp(all_coords[:, 0]), np.ptp(all_coords[:, 1]), np.ptp(all_coords[:, 2])))
    else:
        # Get axis of interest
        axes_list = 'xyz'
        for axis in plane:
            axes_list = axes_list.replace(axis, '')
        aoi = axis2idx(axes_list)
        ax = fig.add_subplot(1, 1, 1)
        for idx, d in enumerate(plot_data):
            coords = d['coords'][np.where(
                (d['coords'][:, aoi] >= -coi) &
                (d['coords'][:, aoi] <= coi)
            )]
            ax.scatter(coords[:, axis2idx(plane[0])], coords[:, axis2idx(plane[1])],
                       c=d['color'], alpha=d['alpha'], zorder=d['z_order'])
        ax.set_xlabel(plane[0]), ax.set_ylabel(plane[1])
        ax.set_aspect('equal')
    plt.show()


def axis2idx(axis):
    """ Returns equivalent index for text axis input (x, y, z). """

    dirs = ['x', 'y', 'z']
    return dirs.index(axis)
