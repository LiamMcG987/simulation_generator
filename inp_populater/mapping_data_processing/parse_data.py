import os.path
import numpy as np
import pandas as pd

from .check_data import check_data_inp
from .check_data import check_data_matlab
from .check_data import check_data_csv


def extract_data_inp(inp_deck):
    """From inp file, coords and dimensionality extracted. Assumed to have no sth profile"""
    coords, shape, connectivity, nset = check_data_inp(inp_deck)
    nodes = [i[0] for i in coords]
    x = [i[1] for i in coords]
    y = [i[2] for i in coords]
    if shape == 3:
        z = [i[3] for i in coords]
    else:
        z = np.zeros(len(nodes))
    coords_np = np.asarray([nodes, x, y, z])
    columns = ['nodes', 'x', 'y', 'z']

    df = pd.DataFrame(coords_np).transpose()
    df.columns = columns
    return df, shape, connectivity


def extract_data_csv(csv_file):
    """From csv, file examined to check for xyz and sth values"""
    ext = os.path.splitext(csv_file)[1]
    return check_data_csv(csv_file, ext)[0:2]


def extract_data_matlab(mat_file):
    """From mat file, assumed to be axisymmetric with sth profile"""
    nodes, x, y, sth = check_data_matlab(mat_file)
    df = pd.DataFrame(nodes)
    df.columns = {'nodes'}
    df['x'] = x
    df['y'] = y
    df['z'] = np.zeros(len(df['x']))
    df['sth'] = sth
    return df


def append_profile(df, variable_map):
    """Adds sth/mod profile onto df map"""
    file_ext = os.path.splitext(variable_map)[-1]
    if file_ext == '.csv':
        map_variable_data = pd.read_csv(variable_map)
    elif file_ext == '.xlsx':
        map_variable_data = pd.read_excel(variable_map)
    else:
        raise TypeError('{} is in incorrect format. Try .csv or .xlsx instead.'
                        .format(os.path.splitext(variable_map)))
    for idx, column in enumerate(map_variable_data.columns):
        if 'thickness' in column.lower():
            sth_index = idx
            return append_profile_check_len(sth_index, map_variable_data, df, 'sth')
        elif 'modulus' in column.lower():
            mod_index = idx
            return append_profile_check_len(mod_index, map_variable_data, df, 'modulus')
        else:
            raise ValueError('Error in appending additional profile.')


def append_profile_check_len(index, map_profile, df, heading):
    if len(df) != len(map_profile.iloc[:, index]):
        raise IndexError('Shapes of dimensions and additional profile do not align.')
    else:
        df[heading] = map_profile.iloc[0:, index]
    return df


def extract_data(input_file):
    """Function handler - directs files depending on ext"""
    file_ext = os.path.splitext(input_file)[-1]
    if file_ext == '.inp':
        return extract_data_inp(input_file)
    elif file_ext in ['.csv', '.xlsx']:
        return extract_data_csv(input_file)
    elif file_ext == '.mat':
        return extract_data_matlab(input_file)
    else:
        print('File extension, {}, not recognised - try .inp, .csv, or .mat.'.format(file_ext))
