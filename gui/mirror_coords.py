import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def flip_coords(sbm_data, mirror_col, visualise_only=None):
    visualise_only = visualise_only if visualise_only is not None else False

    file_ext = os.path.splitext(sbm_data)[-1]
    if file_ext == '.csv':
        data = pd.DataFrame(pd.read_csv(sbm_data))
    elif file_ext == '.xlsx':
        data = pd.DataFrame(pd.read_excel(sbm_data))
    else:
        sys.exit('File not recognised. Ensure .xlsx or .csv file is used.')

    mirror_col_pd_idx, mirror_col_pd = None, None
    for i, col in enumerate(data.columns):
        if mirror_col in col.lower():
            mirror_col_pd_idx = i
            mirror_col_pd = col
            break

    if not mirror_col_pd or not mirror_col_pd_idx:
        sys.exit('Unable to locate column to mirror. Check data provided and selected column.')

    updated_data = copy_and_append_columns(data, mirror_col)

    if not visualise_only:
        return updated_data
    else:
        visualise_df(updated_data)


def copy_and_append_columns(data, mirror_col_name):
    shape = np.shape(data)
    data_new_np = np.zeros(shape=(shape[1], shape[0]*2))

    for i, col in enumerate(data.columns):
        copy = data[col].to_numpy()
        if mirror_col_name == col:
            copy = copy * -1
        elif 'nodes' in col.lower():
            copy = [x + 1 for x in range(len(copy) * 2)]
            data_new_np[i] = copy
            continue

        original_data = np.array(data[col])
        updated_data = np.append(original_data, copy)
        data_new_np[i] = updated_data

    data_new_copied = pd.DataFrame(data_new_np).T
    cols = ['x', 'z', 'y', 'sth', 'modulus', 'modulus_before_scale', 'nodes']
    data_new_copied.columns = cols
    return data_new_copied


def visualise_df(data):
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    z = data['z'].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_aspect(aspect='equal')
    fig.tight_layout()
    plt.show()


def copy_and_save(updated_sbm):
    os.chdir('../inp_populater/mapping_data_inputs/')
    file_name = 'check.xlsx'
    updated_sbm.to_excel(file_name, index=False)


def mirror_coords(sbm_data, mirror_col, visualise_only=None):
    visualise_only = visualise_only if visualise_only is not None else True
    if visualise_only:
        flip_coords(sbm_data, mirror_col, visualise_only=visualise_only)
    else:
        updated_sbm = flip_coords(sbm_data, mirror_col)
        copy_and_save(updated_sbm)


# if __name__ == '__main__':
#     sbm_file = '../mowgli_750_34g_D.xlsx'
#     neg_column = 'z'
#     visualise = False
#     mirror_coords(sbm_file, neg_column, visualise_only=visualise)
