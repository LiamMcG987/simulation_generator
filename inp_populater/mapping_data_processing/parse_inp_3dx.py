import pandas as pd


# Data extraction
def parse_nodes(inp_deck):
    """Locates target points in input deck, enabling co-ordinates to be found.

    Gathers nodes specific to bottle, gets co-ordinates from larger data set
    to form 3d profile of bottle.

    :param inp_deck: obtained from specified directory.
    :return: DataFrame with 3d profile.
    """
    with open(inp_deck, 'r') as fout:
        lines = fout.readlines()

    node_set_start = []
    n_sets = []
    coord_start = []
    element_lines = []
    star_lines = []
    for idx, line in enumerate(lines):
        if '*NSET, NSET="Geometric Group.bottle_all' in line:
            node_set_start.append(idx)
        if '*nset' in line.lower():
            n_sets.append(idx)
        if '*node' in line.lower():
            coord_start.append(idx + 1)
        if '*element, ' in line.lower():
            element_lines.append(idx)
        if '*' in line.lower():
            star_lines.append(idx)

    if 'bulge' in inp_deck.lower():
        return parse_nodes_bulge(lines, coord_start, n_sets)
    else:
        node_set_i = n_sets.index(node_set_start[0])
        node_set_begin_end = n_sets[node_set_i: node_set_i + 2]
        nodes = lines[node_set_begin_end[0] + 1: node_set_begin_end[1]]
        nodes_parsed = []
        for row in nodes:
            for val in row.split(','):
                nodes_parsed.append(int(val))

        coord_end = min(n_sets, key=lambda x: (x - coord_start[0]))
        coords = lines[coord_start[0]: coord_end]
        extract_connectivity(lines, element_lines, star_lines, nodes_parsed)
        return get_coords(nodes_parsed, coords)


def extract_connectivity(lines, elements, stars, bottle_nodes):
    element_end_idx = stars.index(elements[-1])
    connectivity_end = stars[element_end_idx + 1]
    connectivity = []
    i = 0
    for line in lines[elements[0] + 1:connectivity_end]:
        if 'element' in line.lower():
            continue
        node = int(line.split(',')[0])
        if node in bottle_nodes:
            connectivity.append([])
            for val in line.strip().split(', '):
                connectivity[i].append(int(val))
            i += 1

    return connectivity


def parse_nodes_bulge(input_deck_lines, coord_start_location, n_sets_location):
    """Specific to bulge cases, co-ordinates obtained through parsing of inp deck.

    :param input_deck_lines: as obtained previously.
    :param coord_start_location: as obtained previously.
    :param n_sets_location: as obtained previously.
    :return: DataFrame with 3d profile.
    """
    coord_end = min(n_sets_location, key=lambda x: (x - coord_start_location[0]))
    bottle_coords = []
    for line in input_deck_lines[coord_start_location[0]: coord_end]:
        bottle_coords.append(line)
    shape = len(bottle_coords[0].split(',')[1:])
    return bottle_coords, shape


def get_coords(nodes, coords):
    """Unpacks co-ordinates specific to bottle.

    :param nodes: points at which bottle exists.
    :param coords: full data set of co-ordinates (incl. foreign bodies).
    :return: DataFrame of 3d profile.
    """

    bottle_coords = []
    for row in coords:
        node = int(row.split(',')[0])
        if node in nodes:
            bottle_coords.append(row)
    shape = len(bottle_coords[0].split(',')[1:])
    return bottle_coords, shape


def extract_data(data):
    """From map_from file, headings and data parsed and categorised.

    Co-ordinates and field profiles obtained.

    :param data: obtained from check_directory().
    :return: DataFrame with 3d profile, in co-ordinates and fields.
    """
    df = pd.read_excel(data)
    cols = ['x', 'y', 'z', 'sth', 'mod']
    coords, profiles = [], []
    for col in df.columns:
        for col_target in cols:
            if col_target in col.lower():
                if col_target in cols[0:3]:
                    coords.append(df[col].to_list())
                else:
                    profiles.append(df[col].to_list())

    df = pd.DataFrame(coords[0])
    df.columns = {'x'}
    df['y'] = coords[1]
    df['z'] = coords[2]
    df['sth'] = profiles[0]
    df['mod'] = profiles[1]

    return df
