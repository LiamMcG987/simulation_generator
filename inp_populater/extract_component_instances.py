import os
import glob
import sys

import numpy as np

from inp_populater.mapping_data_processing.map_coords import MapInputs


def write_to_translation_file(file, values):
    print(values)
    with open(file, 'w') as f:
        f.write(f"""
Translations

Simulation: 
part: translation matrix (x, y, z)

top_load
rigid_body_top_plate: 0, 0, {round(values[4], 2)}
rigid_body_ground_plate: 0, 0, {round(values[5], 2)}

squeeze
rigid_body_finger: 0, {round(values[2], 2)}, 0
rigid_body_hand: 0, {round(values[3], 2)}, 0

side_load (move in opposite directions)
rigid_body_side: {round(values[0], 2)}, 0, 0
rigid_body_side: {round(values[1], 2)}, 0, 0""")


def kword_strip(line, identifiers):
    for target in identifiers:
        if target in line.lower():
            return str(target)


def get_bcoords_from_nodes(coords):
    bottle_coords = []
    for row in coords:
        coords = []
        for val in row.split(','):
            coords.append(float(str(val).strip()))
        bottle_coords.append(coords)
    shape = len(bottle_coords[0][1:])
    return bottle_coords, shape


def get_section_from_lines(lines, kwords, start, end=None, bottle=None):
    end = end if end is not None else None
    bottle = bottle if bottle is not None else False
    begin = kwords.index(start)
    if bottle:
        begin_end = kwords[begin]
        section = lines[begin_end + 1:]
        return section
    elif end is None:
        begin_end = kwords[begin: begin + 2]
    else:
        end = kwords.index(end)
        begin_end = kwords[begin: end + 2]
    section = lines[begin_end[0] + 1: begin_end[-1]]
    return section


class InstanceInp:

    def __init__(self, inp_file, inp_simulation, sbm_data, bottle_mass):
        self.sbm_data = sbm_data
        self.inp_file = inp_file
        self.inp_simulation = inp_simulation
        self.bottle_mass = bottle_mass
        with open(self.inp_file, 'r') as f:
            self.lines = f.readlines()
        self._get_kword_loc()

    def _get_kword_loc(self):
        """
        Grabs data from part library for relevant part.
        Part library format set up locally, can be updated if needed - shouldn't be necessary.
        """
        kwords = {
            '*node': [],
            '*element': [],
            '*nset': [],
            '*elset': []
        }
        all_kwords = []

        for line_i, line in enumerate(self.lines):
            if '*' in line.lower():
                all_kwords.append(line_i)
            if any(x in line.lower() for x in kwords):
                matched_item = kword_strip(line, kwords)
                kwords.get(matched_item).append(line_i)

        self.kword_locs = kwords
        self.all_kword_locs = all_kwords

        if not self.kword_locs.get('*nset'):
            self.bottle_file = True
            self._get_coords()
            self._get_connectivity(bottle=self.bottle_file)
            self._get_body_groups()
            self._get_base_ring()
            self._get_mapped_coords()
            return self
        else:
            self.bottle_file = False
            self._get_coords()
            self._get_connectivity()
            self._get_ref_node()
            return self

    def _get_coords(self):
        coords_loc = self.kword_locs.get('*node')
        coords_begin = coords_loc[0]
        coords_str = get_section_from_lines(self.lines, self.all_kword_locs, coords_begin)
        coords, shape = get_bcoords_from_nodes(coords_str)
        shape = len(coords[0][1:])
        nodes = [i[0] for i in coords]
        self.coords = coords
        self.shape = shape
        self.nodes = nodes

    def _get_connectivity(self, bottle=False):
        element_loc = self.kword_locs.get('*element')
        connectivity_begin = element_loc[0]
        connectivity_end = element_loc[-1]
        connectivity = get_section_from_lines(self.lines, self.all_kword_locs, connectivity_begin, connectivity_end,
                                              bottle)

        connectivity_parsed = []
        for row in connectivity:
            row_split = row.split(',')
            if '*' in row:
                continue
            else:
                connectivity_parsed.append([int(x) for x in row_split])
        self.connectivity = connectivity_parsed
        self.elset = [i[0] for i in self.connectivity]

    def _get_body_groups(self):
        """
        Separates body group from neck group.
        For each element row, if the average of z-coordinate is above 0 -> neck, else, body.
        """
        coords_np = np.asarray(self.coords)
        neck = []
        body = []
        for element in self.connectivity:
            el_number = element[0]
            nodes = element[1:]
            divisor = len(nodes)
            z = 0
            for node in nodes:
                idx = np.where(coords_np[:, 0] == node)[0]
                try:
                    z += coords_np[idx[0]][-1]
                except IndexError:
                    continue
            z_avg = z/divisor
            if z_avg >= 0:
                neck.append(el_number)
            else:
                body.append(el_number)
        self.neck = neck
        self.body = body
        elset_all = neck + body
        self.elset_all = elset_all

    def _get_base_ring(self):
        coords_np = np.asarray(self.coords)
        min_z = min(coords_np[:, -1])
        min_z_tol = min_z * 0.9999

        base_ring = []
        for coord in coords_np:
            node = coord[0]
            z = coord[-1]
            if z <= min_z_tol:
                base_ring.append(node)

        self.base_ring = base_ring

    def _update_translation_matrix(self):
        coords = np.asarray(self.coords)

        x_max = max(coords[:, 1])
        x_min = min(coords[:, 1])
        y_min = min(coords[:, 2])
        y_max = max(coords[:, 2])
        z_max = max(coords[:, 3])
        z_min = min(coords[:, 3])

        translations = [
            x_max, x_min,
            y_max, y_min,
            z_max, z_min
        ]

        translation_matrix = '../inp_populater/translation_matrix.txt'
        write_to_translation_file(translation_matrix, translations)

    def _get_ref_node(self):
        self.ref_node = int(float((self.lines[1]).strip()))

    def _get_mapped_coords(self):
        sbm_files_dir = '../inp_populater/mapping_data_inputs/*'
        sbm_data = None

        for sbm_file in glob.glob(sbm_files_dir):
            sbm_file_name = os.path.splitext(os.path.split(sbm_file)[-1])[0]
            if sbm_file_name == self.sbm_data:
                sbm_data = sbm_file
                break

        if not sbm_data:
            sys.exit('No SBM data found. Process aborted.')

        self.mapped_coords_mod, self.mapped_coords_sth = MapInputs(sbm_data, self.inp_file, self.bottle_mass).map()
