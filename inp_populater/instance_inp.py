import numpy as np

from inp_populater.mapping_data_processing.map_coords import MapInputs


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

    # TODO investigate builder design creation pattern
    # https://refactoring.guru/design-patterns/builder

    def __init__(self, inp_file, inp_simulation, sbm_data):
        self.sbm_data = sbm_data
        self.inp_file = inp_file
        self.inp_simulation = inp_simulation
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

    def _get_ref_node(self):
        self.ref_node = int(float((self.lines[1]).strip()))

    def _get_mapped_coords(self):
        self.mapped_coords_mod, self.mapped_coords_sth = MapInputs(self.sbm_data, self.inp_file).map()
