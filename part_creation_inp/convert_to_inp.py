import matplotlib.pyplot as plt
import numpy as np
import click

from create_inp import CreateInp


def _user_check():
    if click.confirm('Correct part?', default=True):
        click.echo('Extraction running.')
    else:
        raise SystemExit('Extraction process cancelled.')


def kword_strip(line, identifiers):
    for target in identifiers:
        if target in line.lower():
            return str(target)


def get_section_from_lines(lines, kwords, start, end=None):
    end = end if end is not None else None
    begin = kwords.index(start)
    if end is None:
        begin_end = kwords[begin: begin + 2]
    else:
        end = kwords.index(end)
        begin_end = kwords[begin: end + 2]
    section = lines[begin_end[0] + 1: begin_end[-1]]
    return section


def _visualise_part(coords, ref_node):
    x = [i[1] for i in coords]
    y = [i[2] for i in coords]
    z = [i[3] for i in coords]
    ref_node = [0, 0, 0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.scatter(ref_node[0], ref_node[1], ref_node[2], color='red', s=40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    plt.show()


def parse_values(data, ref_node=False, elset_dup=False):
    ref_node = ref_node if ref_node is not False else False
    elset_dup = elset_dup if elset_dup is not False else False
    ref_point = None

    parsed = []
    for row in data:
        if '*' in row:
            if elset_dup:
                break
            continue
        row_int = row.strip().split(',')
        for val in row_int:
            parsed.append(int(val))
    if ref_node:
        ref_point = parsed[0]
        parsed.pop(0)
    return parsed, ref_point


def get_bcoords_from_nodes(nodes, coords):
    bottle_coords = []
    for row in coords:
        node = int(row.split(',')[0])
        if node in nodes:
            bottle_coords.append([float(x) for x in row.split(',')])
        else:
            continue
    return bottle_coords


class ConvertToInp:

    def __init__(self, inp_file, part_kword):
        self.inp_file = inp_file
        with open(self.inp_file, 'r') as fout:
            self.lines = fout.readlines()
        self.part_kword = part_kword
        self.part_kword_identifier = f"rigid body{str(self.part_kword).lower()}"
        self._get_kword_loc()
        self._get_nodes()
        self._get_coords()
        _visualise_part(self.coords, self.reference_coords)
        _user_check()
        self._get_elements()
        self._reset_index()
        # breakpoint()
        self.create_part_inp()

    def _get_kword_loc(self):
        """
        Extracts relevant data from performance .inp, index based on parts in .inp.
        Fills kwords with location in .inp file where keyword has been identified.
        :return:
        """
        all_kwords = []
        kwords = {
            '*node': [],
            '*element,': [],
            f'*nset, nset="{self.part_kword_identifier}': [],
            f'*elset, elset="{self.part_kword_identifier}': []
        }

        for line_i, line in enumerate(self.lines):
            if '*' in line.lower():
                all_kwords.append(line_i)
            if any(x in line.lower() for x in kwords):
                item = kword_strip(line, kwords)
                kwords.get(item).append(line_i)

        self.kwords = kwords
        self.all_kwords = all_kwords

    def _get_nodes(self):
        nset_loc = self.kwords.get(f'*nset, nset="{self.part_kword_identifier}')
        nodes = get_section_from_lines(self.lines, self.all_kwords, nset_loc[0], end=nset_loc[-1])

        nodes_int, reference_node = parse_values(nodes, ref_node=True)
        self.nodes = nodes_int
        self.reference_node = reference_node

    def _align_ref_point_to_tip(self):
        """
        For parts with point/tip, reference node is shifted to tip.
        Enables easier translations - no need to account for thickness/width of part.
        """
        coords_np = np.asarray(self.coords)
        min_y = min(coords_np[:, 2])
        for i in range(len(coords_np)):
            coords_np[i][2] -= min_y
        self.coords = coords_np

    def _flip_in_y(self):
        coords_np = np.asarray(self.coords)
        for i in range(len(coords_np)):
            coords_np[i][2] = - coords_np[i][2]

        self.coords = coords_np

    def _get_coords(self):
        coord_loc = self.kwords.get('*node')
        coords_all = get_section_from_lines(self.lines, self.all_kwords, coord_loc[0])
        coords = get_bcoords_from_nodes(self.nodes, coords_all)
        self.coords = coords
        self.coords_all = []
        for row in coords_all:
            self.coords_all.append([float(x) for x in row.split(',')])
        # self._remove_duplicates()         # uncomment for side plate
        self._reset_coords()
        # self._align_ref_point_to_tip()    # uncomment for hand
        self._flip_in_y()                   # uncomment for finger
        self._get_reference_coords()

    def _get_reference_coords(self):
        coords_np = np.asarray(self.coords_all)
        ref_idx = np.where(coords_np[:, 0] == self.reference_node)
        reference_coords = coords_np[ref_idx]
        self.reference_coords = reference_coords

    def _reset_coords(self):
        """
        Resets coordinates to have origin on reference node.
        Gets distance from reference node to origin.
        Subtracts xyz distance from each coordinate.

        Last coordinate set to reference point (origin). Used to identify reference node for populater.
        """
        coords_np = np.asarray(self.coords)
        coords_all_np = np.asarray(self.coords_all)
        ref_coords_idx = np.where(coords_all_np[:, 0] == self.reference_node)
        ref_coords = coords_all_np[ref_coords_idx]
        dist_to_origin = []
        for coord in ref_coords[0, 1:]:
            dist_to_origin.append(abs(0 - coord))
        for i, coord_row in enumerate(coords_np):
            for j, coord in enumerate(coord_row[1:]):
                if coord == coord_row[-1] or coord == coord_row[-2]:  # change to x or y for parts
                    coords_np[i][j + 1] = abs(coord) - dist_to_origin[j]
                else:
                    coords_np[i][j + 1] = coord - dist_to_origin[j]
        self.coords = coords_np

    def _remove_duplicates(self):
        """
        Used in case of duplicate parts (e.g., side load).
        Removes part coordinates from one side (if x > 0, remove).
        """
        coords_no_dups = []
        nodes_no_dups = []

        for coord_row in self.coords:
            node = coord_row[0]
            x = coord_row[1]
            if x > 0:
                continue
            else:
                coords_no_dups.append(coord_row)
                nodes_no_dups.append(node)

        self.nodes = [x for x in self.nodes if x in nodes_no_dups]
        self.coords = coords_no_dups

    def _get_elements(self):
        """
        Extracts whole connectivity from .inp file.
        Only extracts connectivity within elset for part.
        """
        elements_loc = self.kwords.get('*element,')
        elements = get_section_from_lines(self.lines, self.all_kwords, elements_loc[0], end=elements_loc[-1])
        elset_loc = self.kwords.get(f'*elset, elset="{self.part_kword_identifier}')
        elset = get_section_from_lines(self.lines, self.all_kwords, elset_loc[0], end=elset_loc[-1])

        elset_elements, reference_element = parse_values(elset, elset_dup=True)
        self.elset = elset_elements

        elements_parsed = []
        for row in elements:
            if '*' in row:
                continue
            element_number = row.strip().split(',')[0]
            if int(element_number) not in elset_elements:
                continue
            elements_parsed.append([int(x) for x in row.strip().split(',')])
        self.elements = elements_parsed

    def _reset_index(self):
        """
        Resets coordinates, nsets and elsets to 1.
        Parts with S3R and S4 require extra step (lengths used to check for both element types).
        If both element types are in connectivity, min_element_number resets S4 to 0, therefore
        min_element_number - 1 used to reset to 1.
        Reference node updated with new index at end, becomes equal to last index + 1.
        """

        min_node = min(self.nodes) - 1
        min_elset = min(self.elset) - 1
        min_element_number = 100000
        lengths = []

        for row in self.elements:
            element_length = len(row[1:])
            lengths.append(element_length)
            for element in row[1:]:
                if element < min_element_number:
                    min_element_number = element - 1

        for val_i, val in enumerate(self.coords):
            self.coords[val_i][0] = val[0] - min_node

        for row_i, row in enumerate(self.elements):
            self.elements[row_i][0] = row[0] - min_elset
            for val_i, val in enumerate(row[1:]):
                if 3 and 4 in lengths:
                # if element_type == 4:
                    self.elements[row_i][val_i + 1] = val - min_element_number + 1
                # elif element_type == 5:
                else:
                    self.elements[row_i][val_i + 1] = val - min_element_number

        for val_i, val in enumerate(self.elset):
            self.elset[val_i] = val - min_elset

        for val_i, val in enumerate(self.nodes):
            self.nodes[val_i] = val - min_node

        self.ref_node = self.coords[-1][0] + 1

    def create_part_inp(self):
        CreateInp(self.ref_node, self.coords, self.nodes, self.elements, self.elset, self.part_kword).write_to_inp()


if __name__ == '__main__':
    file = 'Reckitt_Mowgli_750ml_34g_RIBS_Squeeze.inp'
    kword = 'finger'
    ConvertToInp(file, kword)
