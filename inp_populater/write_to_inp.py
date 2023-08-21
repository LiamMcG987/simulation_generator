import shutil
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def kword_strip(line, identifiers):
    for target in identifiers:
        if target in line:
            return str(target)


def get_write_out_function(write_line_index, kwords_dict):
    func = False
    for kword_value in kwords_dict.values():
        try:
            indexer = kword_value[1]
        except IndexError:
            continue
        if write_line_index == indexer:
            func = kword_value[0]
    return func


def _visualise_assembly(coords):
    coords_np = np.asarray(coords)
    x = [i[0] for i in coords_np]
    y = [i[1] for i in coords_np]
    z = [i[2] for i in coords_np]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    plt.tight_layout()
    plt.show()


def _write_set(fout, data, surface=False, rigid_body=False, bulge=False):
    surface = surface if surface is not False else False
    rigid_body = rigid_body if rigid_body is not False else False
    bulge = bulge if bulge is not False else False

    if bulge:
        for val in data:
            fout.write(f'{int(val)}, SPOS\n')
    elif rigid_body:
        fout.write(f'{data}')
    elif surface:
        for val in data:
            fout.write(f'{int(val)}\n')
    elif isinstance(data, list):
        for idx, val in enumerate(data):
            if (idx + 1) % 16 == 0 or val == data[-1]:
                fout.write(f'{int(val)}\n')
            else:
                fout.write(f'{int(val)}, ')
    else:
        fout.write(f'{int(data)}\n')


class WriteToInp:
    """
    Following sets are self-contained:

    self.coords_all = all coordinates
    self.nsets = nsets for all parts
    self.elsets = elsets for all parts
    self.elements_s3/4 = all elements for all parts
    self.element_numbers_s3/4 = all element numbers for all parts

    parts_dict.get('bottle').neck/body = neck and body elsets
    """

    def __init__(self, parts_dict, template_file, simulation_file, translation_matrix):
        self.parts_dict = parts_dict
        self.template_file = template_file
        self.simulation_file = simulation_file
        self.translation_matrix = translation_matrix
        self._get_template_locs()
        self._translate_coords()
        self._append_coords()
        self._append_connectivity()
        # _visualise_assembly(self.coords_all)
        self.write()
        if 'bulge' in self.simulation_file:
            self._elements_check()

    def _get_template_locs(self):
        self.kwords = {
            '*ELSET, ELSET="Spatial Groupbody-1"': [self._write_elset_body],
            '*ELSET, ELSET="Spatial Groupneck-1"': [self._write_elset_neck],
            '*HEADING': [self._write_heading],
            '*NODE': [self._write_nodes],
            '*ELEMENT, TYPE=S3R': [self._write_elements_s3r],
            '*ELEMENT, TYPE=S4': [self._write_elements_s4],
            '*DISTRIBUTION, LOCATION=ELEMENT, NAME="Spatial Groupbody-1", TABLE="Spatial Groupbody-1"':
                [self._write_thickness],
            '*ELASTIC, TYPE=ISOTROPIC': [self._write_modulus],
            '*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=1': [self._write_initial_conds]
        }

        with open(self.template_file, 'r') as f:
            self.lines = f.readlines()

        for line_i, line in enumerate(self.lines):
            if any(x in line for x in self.kwords):
                matched_item = kword_strip(line, self.kwords)
                if matched_item.startswith('*RIGID BODY'):
                    self.kwords.get(matched_item).append(line_i)
                elif matched_item.startswith('*ELASTIC') and 'pet' not in self.lines[line_i - 1].lower():
                    continue
                else:
                    self.kwords.get(matched_item).append(line_i + 1)

    def _append_coords(self):
        """
        Coords appended to each other - index counts in order.
        Nsets ordered - labelled for each part.
        """
        coords_all = []
        nodes_all = []
        nsets = {}
        node_count = 0
        for i, item in enumerate(self.parts_dict):
            coords = np.asarray(self.parts_dict.get(item).coords)
            for row in coords:
                nodes_all.append(row[0] + node_count)
                coords_all.append([float(x) for x in row[1:]])
            nsets[item] = [i[0] + node_count for i in coords]
            node_count += len(coords)

        self.coords_all = coords_all
        self.nsets = nsets
        node_length = len(coords_all)

        i = 0
        for item in self.parts_dict:
            if 'rigid' not in item:
                pass
            else:
                i += 1
                self.parts_dict.get(item).ref_node = node_length + i

    def _append_connectivity(self):
        """
        Elsets ordered - labelled for each part also.
        Elements appended - extra layer required to update S3R and S4.
        """

        element_numbers_s3r_all = []
        element_numbers_s4_all = []
        nodes_s3r_all = []
        nodes_s4_all = []
        elsets = {}

        element_number_count = 0
        connectivity_count = 0

        for item in self.parts_dict:
            connectivity = self.parts_dict.get(item).connectivity
            s3_appended = 0
            s4_appended = 0
            element_numbers = []
            nodes = []

            for i, row in enumerate(connectivity):
                element_numbers.append(row[0])
                for val in row[1:]:
                    nodes.append(val)
                if len(row) == 4:
                    element_numbers_s3r_all.append(row[0] + element_number_count)
                    nodes_s3r_all.append([int(x + connectivity_count) for x in row])
                    s3_appended += 1
                else:
                    element_numbers_s4_all.append(row[0] + element_number_count)
                    nodes_s4_all.append([int(x + connectivity_count) for x in row])
                    s4_appended += 1

            elsets[item] = [x[0] + element_number_count for x in connectivity]
            element_number_count += s3_appended + s4_appended
            connectivity_count += np.amax(nodes)

        self.elsets = elsets
        self.elements_s4 = nodes_s4_all
        self.element_numbers_s4 = element_numbers_s4_all

        self.elements_s3 = nodes_s3r_all
        self.element_numbers_s3 = element_numbers_s3r_all

    def _translate_coords(self):
        translation_matrix_np = np.zeros(shape=(2, 3))

        for i, translation in enumerate(self.translation_matrix):
            for j, x in enumerate(translation):
                translation_matrix_np[i, j] = x

        parts_for_translation = []
        for part in self.parts_dict:
            if 'rigid' in part:
                parts_for_translation.append(part)

        if not parts_for_translation:
            return

        self.translation_coords = []
        for j, part in enumerate(parts_for_translation):
            self.translation_coords.append(translation_matrix_np[j])
            coords = self.parts_dict.get(part).coords
            for row_i, row in enumerate(coords):
                for i in range(len(row[1:])):
                    coords[row_i][i + 1] += translation_matrix_np[j][i]
            self.parts_dict.get(part).coords = coords

    def write(self):
        os.chdir('../../simulation_inps/')

        # os.chdir('C:\\Users\\LiamMcGovern\\Documents\\Software\\5000\\3dx_performance\\simulation_inps')
        shutil.copy(self.template_file, self.simulation_file)

        rigid_body_lines = 0
        with open(self.simulation_file, 'w') as f:
            for line_i, line in enumerate(self.lines):
                if line.lower().startswith('*rigid body'):
                    rigid_body_lines += 1
                    data_to_write, surface, rigid_body, bulge = self._extract_kwords_from_line(line)
                elif rigid_body_lines == 2 and self.lines[line_i - 1].lower().startswith('*rigid body'):
                    data_to_write = []
                else:
                    data_to_write, surface, rigid_body, bulge = self._extract_kwords_from_line(self.lines[line_i - 1])

                if not data_to_write:
                    func = get_write_out_function(line_i, self.kwords)
                    if not func:
                        f.write(line)
                    else:
                        func(f)
                else:
                    _write_set(f, data_to_write, surface, rigid_body, bulge)

    def _extract_kwords_from_line(self, line):
        parts_kwords = [
            'bottle',
            'top_plate', 'ground_plate',
            'finger', 'hand',
            'side_1', 'side_2'
        ]

        sets_kwords = {
            '*nset': self.nsets,
            '*elset': self.elsets,
            '*surface': self.elsets,
            '*rigid body': self.parts_dict
        }

        surface, rigid_body, bulge = False, False, False
        data_set = []
        for set_kword in sets_kwords:
            if line.lower().startswith(set_kword):
                if set_kword == '*surface':
                    surface = True
                    if 'bulge' in line.lower():
                        bulge = True
                elif set_kword == '*rigid body':
                    rigid_body = True
                for part in parts_kwords:
                    if part in line.lower():
                        identifier = f'rigid_body_{part}' if part != 'bottle' else 'bottle'
                        data_set = sets_kwords.get(set_kword).get(identifier)
                        if 'referencenode' in line.lower():
                            data_set = self.parts_dict.get(identifier).ref_node
                        elif rigid_body:
                            reference_node = str(self.parts_dict.get(identifier).ref_node)
                            data_set = line.replace('~~', reference_node)
                        elif 'basering' in line.lower():
                            data_set = self.parts_dict.get(identifier).base_ring
        return data_set, surface, rigid_body, bulge

    def _elements_check(self):
        with open(self.simulation_file, 'r') as fout:
            lines = fout.readlines()

        empty_headings = []
        for line_i, line in enumerate(lines):
            if '~~' in line:
                empty_headings.append(line_i)
                empty_headings.append(line_i - 1)

        with open(self.simulation_file, 'w') as fout:
            for line_i, line in enumerate(lines):
                if line_i in empty_headings:
                    continue
                else:
                    fout.write(line)

    def _write_heading(self, fout):
        simulation_name = os.path.splitext(self.simulation_file)[0]
        now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        fout.write(f'Simulation {simulation_name}, Created {now} in version 3DEXPERIENCE R2023x\n')

    def _write_nodes(self, fout):
        coords = self.coords_all
        for row_i, row in enumerate(coords):
            fout.write(f'{int(row_i + 1)}, {row[0]}, {row[1]}, {row[2]}\n')

        if 'bulge' in str(self.simulation_file).lower():
            pass
        else:
            trans = self.translation_coords
            for i, tran in enumerate(trans):
                fout.write(f'{int(len(coords) + 1 + i)}, {tran[0]}, {tran[1]}, {tran[2]}\n')

    def _write_elements_s3r(self, fout):
        elements = self.elements_s3
        element_numbers = self.element_numbers_s3
        for i in range(len(elements)):
            fout.write(
                f'{int(element_numbers[i])}, {int(elements[i][1])}, {int(elements[i][2])}, {int(elements[i][3])}\n')

    def _write_elements_s4(self, fout):
        elements = self.elements_s4
        element_numbers = self.element_numbers_s4
        for i in range(len(elements)):
            fout.write(
                f'{int(element_numbers[i])}, {int(elements[i][1])}, {int(elements[i][2])}, '
                f'{int(elements[i][3])}, {int(elements[i][4])}\n')

    def _write_elset_body(self, fout):
        body_elset = self.parts_dict.get('bottle').body
        _write_set(fout, body_elset)

    def _write_elset_neck(self, fout):
        neck_elset = self.parts_dict.get('bottle').neck
        _write_set(fout, neck_elset)

    def _write_thickness(self, fout):
        sth = self.parts_dict.get('bottle').mapped_coords_sth
        fout.write(', 0.3\n')
        for i, number in enumerate(sth['sth']):
            fout.write(f'{int(sth["nodes"][i])}, {round(sth["sth"][i], 3)}\n')

    def _write_modulus(self, fout):
        mod = self.parts_dict.get('bottle').mapped_coords_mod['modulus']
        for i, val in enumerate(mod):
            fout.write(f'{round(val, 1)}, 0.32,    , {i + 1}\n')
        fout.write('''*PLASTIC, HARDENING=ISOTROPIC
90., 0.
101.4984602, 0.0275
116.7347118, 0.07597
130.2572811, 0.1199
151.8065992, 0.17745
170.0140783, 0.2322
259.0059833, 0.5\n''')

    def _write_initial_conds(self, fout):
        modulus = self.parts_dict.get('bottle').mapped_coords_mod['modulus']
        for i in range(len(modulus)):
            fout.write(f'{int(i + 1)}, {int(i + 1)}\n')
        fout.write('*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=2\n')
        for i, number in enumerate(modulus):
            fout.write(f'{int(i + 1)}, {round(number, 3)}\n')
