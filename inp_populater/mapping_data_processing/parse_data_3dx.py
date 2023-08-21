import pandas as pd
import numpy as np
import tripy
from scipy.spatial.transform import Rotation as R


def kword_strip(line, identifiers):
    for target in identifiers:
        if target in line.lower():
            return str(target)


def get_bcoords_from_nodes(nodes, coords):
    bottle_coords = []
    for row in coords:
        node = int(row.split(',')[0])
        coords = []
        if node in nodes:
            continue
        else:
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


def element2xy(pts):
    """
    Takes coordinates of three element nodes (that make up a plane in 3d space)
    and rotates points to the xy plane for manipulation.
    """

    # Align origin with p1
    pts_t = pts - pts[0]
    # Compute the plane normal
    plane_normal = np.cross(pts_t[1], pts_t[2])
    # Compute the unit normal
    n = plane_normal / np.linalg.norm(plane_normal)
    if list(n) == [0, 0, 1]:
        # Check if element already lies on the xy plane
        pts_ry = pts_t
    else:  # Compute Rz (rotate plane to positive xz)
        e1 = n[0] / (n[0] ** 2 + n[1] ** 2) ** 0.5
        e2 = n[1] / (n[0] ** 2 + n[1] ** 2) ** 0.5
        R_z = np.array([[e1, e2, 0],
                        [-e2, e1, 0],
                        [0, 0, 1]])
        # Apply rotation
        r = R.from_matrix(R_z)
        pts_rz = r.apply(pts_t)
        # Compute Ry (align to positive xy plane)
        n_d = r.apply(n)
        R_y = np.array([[n_d[2], 0, -n_d[0]],
                        [0, 1, 0],
                        [n_d[0], 0, n_d[2]]])
        # Apply rotation
        r = R.from_matrix(R_y)
        pts_ry = r.apply(pts_rz)
    return pts_ry


def split_element(coords, ele_list):
    """
    Takes xy coordinates of element nodes (z should be zero), splits
    element (internally) into triangles and returns a list of indexes.
    """

    # Split element in xy space
    polygon = coords[:, :2]
    triangles = np.array(tripy.earclip(polygon))
    # Get indexes of triangle points
    tri_idxs = np.zeros((len(triangles), 3))
    for i, triangle in enumerate(triangles):
        for j, pt in enumerate(triangle):
            idx = np.flatnonzero((polygon == pt).all(1))[0]
            tri_idxs[i, j] = ele_list[idx]
    return tri_idxs


class InstanceThreeDx:

    # TODO investigate builder design creation pattern
    # https://refactoring.guru/design-patterns/builder

    def __init__(self, inp_file, coords_only=None):
        self.coords_only = coords_only if coords_only is not None else False
        self.inp_file = inp_file
        with open(self.inp_file, 'r') as f:
            self.lines = f.readlines()
        self.rigid_body_nodes = None
        self._get_kword_loc()
        self.coords, self.shape, self.nodes = self._get_bottle_coords()

        if not self.coords_only:
            self.connectivity = self._get_bottle_connectivity()
            self._split_connectivity()
            self._get_ntri_connectivity()

    def _get_kword_loc(self):
        kwords = {
            '*node': [],
            '*element,': [],
            '*nset, nset="rigid': []
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

        if not kwords.get('*nset, nset="rigid'):
            pass
        elif self.coords_only:
            pass
        else:
            self.rigid_body_nodes = True
            self._get_rigid_body_nodes()

    def _get_rigid_body_nodes(self):
        rigid_bodies_nodes = self.kword_locs.get('*nset, nset="rigid')

        rigid_bodies_begin = rigid_bodies_nodes[0]
        rigid_bodies_end = rigid_bodies_nodes[-1]
        rigid_bodies = get_section_from_lines(self.lines, self.all_kword_locs, rigid_bodies_begin, rigid_bodies_end)

        rigid_bodies_parsed = []
        for row in rigid_bodies:
            if '*nset' in str(row).lower():
                continue
            for val in row.split(','):
                rigid_bodies_parsed.append(int(val))
        self.rigid_body_nodes = rigid_bodies_parsed

    def _get_bottle_coords(self):
        coords_loc = self.kword_locs.get('*node')
        coords_begin = coords_loc[0]
        coords = get_section_from_lines(self.lines, self.all_kword_locs, coords_begin)

        if not self.rigid_body_nodes:
            bottle_coords = []
            for row in coords:
                coords = []
                for val in row.split(','):
                    coords.append(float(str(val).strip()))
                bottle_coords.append(coords)
            shape = len(bottle_coords[0][1:])
        else:
            bottle_coords, shape = get_bcoords_from_nodes(self.rigid_body_nodes, coords)
        bnodes = [i[0] for i in bottle_coords]
        if self.rigid_body_nodes:
            bnodes = [x for x in bnodes if x not in self.rigid_body_nodes]
            bottle_coords = [x for x in bottle_coords if x[0] not in self.rigid_body_nodes]
        return bottle_coords, shape, bnodes

    def _get_bottle_connectivity(self):
        element_loc = self.kword_locs.get('*element,')
        bnodes = self.nodes

        connectivity_begin = element_loc[0]
        connectivity_end = element_loc[-1]
        connectivity = get_section_from_lines(self.lines, self.all_kword_locs, connectivity_begin, connectivity_end,
                                              bottle=True)

        connectivity_parsed = []
        for row in connectivity:
            row_split = row.split(',')
            elements = row_split[1:]
            if '*' in row:
                continue
            elif any(int(x) not in bnodes for x in elements):
                continue
            else:
                connectivity_parsed.append([int(x) for x in row_split])
        return connectivity_parsed

    def _split_connectivity(self):
        connectivity = self.connectivity

        n_connectivity = {
            2: [],
            3: [],
            4: []
        }

        for row in connectivity:
            no_nodes = len(row) - 1
            nodes = row[1:]
            for n in n_connectivity:
                if n == no_nodes:
                    n_connectivity.get(n).append(nodes)

        self.dual_connectivity = n_connectivity.get(2)
        self.tri_connectivity = n_connectivity.get(3)
        self.quad_connectivity = n_connectivity.get(4)

    def _get_ntri_connectivity(self):
        self.coords_ntri = [[x[1], x[2], x[3]] for x in self.coords]
        coords = np.asarray(self.coords)

        count = 0
        split_connectivity = np.zeros((len(self.quad_connectivity) * 2, 3))
        for node_list in self.quad_connectivity:
            # Get nodal coordinates of element
            ele_coords = []
            for node_id in node_list:
                node_idx = np.where(coords[:, 0] == node_id)[0][0]
                # ele_coords.append(list(self.coords_ntri[node_id - 1]))
                ele_coords.append(list(coords[node_idx, 1:]))
            # Rotate/translate points to xy plane
            pts_rot = element2xy(np.array(ele_coords))
            # Split element and get indexes of triangles
            idxs = split_element(pts_rot, node_list)
            split_connectivity[count:count + 2, :] = idxs
            count += 2
        # Check if any triangle elements are already in the mesh
        if len(self.tri_connectivity) > 0:
            ntri_connectivity = np.vstack((self.tri_connectivity, split_connectivity))
        else:
            ntri_connectivity = split_connectivity

        self.ntri_connectivity = ntri_connectivity.astype(int)


def get_element_sth(b_nodes, connectivity_set, mapped_values):
    mapped_values_np = mapped_values.to_numpy()[:, 1:]
    avgs_all = []

    for elements in connectivity_set:
        node = elements[0]
        element_nodes = elements[1:]
        if any(x not in b_nodes for x in element_nodes):
            continue
        else:
            no_of_nodes = len(elements) - 1
            avgs = get_element_sth_return_avg(
                element_nodes, no_of_nodes, mapped_values_np)
            if any(x == 0 for x in avgs):
                continue
            else:
                avgs = np.concatenate(([node], avgs))
                avgs_all.append(avgs)

    avgs_all = np.asarray(avgs_all)
    columns_elemental = ['x', 'y', 'z', 'sth', 'modulus']

    mapped_values_elemental_sth = pd.DataFrame(avgs_all[:, 0])
    mapped_values_elemental_sth.columns = {'nodes'}

    for i in range(len(columns_elemental)):
        mapped_values_elemental_sth[columns_elemental[i]] = avgs_all[:, i + 1]
    return mapped_values_elemental_sth


def get_element_sth_return_avg(elements, divisor, mapped_values):
    totals = np.zeros(shape=5)
    for element in elements:
        idx = np.where(mapped_values[:, 0] == element)
        try:
            for i in range(len(totals)):
                totals[i] += mapped_values[idx, i + 1]
        except ValueError:
            pass
    totals_avg = totals / divisor
    return totals_avg
