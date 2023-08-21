import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import tripy


# Classes

class Instance:
    """ Class to return key data for a given instance. """

    def __init__(self, instance_type, idx_pair, lines):
        self.instance_type = instance_type
        self.start_idx = idx_pair[0]
        self.lines = lines
        self.coords, self.height, self.neck_height = self.extract_coords(lines)
        self.dual_connectivity, self.tri_connectivity, self.quad_connectivity = self.extract_connectivity(lines)
        self.ntri_connectivity = self.split_quadratic_mesh(
            self.coords, self.tri_connectivity, self.quad_connectivity)

    def get_start_coord_idx(self, lines):
        """ Returns starting index for coordinate list. """

        # if '*Node' not in lines[self.start_idx + 1]:
        # 	if '*Node' not in lines[self.start_idx + 2]:
        # 		coord_start = 4
        # 	else:
        # 		coord_start = 3
        # else:
        coord_start = 2
        return coord_start

    def parse_data(self, data_in, drop_index=True):
        """ Extracts element arrays and convert list of strings to list of lists. """

        # Initialise output data array
        no_cols = len(data_in[0].split(sep=','))
        data_out = np.zeros(shape=(len(data_in), no_cols))
        # Populate output data array
        for idx, val in enumerate(data_in):
            row = (val.split(sep=','))
            for jdx in range(np.shape(data_out)[1]):
                data_out[idx, jdx] = row[jdx]
        if drop_index is True:
            data_out = data_out[:, 1:]
        return data_out

    def extract_coords(self, lines):
        """ Extracts coordinates for all nodes and tip (assumes bottle height is longest direction). """

        # Extract nodal coordinates
        coord_start = self.get_start_coord_idx(lines)
        coord_end = [idx for idx, line in enumerate(lines) if '*element' in line.lower()]
        instance_coords = np.asarray(lines[coord_start:coord_end[0]])
        coords = self.parse_data(instance_coords)
        # Get bottle axis
        max_idx = np.unravel_index(np.abs(coords).argmax(), np.abs(coords).shape)
        axis_idx = max_idx[1]
        # Extract bottle (midplane) height
        height = np.max(np.abs(coords[:, axis_idx]))
        # Extract neck height
        neck_height = np.max(coords[:, axis_idx])
        return coords, height, neck_height

    def extract_connectivity(self, lines):
        """ Extracts connectivity for all element types in the mesh. """

        # Extract element connectivity indexes
        start_idxs = [[idx for idx, line in enumerate(lines) if '*element, type=' in line.lower()][1]]
        end_idx = [[i for i, s in enumerate(lines) if '*node' in s.lower()][1]]
        idx_list = start_idxs + end_idx
        idx_pairs = []
        for i, idx in enumerate(idx_list):
            if i == 0:
                idx_pairs.append(idx)
            elif i == len(idx_list) - 1:
                idx_pairs.append(idx)
            else:
                idx_pairs.extend([idx, idx])
        idx_pairs = list(zip(*(iter(idx_pairs),) * 2))
        # Extract connectivity
        dual_connectivity = []
        tri_connectivity = []
        quad_connectivity = []
        for idx_pair in idx_pairs:
            connectivity = np.array(lines[idx_pair[0] + 1:idx_pair[1]])
            connectivity = self.parse_data(connectivity) - 1  # Account for indexing
            if np.shape(connectivity)[1] == 2:
                dual_connectivity.extend(connectivity)
            if np.shape(connectivity)[1] == 3:
                tri_connectivity.extend(connectivity)
            elif np.shape(connectivity)[1] == 4:
                quad_connectivity.extend(connectivity)
        return np.array(dual_connectivity, dtype=int), \
               np.array(tri_connectivity, dtype=int), \
               np.array(quad_connectivity, dtype=int)

    def split_quadratic_mesh(self, coords, tri_connectivity, quad_connectivity):
        """ Function to split quadratic mesh into ntri mesh for mass calculation. """

        count = 0
        split_connectivity = np.zeros((len(quad_connectivity) * 2, 3))
        for node_list in quad_connectivity:
            # Get nodal coordinates of element
            ele_coords = []
            for node_id in node_list:
                ele_coords.append(list(coords[node_id]))
            # Rotate/translate points to xy plane
            pts_rot = self.element2xy(np.array(ele_coords))
            # Split element and get indexes of triangles
            idxs = self.split_element(pts_rot, node_list)
            split_connectivity[count:count + 2, :] = idxs
            count += 2
        # Check if any triangle elements are already in the mesh
        if len(tri_connectivity) > 0:
            ntri_connectivity = np.vstack((tri_connectivity, split_connectivity))
        else:
            ntri_connectivity = split_connectivity
        return ntri_connectivity.astype(int)

    def element2xy(self, pts, plot_data=False):
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
        if plot_data == True:
            # Plot element translation
            data = [pts_t, pts_rz, pts_ry]
            self.check_elements(data)
        return pts_ry

    def split_element(self, coords, ele_list, plot_data=False):
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
        if plot_data == True:
            # Plot element split
            self.check_triangles(triangles)
        return tri_idxs

    def check_elements(self, data_pts):
        """ Plots element translations/rotations to check algorithm. """

        fig = plt.figure()
        ax = Axes3D(fig)
        colours = ['black', 'lightgreen', 'green']
        for idx, data in enumerate(data_pts):
            data = np.vstack((data, data[0]))
            ax.plot(data[:, 0], data[:, 1], data[:, 2], c=colours[idx])
        ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
        plt.show()

    def check_triangles(self, coords):
        """ Plots split triangles to check algorithm. """

        f, ax = plt.subplots()
        colours = ['blue', 'red']
        for idx, triangle in enumerate(coords):
            triangle = np.vstack((triangle, triangle[0]))
            ax.plot(triangle[:, 0], triangle[:, 1], c=colours[idx])
        ax.plot()
        ax.set_xlabel('x'), ax.set_ylabel('y')
        plt.show()
