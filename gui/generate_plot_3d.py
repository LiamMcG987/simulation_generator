import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from inp_populater.mapping_data_processing.parse_data_3dx import InstanceThreeDx


def translate_coords(coordinates, translation_matrix):
    coords_orig = np.asarray(coordinates)[:, 1:]
    coords_translated = np.zeros(shape=np.shape(coords_orig))

    for i in range(len(coords_orig)):
        coordinate = coords_orig[i]
        for j in range(len(coordinate)):
            coords_translated[i][j] = coords_orig[i][j] + float(translation_matrix[j])
    return coords_translated


class GeneratePlot:

    def __init__(self, root, bottle_file, rigid_body_files, translation_matrices):
        self.root = root
        self.bottle_file = bottle_file
        self.rigid_body_files = rigid_body_files
        self.translation_matrices = translation_matrices
        self._extract_coords_bottle()
        self._extract_coords_rigid_bodies()
        self.figure = self._create_plot()

    def _extract_coords_bottle(self):
        bottle_instance = InstanceThreeDx(self.bottle_file, coords_only=True)
        coords_bottle_np = np.asarray(bottle_instance.coords)[:, 1:]
        self.coords_bottle = coords_bottle_np

    def _extract_coords_rigid_bodies(self):
        rb_coords_dict = {
            f'{self.rigid_body_files[0]}': 0,
            f'{self.rigid_body_files[1]}': 0,
        }

        for i, rigid_body in enumerate(self.rigid_body_files):
            rb_file = f'./part_library/rigid_body_{rigid_body}.inp'
            try:
                rb_instance = InstanceThreeDx(rb_file, coords_only=True)
                rb_coords = rb_instance.coords

                rb_coords_translated = translate_coords(rb_coords, self.translation_matrices[i])
                dict_update = {rigid_body: rb_coords_translated}
                rb_coords_dict.update(dict_update)
            except FileNotFoundError:
                continue
        self.rb_coords = rb_coords_dict

    def _create_plot(self):
        bottle_coords = self.coords_bottle
        rigid_body_coords = self.rb_coords
        root = self.root
        root.configure(fg_color='transparent')

        plt.style.use('dark_background')

        figure = plt.figure(figsize=(5, 7.5), dpi=100)
        ax = figure.add_subplot(projection='3d')
        chart_type = FigureCanvasTkAgg(figure, root)
        chart_type.get_tk_widget().grid(column=0, row=0, sticky='nsew')

        ax.scatter(bottle_coords[:, 0], bottle_coords[:, 1], bottle_coords[:, 2])
        for rb in rigid_body_coords:
            try:
                rb_coords = rigid_body_coords.get(rb)
                label = rb
                ax.scatter(rb_coords[:, 0], rb_coords[:, 1], rb_coords[:, 2], label=label)
                ax.legend(loc='best', prop={'size': 10})
            except TypeError:
                continue

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect(aspect='equal')

        plt.tight_layout()
        return figure
