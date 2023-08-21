import os.path
from pathlib import Path

from inp_populater.mapping_data_processing.parse_data import extract_data
from inp_populater.mapping_data_processing.map_data import mapping_handler
from inp_populater.mapping_data_processing.check_data import check_data_from, check_data_to, check_orientation


class MapInputs:
    cwd = os.getcwd()

    def __init__(self, map_from, map_to, bottle_mass,
                 inp_out=None, sth_profile=None, mod_profile=None):
        """Constructor.

        param map_from: file with co-ordinates and sth/mod if mapped.
        param map_to: file with co-ordinates (usually larger dataset).
        param inp_out: .inp file to be rewritten.
        param sth_profile: sth profile to append to map_from file if not present.
        """

        self.map_from = map_from
        self.map_to = map_to
        self.bottle_mass = float(bottle_mass)
        self.map_from_coords, self.map_from_shape = extract_data(self.map_from)
        self.map_to_coords, self.map_to_shape, self.connectivity = extract_data(self.map_to)

        self.checked = False
        self.written = False

        self.inp_out = inp_out if inp_out is not None else None
        self.sth_profile = sth_profile if sth_profile is not None else None
        self.mod_profile = mod_profile if mod_profile is not None else None

    # Methods
    def check(self):
        """
        Scans through provided data to check compatibility and sufficient info
        map_from: x, y, z, sth, mod
        map_to: x, y, z
        In both, z artificially added if none found
        """
        check_data_from(self.map_from)
        check_data_to(self.map_to)
        map_from_coords, map_to_coords = extract_data(self.map_from)[0], extract_data(self.map_to)[0]
        check_orientation(map_from_coords, map_to_coords)
        self.checked = True

    def map(self):
        """Maps data from map_from to map_to
        regression_fit: boolean
        if True, map_from file must contain rel_stretch_ratio and rel_modulus
        in order to fit relationship (not full length column). Column with
        stretch ratios must also be provided (of equal length to other dims).
        modulus_map_shape: 'linear', 'poly'
        """
        Path('{}\\Outputs\\mapped_files'.format(MapInputs.cwd)).mkdir(parents=True, exist_ok=True)

        scaling_params = {
            'map_from_file': self.map_from,
            'map_to_file': self.map_to,
            'map_from_coords': self.map_from_coords,
            'map_to_coords': self.map_to_coords,
            'map_from_shape': self.map_from_shape,
            'map_to_shape': self.map_to_shape,
            'connectivity': self.connectivity,
            'sth_profile': self.sth_profile,
            'mod_profile': self.mod_profile,
            'scaling': 'global',  # Type of scaling, leave empty for no scaling
            'axis_height': 'z',  # Refers to axes of output bottle
            'axis_pri': 'x',
            'axis_sec': 'y',
            'shift_factor': 0,  # Leave empty for no shift, use 'mould' to use mould height
            'rotate': [  # Leave empty for no rotation
                {'axis': [0, 0, 0], 'degrees': 90}  # [x, y, z]
            ],

            'sth_map': True,
            'modulus_map': True,
            'adj_sth': True,
            'adj_mod': False,
            'constant_thickness': None,  # None if thickness and/or modulus mapping is required
            'constant_modulus': None,
            'modulus_limits': [1700, 6500],
            'regression_fit': False,  # Fitting modulus given match moduli values and stretch ratios
            'modulus_fit_shape': 'linear',  # if above is True: 'linear' or 'poly'
            'density': 1.335e-09,
            'poissons_ratio': 0.495,
            'bottle_mass': self.bottle_mass,  # Exclude neck weight, leave empty if mass has been exported
            'neck_mass': 3.5,
            'section_cuts': [10, 40, 70, 130]
        }

        self.mapped_coords_mod, self.mapped_coords_sth = mapping_handler(scaling_params)
        return self.mapped_coords_mod, self.mapped_coords_sth
