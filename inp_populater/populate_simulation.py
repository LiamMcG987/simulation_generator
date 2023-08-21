import os

from .instance_inp import InstanceInp
from .write_to_inp import WriteToInp


class PopulateSimulation:
    cwd = os.getcwd()

    def __init__(self, sbm_data, bottle: str, simulation: str):
        """
        SBM data required for mapping side.
        Bottle file name required - only name, no extension (e.g., 'bottle_1', not 'bottle_1.inp').
        Simulation name required - only load case (e.g., 'top load', 'side load').

        Script extracts template file based on simulation name, bottle file based on bottle name,
        and grabs performance parts based on simulation name.
        """
        self.sbm_data = sbm_data
        self.bottle = bottle
        self.simulation = simulation
        self.inp_files = []
        self._get_template_file()
        self._get_bottle_files()
        self._get_performance_parts()
        self._write_simulation_inp()

    def _get_template_file(self):
        self.template_file = None
        os.chdir('./template_files')
        self.simulation = self.simulation.strip().replace(' ', '_').lower()
        for file in os.listdir():
            if file.startswith(f'{self.simulation}'):
                self.template_file = file
                self.template_file_path = f'{PopulateSimulation.cwd}/template_files/{file}'
        if self.template_file is None:
            raise TypeError('No template file found matching input. Check template_files folder for current templates.')

    def _get_bottle_files(self):
        self.bottle_file = None
        os.chdir('../part_library/bottles')
        bottle = self.bottle.strip().lower()
        for file in os.listdir():
            if bottle in file.lower():
                self.bottle_file = file
        if self.bottle_file is None:
            raise TypeError('Bottle file not located. Ensure correct directory is used.')
        self.inp_files.append(f'{PopulateSimulation.cwd}/part_library/bottles/{self.bottle_file}')

    def _get_performance_parts(self):
        os.chdir('..')
        sim_parts = {
            'top_load': ['top_plate', 'ground_plate'],
            'squeeze': ['finger', 'hand'],
            'side_load': ['side', 'side']
        }

        parts = []
        for part in sim_parts.get(self.simulation):
            identifier = f'rigid_body_{part}.inp'
            if identifier in os.listdir():
                parts.append(identifier)
                self.inp_files.append(f'{PopulateSimulation.cwd}/part_library/{identifier}')
        if not parts:
            raise TypeError('Performance part files not located. Ensure correct directory is created.')

    def _write_simulation_inp(self):
        """
        All data held in dictionary (sim_comps).
        Call key of part to get instance of component - use attributes to extract data.
        """
        os.chdir(f'{PopulateSimulation.cwd}/simulation_inps')
        bottle_name = f'{self.bottle.strip().replace(" ", "_")}'
        simulation_inp = f'{os.path.splitext(bottle_name)[0]}_{self.simulation}.inp'

        simulation_inp_components = {}
        for file in self.inp_files:
            if 'rigid_body' not in str(file).lower():
                file_name = 'bottle'
            else:
                file_name = os.path.splitext(os.path.split(file)[-1])[0]
            print(file_name)
            simulation_inp_components[file_name] = InstanceInp(file, self.simulation, self.sbm_data)

        self.parts_dict = simulation_inp_components
        WriteToInp(self.parts_dict, self.template_file_path, simulation_inp)
