import os


class CreateInp:

    def __init__(self, ref_node, coords, nodes, elements, elsets, part):
        self.ref_node = ref_node
        self.coords = coords
        self.nodes = nodes
        self.elements = elements
        self.elset = elsets
        self.part = part
        self.kword_identifier = f'Rigid Body{self.part}'

    def write_to_inp(self):
        os.chdir('../part_library')
        inp_file = f'rigid_body_{self.part}.inp'
        with open(inp_file, 'w') as f:
            self._write_ref_node(f)
            self._write_coords(f)
            self._write_nsets(f)
            self._write_connectivity(f)
            self._write_elsets(f)

    def _write_ref_node(self, fout):
        fout.write('*REF NODE\n')
        fout.write(f'{self.ref_node}\n')

    def _write_coords(self, fout):
        fout.write('*NODE\n')
        for i in range(len(self.coords)):
            fout.write(f'{i + 1}, {self.coords[i][1]}, {self.coords[i][2]}, {self.coords[i][3]}\n')

    def _write_nsets(self, fout):
        fout.write(f'*NSET, NSET="{self.kword_identifier}"\n')
        for node in self.nodes:
            fout.write(f'{node}\n')

    def _write_connectivity(self, fout):
        fout.write('*ELEMENT\n')
        for i in range(len(self.elements)):
            if len(self.elements[i]) == 4:
                fout.write(
                    f'{self.elements[i][0]}, {self.elements[i][1]}, {self.elements[i][2]}, {self.elements[i][3]}\n'
                )
            else:
                fout.write(
                    f'{self.elements[i][0]}, {self.elements[i][1]}, {self.elements[i][2]}, {self.elements[i][3]}, '
                    f'{self.elements[i][4]}\n'
                )

    def _write_elsets(self, fout):
        fout.write(f'*ELSET, ELSET="{self.kword_identifier}"\n')
        for i in range(len(self.elset)):
            fout.write(f'{self.elset[i]}\n')
