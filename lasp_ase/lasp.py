""" 
This module defines a FileIOCalculator for LASP
Only support define the calculator and call the binary, input generation
 and output analysis don't support yet.
"""

import os
import numpy as np
from ase.calculators.calculator import FileIOCalculator


class Lasp(FileIOCalculator):
    if 'ASE_LASP_COMMAND' in os.environ:
        command = os.environ['ASE_LASP_COMMAND'] 
    else:
        command = 'mpirun -np 2 lasp'
        
    implemented_properties = ['energy', 'forces']
 
    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='lasp', atoms=None, **kwargs):

        self.lines = None
        self.atoms = None
        self.atoms_input = None
        self.do_forces = False
        self.outfilename = 'lasp.out'
        if not kwargs.get('command'):
            kwargs['command'] = self.command
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        from ase.io import write
        if properties is not None:
            if 'forces' in properties or 'stress' in properties:
                self.do_forces = True
        FileIOCalculator.write_input(
            self, atoms, properties, system_changes)
        write(os.path.join(self.directory, 'input.arc'), atoms, format="dmol-arc",
              parallel=False)
        # self.atoms is none until results are read out,
        # then it is set to the ones at writing input
        self.atoms_input = atoms
        self.atoms = None      

    def read_results(self):

        with open(os.path.join(self.directory, 'allfor.arc'), 'r') as fd:
            self.lines = fd.readlines()

        self.atoms = self.atoms_input
        energy = float(self.lines[0].split()[3])
        self.results['energy'] = energy
        if self.do_forces:
            forces = self.read_forces()
            self.results['forces'] = forces

    def read_forces(self):

        gradients = []
        for i in range(2,len(self.atoms)+2): #self.lines)-1):
            tmp = []
            for j in range(0,3):
                tmp.append(float(self.lines[i].split()[j]))
            gradients.append(tmp)      

        return np.array(gradients) 

   
   



