"""Demonstrates molecular dynamics with constant temperature."""

from ase.io import read, write
from ase.constraints import FixAtoms
from lasp_ase.lasp import Lasp
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS


atoms_0 = read("input.cif")
atoms = atoms_0.repeat((2,2,1))
write("input_2x2.cif",atoms)

c = FixAtoms(indices=[atom.index for atom in atoms if atom.z < 3])
atoms.set_constraint(c)
meta_global_factor = 0.5
calc=Lasp()                                                                        
atoms.calc = calc  
opt = BFGS(atoms,trajectory='opt.traj')
opt.run (fmax=0.05)
atoms.write('opt.cif')

