from ase.io import read
from ase.constraints import FixAtoms
from lasp_ase.lasp import Lasp
from ase.optimize import BFGS

atoms = read("input.cif")
c = FixAtoms(indices=[atom.index for atom in atoms if atom.z < 3])
atoms.set_constraint(c)
calc=Lasp()
atoms.calc = calc  
opt = BFGS(atoms,trajectory='opt.traj')
opt.run (fmax=0.1)

