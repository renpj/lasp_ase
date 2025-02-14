"""Demonstrates molecular dynamics with constant temperature."""

from ase.io import read,write,trajectory
from ase import Atom,Atoms
from ase import units
from ase.constraints import FixAtoms
from lasp_ase.lasp import Lasp
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from lasp_ase.andersen_new import Andersen
import numpy as np
from ase.optimize import BFGS

atoms_0 = read("input.cif")
atoms = atoms_0.repeat((2,2,1))
write("input_2x2.cif",atoms)

c = FixAtoms(indices=[atom.index for atom in atoms if atom.z < 3])
atoms.set_constraint(c)
#meta_global_factor = (len(atoms)-120)*1
meta_global_factor = 0.5
calc=Lasp()                                                                        
atoms.calc = calc  
surface_top = 5.40
r1=atoms.cell[2][2]
print(f"surface_top is {surface_top}, r1 is {r1}")

######## opt #######
#opt = BFGS(atoms,trajectory='opt.traj')
#opt.run (fmax=0.05)
#atoms.write('opt.cif')

######### MD #########

MaxwellBoltzmannDistribution(atoms, temperature_K=400)
dyn = Andersen(
    atoms, 0.5 * units.fs, 400, 0.2, trajectory="md.traj", 
    logfile="md.out", loginterval=20, nmd_height=surface_top - 1, nmd=True, 
    nano_type=2, nano_k1=0.1, nano_k2=0.1, nano_t1=2000 * units.fs, 
    nano_t2=1000 * units.fs, nano_r1=r1, nano_r2=surface_top + 2, 
    nmd_time=2000 * units.fs, md_time=6000 * units.fs, meta=True, 
    meta_global_factor=meta_global_factor, meta_width=0.2, meta_ramp=0.03, 
    meta_maxsave=1, meta_time=10 * units.fs, meta_zmax=surface_top + 3.5,
    meta_zmin=surface_top - 3.5 # , meta_symbol = ['C','H','O']
)
        
dyn.run(200)
write("output.cif",atoms)
