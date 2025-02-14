"""Andersen dynamics class."""
from typing import IO, Optional, Union

from numpy import cos, log, ones, pi, random, repeat

from ase import Atoms, units
from lasp_ase.md_new import MolecularDynamics
from ase.parallel import DummyMPI, world

import math

class Andersen(MolecularDynamics):
    """Andersen (constant N, V, T) molecular dynamics."""

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        andersen_prob: float,
        fixcm: bool = True,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        communicator=world,
        rng=random,
        append_trajectory: bool = False,
        nmd_height = 8.0,nmd=False, nano_type=2,nano_k1=1,nano_k2=0.5,
        nano_t1=2,nano_t2=0.5,nano_r1=14,nano_r2=8,
        nano_center=[0,0,0],nmd_time=1000,md_time = 2000,
        meta=False, meta_global_factor = 1.0,meta_width = 0.2,
        meta_ramp = 0.03, meta_maxsave = 10, meta_time = 1000,
        meta_zmax = 2.0,meta_zmin = 1.0    
    ):
        """"
        Parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature_K: float
            The desired temperature, in Kelvin.

        andersen_prob: float
            A random collision probability, typically 1e-4 to 1e-1.
            With this probability atoms get assigned random velocity components.

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        rng: RNG object (optional)
            Random number generator, by default numpy.random.  Must have a
            random_sample method matching the signature of
            numpy.random.random_sample.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str (optional)
            Attach trajectory object. If *trajectory* is a string a
            Trajectory will be constructed. Use *None* (the default) for no
            trajectory.

        communicator: MPI communicator (optional)
            Communicator used to distribute random numbers to all tasks.
            Default: ase.parallel.world. Set to None to disable communication.

        append_trajectory: bool (optional)
            Defaults to False, which causes the trajectory file to be
            overwritten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        The temperature is imposed by stochastic collisions with a heat bath
        that acts on velocity components of randomly chosen particles.
        The algorithm randomly decorrelates velocities, so dynamical properties
        like diffusion or viscosity cannot be properly measured.

        H. C. Andersen, J. Chem. Phys. 72 (4), 2384–2393 (1980)
        """
        self.temp = units.kB * temperature_K
        self.andersen_prob = andersen_prob
        self.fix_com = fixcm
        self.rng = rng
        if communicator is None:
            communicator = DummyMPI()
        self.communicator = communicator
        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,nmd_height,
                                   nmd, nano_type,nano_k1,nano_k2,
                                   nano_t1,nano_t2,nano_r1,nano_r2,
                                   nano_center,nmd_time,md_time,
                                   meta, meta_global_factor,meta_width,
                                   meta_ramp, meta_maxsave, meta_time,
                                   meta_zmax,meta_zmin,
                                   append_trajectory=append_trajectory)

    def set_temperature(self, temperature_K):
        self.temp = units.kB * temperature_K

    def set_andersen_prob(self, andersen_prob):
        self.andersen_prob = andersen_prob

    def set_timestep(self, timestep):
        self.dt = timestep

    def boltzmann_random(self, width, size):
        x = self.rng.random_sample(size=size)
        y = self.rng.random_sample(size=size)
        z = width * cos(2 * pi * x) * (-2 * log(1 - y))**0.5
        return z

    def get_maxwell_boltzmann_velocities(self):
        natoms = len(self.atoms)
        masses = repeat(self.masses, 3).reshape(natoms, 3)
        width = (self.temp / masses)**0.5
        velos = self.boltzmann_random(width, size=(natoms, 3))
        return velos  # [[x, y, z],] components for each atom

    def step(self, forces=None):
        atoms = self.atoms
        
        nano_ft = math.floor(self.nsteps * self.dt/self.md_time)
        nano_f = self.nmd_time/self.md_time + nano_ft - self.nsteps * self.dt/self.md_time
        if nano_f >= 0.0:
            self.nmd = True
            self.meta = False
        elif nano_f < 0.0:
            self.nmd = False
            self.meta = True   

        if forces is None:
            forces = atoms.get_forces(md=True)
            
        if self.nmd:
            nano_forces = self.nmd_forces()
            forces -= nano_forces    
                
        if self.meta:       
            meta_forces = self.metadynic()        
            forces -= meta_forces        

        self.v = atoms.get_velocities()

        # Random atom-wise variables are stored as attributes and broadcasted:
        #  - self.random_com_velocity  # added to all atoms if self.fix_com
        #  - self.random_velocity      # added to some atoms if the per-atom
        #  - self.andersen_chance      # andersen_chance <= andersen_prob
        # a dummy communicator will be used for serial runs

        if self.fix_com:
            # add random velocity to center of mass to prepare Andersen
            width = (self.temp / sum(self.masses))**0.5
            self.random_com_velocity = (ones(self.v.shape)
                                        * self.boltzmann_random(width, (3)))
            self.communicator.broadcast(self.random_com_velocity, 0)
            self.v += self.random_com_velocity

        self.v += 0.5 * forces / self.masses * self.dt

        # apply Andersen thermostat
        self.random_velocity = self.get_maxwell_boltzmann_velocities()
        self.andersen_chance = self.rng.random_sample(size=self.v.shape)
        self.communicator.broadcast(self.random_velocity, 0)
        self.communicator.broadcast(self.andersen_chance, 0)
        self.v[self.andersen_chance <= self.andersen_prob] \
            = self.random_velocity[self.andersen_chance <= self.andersen_prob]

        x = atoms.get_positions()
        if self.fix_com:
            old_com = atoms.get_center_of_mass()
            self.v -= self._get_com_velocity(self.v)
        # Step: x^n -> x^(n+1) - this applies constraints if any
        atoms.set_positions(x + self.v * self.dt)
        if self.fix_com:
            atoms.set_center_of_mass(old_com)

        # recalc velocities after RATTLE constraints are applied
        self.v = (atoms.get_positions() - x) / self.dt
        forces = atoms.get_forces(md=True)
        
        if self.nmd:
            nano_forces = self.nmd_forces()
            forces -= nano_forces
            
        if self.meta:   
            meta_forces = self.metadynic()    
            forces -= meta_forces    

        # Update the velocities
        self.v += 0.5 * forces / self.masses * self.dt

        if self.fix_com:
            self.v -= self._get_com_velocity(self.v)

        if self.nmd:
            self.position_in_pbc()
            self.meta_struc = list()
            self.meta_nstruc = 0
            self.meta_step    = 0.0
            self.meta_run     = 0.0

        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)

        return forces
