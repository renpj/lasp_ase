"""Molecular Dynamics."""
import warnings
from typing import IO, Optional, Union

import numpy as np
np.set_printoptions(threshold=np.inf)
import math

from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.logger import MDLogger
from ase.optimize.optimize import Dynamics
from lasp_ase import rmsd_new

def process_temperature(
    temperature: Optional[float],
    temperature_K: Optional[float],
    orig_unit: str,
) -> float:
    """Handle that temperature can be specified in multiple units.

    For at least a transition period, molecular dynamics in ASE can
    have the temperature specified in either Kelvin or Electron
    Volt.  The different MD algorithms had different defaults, by
    forcing the user to explicitly choose a unit we can resolve
    this.  Using the original method then will issue a
    FutureWarning.

    Four parameters:

    temperature: None or float
        The original temperature specification in whatever unit was
        historically used.  A warning is issued if this is not None and
        the historical unit was eV.

    temperature_K: None or float
        Temperature in Kelvin.

    orig_unit: str
        Unit used for the `temperature`` parameter.  Must be 'K' or 'eV'.

    Exactly one of the two temperature parameters must be different from
    None, otherwise an error is issued.

    Return value: Temperature in Kelvin.
    """
    if (temperature is not None) + (temperature_K is not None) != 1:
        raise TypeError("Exactly one of the parameters 'temperature',"
                        + " and 'temperature_K', must be given")
    if temperature is not None:
        w = "Specify the temperature in K using the 'temperature_K' argument"
        if orig_unit == 'K':
            return temperature
        elif orig_unit == 'eV':
            warnings.warn(FutureWarning(w))
            return temperature / units.kB
        else:
            raise ValueError("Unknown temperature unit " + orig_unit)

    assert temperature_K is not None
    return temperature_K


class MolecularDynamics(Dynamics):
    """Base-class for all MD classes."""

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        nmd_height = 8.0,
        nmd=False, nano_type=2,nano_k1=1,nano_k2=0.5,
        nano_t1=2,nano_t2=0.5,nano_r1=14,nano_r2=8,
        nano_center=[0,0,0],nmd_time = 1000, md_time=2000,
        meta=False, meta_global_factor = 1.0,meta_width = 0.2,
        meta_ramp = 0.03, meta_maxsave = 10, meta_time = 1000,
        meta_zmax = 2.0,meta_zmin = 1.0, 
        append_trajectory: bool = False,
    ):
        """Molecular Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        timestep: float
            The time step in ASE time units.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.
            Default: 1

        append_trajectory: boolean (optional)
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.
        """
        # dt as to be attached _before_ parent class is initialized
        self.dt = timestep

        super().__init__(atoms, logfile=None, trajectory=None)

        # Some codes (e.g. Asap) may be using filters to
        # constrain atoms or do other things.  Current state of the art
        # is that "atoms" must be either Atoms or Filter in order to
        # work with dynamics.
        #
        # In the future, we should either use a special role interface
        # for MD, or we should ensure that the input is *always* a Filter.
        # That way we won't need to test multiple cases.  Currently,
        # we do not test /any/ kind of MD with any kind of Filter in ASE.
        self.atoms = atoms
        self.masses = self.atoms.get_masses()

        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')

        self.masses.shape = (-1, 1)

        if not self.atoms.has('momenta'):
            self.atoms.set_momenta(np.zeros([len(self.atoms), 3]))

        # Trajectory is attached here instead of in Dynamics.__init__
        # to respect the loginterval argument.
        if trajectory is not None:
            if isinstance(trajectory, str):
                mode = "a" if append_trajectory else "w"
                trajectory = self.closelater(
                    Trajectory(trajectory, mode=mode, atoms=atoms)
                )
            self.attach(trajectory, interval=loginterval)

        if logfile:
            logger = self.closelater(
                MDLogger(dyn=self, atoms=atoms, logfile=logfile))
            self.attach(logger, loginterval)
            
        self.nmd = nmd        
        if self.nmd:        
            self.nano_type = nano_type        
            self.nano_k1 = nano_k1        
            self.nano_k2 = nano_k2        
            self.nano_t1 = nano_t1        
            self.nano_t2 = nano_t2        
            self.nano_r1 = nano_r1        
            self.nano_r2 = nano_r2        
            self.nmd_time = nmd_time      
            self.md_time = md_time    
            if self.nano_type == 1:        
                self.nano_center = nano_center        
                        
        self.meta = meta    
        if self.meta:    
            self.meta_global_factor  = meta_global_factor    
            self.meta_width   = meta_width     
            self.meta_ramp    = meta_ramp    
            self.meta_maxsave = meta_maxsave    
            self.meta_time    = meta_time    
            self.meta_zmax    = meta_zmax    
            self.meta_zmin    = meta_zmin    
            self.meta_number  = list()    
            self.meta_factor  = np.empty([self.meta_maxsave])    
            self.meta_nstruc  = 0    
            self.meta_step    = 0.0    
            self.meta_run     = 0.0    
            self.meta_struc   = list()  
     
        if self.nmd or self.meta:
            self.nmd_height = nmd_height
            self.nmd_number = list()
            for atom in self.atoms:
                if atom.z >= self.nmd_height:
                    self.nmd_number.append(atom.index)
            
    def todict(self):
        return {'type': 'molecular-dynamics',
                'md-type': self.__class__.__name__,
                'timestep': self.dt}

    def irun(self, steps=50):
        """Run molecular dynamics algorithm as a generator.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of molecular dynamics steps to be run.

        Yields
        ------
        converged : bool
            True if the maximum number of steps are reached.
        """
        try:
            return Dynamics.irun(self, steps=steps)
        except:
            self.max_steps = steps + self.nsteps
            return Dynamics.irun(self)


    def run(self, steps=50):
        """Run molecular dynamics algorithm.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of molecular dynamics steps to be run.

        Returns
        -------
        converged : bool
            True if the maximum number of steps are reached.
        """
        try:
            return Dynamics.run(self, steps=steps)
        except:
            self.max_steps = steps + self.nsteps
            return Dynamics.run(self)


    def get_time(self):
        return self.nsteps * self.dt

    def converged(self):
        """ MD is 'converged' when number of maximum steps is reached. """
        return self.nsteps >= self.max_steps

    def _get_com_velocity(self, velocity):
        """Return the center of mass velocity.
        Internal use only. This function can be reimplemented by Asap.
        """
        return np.dot(self.masses.ravel(), velocity) / self.masses.sum()
        
    def nmd_forces(self):        
        if self.nmd:        
            nano_ft = math.floor(self.nsteps * self.dt/self.nano_t1)        
            nano_f = self.nano_t2/self.nano_t1 + nano_ft - self.nsteps * self.dt/self.nano_t1        
            if nano_f > 0:        
                nano_k = self.nano_k2        
                nano_r = self.nano_r2        
            else:        
                nano_k = self.nano_k1        
                nano_r = self.nano_r1        
            n = len(self.atoms)        
            nano_force = np.empty([n,3])        
            if self.nano_type == 1:        
                for atom in self.atoms:        
                    position = atom.position - self.nano_center        
                    index = atom.index        
                    r = math.sqrt(position[0]**2+position[1]**2+position[2]**2)        
                    if (r - nano_r) > 0:        
                        nano_force[index] = atom.mass*nano_k*(r-nano_r)*position/r        
                    else:        
                        nano_force[index] = [0,0,0]        
            elif self.nano_type == 2:        
                for atom in self.atoms:        
                    index = atom.index        
                    r = abs(atom.z)        
                    nano_array = np.array([0,0,1])        
                    if (r - nano_r) > 0 and (atom.index in self.nmd_number):        
                        nano_force[index] = atom.mass*nano_k*(r-nano_r)*nano_array        
                    else:        
                        nano_force[index] = [0,0,0]        
        return nano_force        
            
    def position_in_pbc(self):        
        cell = self.atoms.cell        
        a1 = cell[1,1]/(cell[0,0]*cell[1,1]-cell[0,1]*cell[1,0])        
        b1 = -cell[0,1]/(cell[0,0]*cell[1,1]-cell[0,1]*cell[1,0])        
        c1 = -cell[1,0]/(cell[0,0]*cell[1,1]-cell[0,1]*cell[1,0])        
        d1 = cell[0,0]/(cell[0,0]*cell[1,1]-cell[0,1]*cell[1,0])        
        for atom in self.atoms:        
            direct_x = atom.x*a1+atom.y*c1        
            direct_y = atom.x*b1+atom.y*d1        
            if (direct_x>1. and direct_y>=-0. and direct_y<=1.):         
                atom.x = atom.x - cell[0,0]        
                atom.y = atom.y - cell[0,1]        
            elif (direct_x<-0. and direct_y>=-0. and direct_y<=1.):        
                atom.x = atom.x + cell[0,0]        
                atom.y = atom.y + cell[0,1]                        
            elif (direct_y>1. and direct_x>=-0. and direct_x<=1.):        
                atom.x = atom.x - cell[1,0]        
                atom.y = atom.y - cell[1,1]                         
            elif (direct_y<-0. and direct_x>=-0. and direct_x<=1.):        
                atom.x = atom.x + cell[1,0]        
                atom.y = atom.y + cell[1,1]         
            elif (direct_x<-0.0 and direct_y>1.):        
                atom.x = atom.x + cell[0,0] - cell[1,0]        
                atom.y = atom.y + cell[0,1] - cell[1,1]                         
            elif (direct_x>1. and direct_y>1.):        
                atom.x = atom.x - cell[0,0] - cell[1,0]        
                atom.y = atom.y - cell[0,1] - cell[1,1]                         
            elif (direct_x<-0. and direct_y<-0.):        
                atom.x = atom.x + cell[0,0] + cell[1,0]        
                atom.y = atom.y + cell[0,1] + cell[1,1]        
            elif (direct_x>1. and direct_y<-0.):        
                atom.x = atom.x - cell[0,0] + cell[1,0]        
                atom.y = atom.y - cell[0,1] + cell[1,1]        
                     
    def metadynic(self):        
        if self.meta:        
            #reference atoms        
            self.meta_number = list()        
            for atom in self.atoms:        
                if atom.z >= self.meta_zmin and atom.z <= self.meta_zmax\
                and (atom.index in self.nmd_number):        
                    self.meta_number.append(atom.index)        
                    
            # step of metadynamics        
            self.meta_run += 1.0        
                    
            #initial structure        
            if self.meta_nstruc == 0:        
                displacement = 1E-6        
                atoms = self.atoms.copy()        
                n = len(self.meta_number)        
                for i in range(0,n):        
                    while True:        
                        rcood = np.random.rand(3)        
                        rcood = 2.0* rcood - 1.0        
                        if (np.linalg.norm(rcood))>=1E-8:        
                            break        
                    rcood = rcood/np.linalg.norm(rcood)          
                    atoms[self.meta_number[i]].position = atoms[self.meta_number[i]].position + displacement*rcood                       
                self.meta_nstruc = 1        
                self.meta_struc.append(atoms)        
                    
            self.meta_step += 1.0        
            self.meta_factor[0:self.meta_nstruc] = self.meta_global_factor        
            self.meta_factor[self.meta_nstruc-1] = self.meta_factor[self.meta_nstruc-1]*\
            (2.0/(1.0+math.exp(-self.meta_ramp*self.meta_step))-1.0)        
                                    
            #update structure        
            if (self.meta_run*self.dt>=self.meta_time) and ((self.meta_run*self.dt)%\
            self.meta_time<1E-4):        
                if self.meta_nstruc < self.meta_maxsave:        
                    self.meta_step = 0.0        
                    self.meta_nstruc += 1        
                    atoms = self.atoms.copy()        
                    self.meta_struc.append(atoms)        
                elif self.meta_nstruc >= self.meta_maxsave:        
                    for i in range(1,self.meta_maxsave):        
                        self.meta_struc[i-1] = self.meta_struc[i]        
                    atoms = self.atoms.copy()        
                    self.meta_struc[self.meta_maxsave-1] = atoms        
                        
            #metadynamics        
            g = np.empty([len(self.atoms),3])     
            g[:,:] = 0.0    
            for iref in range(0,self.meta_nstruc):    
                xyzref = np.empty([len(self.meta_number),3])    
                xyzdup = np.empty([len(self.meta_number),3])    
                for i in range(0,len(self.meta_number)):    
                    xyzref[i,:] = self.meta_struc[iref][self.meta_number[i]].position    
                    xyzdup[i,:] = self.atoms[self.meta_number[i]].position    
                rmsdval,grad,g_ = rmsd_new.rmsd(len(self.meta_number),xyzdup,xyzref)
                #e = self.meta_factor[iref]*math.exp(-self.meta_width*rmsdval**2)
                #ebias = ebias + e# as a static method. 
                #etmp = -2.0 * self.meta_width * e * rmsdval                                
                e = self.meta_factor[iref]*math.exp(-self.meta_width)
                etmp = -self.meta_width * e
                print("step")
                print(self.meta_run)
                print("rmsd")
                print(rmsdval)
                print("k")
                print(self.meta_factor[iref])
                print("rmsd_grad")
                print(g_)
                print("grad_vector")
                print(grad)
                print("k*grad_vector")
                print(-self.meta_factor[iref]*grad[:,:])
                print("\n")
                for i in range(0,len(self.meta_number)):
                    #g[self.meta_number[i],:] += etmp*grad[i,:]                    
                    g[self.meta_number[i],:] += -self.meta_factor[iref]*grad[i,:]
            for i in range(0,len(self.atoms)):
                g[i,:]=g[i,:]*self.atoms[i].mass                        
        return g

    # Make the process_temperature function available to subclasses
    # as a static method.  This makes it easy for MD objects to use
    # it, while functions in md.velocitydistribution have access to it
    # as a function.
    _process_temperature = staticmethod(process_temperature)
