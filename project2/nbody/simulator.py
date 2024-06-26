import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies



"""

class NBodySimulator:

    def __init__(self, particles: Particles):
        
        self.particles = particles
        self.setup()   #use default settings

        return

    def setup(self, G=1.,
                    rsoft=0.01,
                    method="RK4",
                    io_freq=10,
                    io_header="nbody",
                    io_screen=True,
                    use_numba=True,
                    output_energy = False,
                    snapshots=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not. 
        """
        self.particles.G = G
        self._G = G
        self.particles.rsoft = rsoft
        self._rsoft = rsoft
        self.method = method
        self.io_freq = io_freq
        self.io_header = io_header
        self.io_screen = io_screen
        self.use_numba = use_numba
        self.output_energy = output_energy
        self.snapshots = snapshots
        if method.lower() == 'euler':
            self._advance_particles = self._advance_particles_Euler
        elif method.lower() == 'rk2':
            self._advance_particles = self._advance_particles_RK2
        elif method.lower() == 'rk4':
            self._advance_particles = self._advance_particles_RK4
        elif method.lower() == 'leapfrog':
            self._advance_particles = self._advance_particles_LF
        
        return 
 

    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """

        # TODO
        time = self.particles.time
        nsteps = int(np.ceil((tmax-time)/dt))
        
        for n in range(nsteps):
            #make sure the last step is correct
            if (time+dt) > tmax:
                dt = tmax-time
            
            #update (physics)
            self.particles = self._advance_particles(dt,self.particles)
            self.particles.time = time

            #check io
            if self.io_screen == True:
                print("Times:",time,"dt:",dt)


            if self.io_freq!=0:
                if (n % self.io_freq) == 0:
                    use_numba = self.use_numba
                    output_energy = self.output_energy
                    #checkout directory
                    folder = "data_"+self.io_header
                    path = Path(folder).mkdir(parents=True,exist_ok=True)
                    fn = self.io_header+"_"+str(n).zfill(6)+".dat"
                    self.particles.output(f"{folder}/{fn}",output_energy,use_numba)
                    if self.snapshots:
                        self.particles.draw(dim=2,pov='xy')
                        plt.savefig(f'{folder}/fig'+str(n).zfill(6)+'.png')

            time+=dt

        print("Simulation is done!")
        return

    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """
        accelerations = np.zeros_like(positions)
        rsoft = self._rsoft
        G = self._G
        if self.use_numba:
            accelerations = calculate_acceleration_numba(nparticles,masses,positions,accelerations,rsoft,G)
        else:
            for i in range(nparticles):
                for j in range(nparticles):
                    if (j>i):
                        rij = positions[i]-positions[j]
                        r = np.sqrt(np.sum(rij**2)+rsoft**2)
                        force = - G*masses[i]*masses[j]/r**3*rij
                        accelerations[i] = accelerations[i]+force/masses[i]
                        accelerations[j] = accelerations[j]-force/masses[j]

        return accelerations
    
    def _advance_particles_Euler(self, dt, particles:Particles):
        nparticles = particles.nparticles
        pos = particles.positions
        vel = particles.velocities
        mass = particles.masses
        acc = self._calculate_acceleration(nparticles,mass,pos)

        pos = pos+vel*dt
        vel = vel+acc*dt
        # acc = self._calculate_acceleration(nparticles,mass,pos)

        particles.positions = pos
        particles.velocities = vel
        particles.accelerations = acc

        return particles

    def _advance_particles_RK2(self, dt, particles:Particles):

        # TODO
        nparticles = particles.nparticles
        pos = particles.positions
        vel = particles.velocities
        mass = particles.masses
        acc = self._calculate_acceleration(nparticles,mass,pos)

        pos2 = pos+vel*dt
        vel2 = vel+acc*dt
        acc2 = self._calculate_acceleration(nparticles,mass,pos2)

        pos2 = pos2+vel2*dt
        vel2 = vel2+acc2*dt

        pos = 0.5*(pos+pos2)
        vel = 0.5*(vel+vel2)
        acc = self._calculate_acceleration(nparticles,mass,pos)

        particles.positions = pos
        particles.velocities = vel
        particles.accelerations = acc

        return particles

    def _advance_particles_RK4(self, dt, particles:Particles):
        
        #TODO
        nparticles = particles.nparticles
        pos = particles.positions
        vel = particles.velocities
        mass = particles.masses
        acc = self._calculate_acceleration(nparticles,mass,pos)
        
        pos1 = pos+vel*0.5*dt
        vel1 = vel+acc*0.5*dt
        acc1 = self._calculate_acceleration(nparticles,mass,pos1)

        pos2 = pos+vel1*0.5*dt
        vel2 = vel+acc1*0.5*dt
        acc2 = self._calculate_acceleration(nparticles,mass,pos2)
        
        pos3 = pos+vel2*dt
        vel3 = vel+acc2*dt
        acc3 = self._calculate_acceleration(nparticles,mass,pos3)

        pos = pos+(vel+2*vel1+2*vel2+vel3)*dt/6
        vel = vel+(acc+2*acc1+2*acc2+acc3)*dt/6
        acc = self._calculate_acceleration(nparticles,mass,pos)
        
        particles.positions = pos
        particles.velocities = vel
        particles.accelerations = acc

        return particles
    
    def _advance_particles_LF(self,dt,particles:Particles):
        nparticles = particles.nparticles
        pos = particles.positions
        vel = particles.velocities
        mass = particles.masses
        acc = self._calculate_acceleration(nparticles,mass,pos)
        
        vel += 0.5*acc*dt
        pos += vel*dt
        acc = self._calculate_acceleration(nparticles,mass,pos)
        vel += 0.5*acc*dt

        particles.positions = pos
        particles.velocities = vel
        particles.accelerations = acc

        return particles

    
@njit(parallel=True)
def calculate_acceleration_numba(nparticles, masses, positions,accelerations,rsoft,G):
        """
        Calculate the acceleration of the particles
        """
        
        for i in prange(nparticles):
            for j in prange(nparticles):
                if (j>i):
                    rij = positions[i,:]-positions[j,:]
                    r = (sum(rij**2)+rsoft**2)**0.5
                    force = - G*masses[i,0]*masses[j,0]/r**3*rij
                    accelerations[i,:] = accelerations[i,:]+force/masses[i,0]
                    accelerations[j,:] = accelerations[j,:]-force/masses[j,0]

        return accelerations



if __name__ == "__main__":
    
    pass