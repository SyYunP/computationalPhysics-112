import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads

class Particles:
    """
    Particle class to store particle properties
    """
    def __init__(self,N:int=100,G:int=1,rsoft:float=0.01):
        self.nparticles = N
        self.time = 0
        self._G = G
        self._rsoft = rsoft
        self._tags = np.arange(N)
        self._masses = np.ones((N,1))
        self._positions = np.zeros((N,3))
        self._velocities = np.zeros((N,3))
        self._accelerations = np.zeros((N,3))
        return
    
    def _calculate_ke(self):
        mass = self.masses
        vel = self.velocities
        vel = np.linalg.norm(vel,axis=1).reshape(mass.shape)
        return 0.5*mass*vel*vel
    
    def _calculate_pe(self,use_numba):
        nparticles = self.nparticles
        G = self.G
        mass = self.masses
        pos = self.positions
        rsoft = self.rsoft
        pe = np.zeros_like(mass)
        if use_numba:
            pe = calculate_pe_kernel(pe,nparticles,G,mass,pos,rsoft)
        else:
            for i in range(nparticles):
                pe_i = 0
                for j in range(nparticles):
                    if (j!=i):
                        rij = pos[i,:]-pos[j,:]
                        r = (sum(rij*rij)+rsoft*rsoft)**0.5
                        pe_i -= G*mass[i,0]*mass[j,0]/r
                pe[i] = pe_i
        return pe

    def output(self,filename,output_energy=False,use_numba=True):
        if output_energy:
            ke = self._calculate_ke()
            pe = self._calculate_pe(use_numba)
            header = "# time,masses,tags,x,y,z,vx,vy,vz,ax,ay,az,ke,pe"
            np.savetxt(filename,np.hstack((np.ones((self.nparticles,1))*self.time,
                                           self._masses,
                                           self._tags.reshape(-1,1),
                                           self._positions,
                                           self._velocities,
                                           self._accelerations,
                                           ke,
                                           pe)),
                                           delimiter=",",header=header)
        else:
            header = "# time,masses,tags,x,y,z,vx,vy,vz,ax,ay,az"
            np.savetxt(filename,np.hstack((np.ones((self.nparticles,1))*self.time,
                                           self._masses,
                                           self._tags.reshape(-1,1),
                                           self._positions,
                                           self._velocities,
                                           self._accelerations)),
                                           delimiter=",",header=header)
        return

    def draw(self,dim,pov = None):
        if dim == 2:
            if pov == 'xy':
                plt.scatter(self._positions[:,0],self._positions[:,1],alpha=.2,c = 'orange',s = 2)
                plt.xlabel('x')
                plt.ylabel('y')
            elif pov == 'xz':
                plt.scatter(self._positions[:,0],self._positions[:,2],alpha=.2,c = 'orange',s = 2)
                plt.xlabel('x')
                plt.ylabel('z')
            elif pov == None: 
                print('Point of view is not defined.')
            else:
                print('Point of view is not valid.')
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self._positions[:,0],self._positions[:,1],self._positions[:,2],alpha=.2,c = 'orange',s = 2)
        else:
            print('Input is not valid.')
        return

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, some_G):
        if type(some_G) != float:
            print("G is a float!")
            raise TypeError
        self._G = some_G
        return
    
    @property
    def rsoft(self):
        return self._rsoft

    @rsoft.setter
    def rsoft(self, some_rsoft):
        if type(some_rsoft) != float:
            print("rsoft is a float!")
            raise TypeError
        self._rsoft = some_rsoft
        return

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, some_tags):
        if len(some_tags) != self.nparticles:
            print("Number of particles does not match!")
            raise ValueError
        self._tags = some_tags
        return

    @property
    def masses(self):
        return self._masses

    @masses.setter
    def masses(self, some_masses):
        if some_masses.shape != self._masses.shape:
            print("Shape does not match!")
            raise ValueError
        self._masses = some_masses
        return
    
    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, some_positions):
        if some_positions.shape != self._positions.shape:
            print("Shape does not match!")
            raise ValueError
        self._positions = some_positions
        return
    
    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, some_velocities):
        if some_velocities.shape != self._velocities.shape:
            print("Shape does not match!")
            raise ValueError
        self._velocities = some_velocities
        return
    
    @property
    def accelerations(self):
        return self._accelerations

    @accelerations.setter
    def accelerations(self, some_accelerations):
        if some_accelerations.shape != self._accelerations.shape:
            print("Shape does not match!")
            raise ValueError
        self._accelerations = some_accelerations
        return

@njit(parallel=True)
def calculate_pe_kernel(pe,nparticles,G,masses,positions,rsoft):
    for i in prange(nparticles):
        pe_i = 0
        for j in prange(nparticles):
            if (j!=i):
                rij = positions[i,:]-positions[j,:]
                r = (sum(rij*rij)+rsoft*rsoft)**0.5
                pe_i -= G*masses[i,0]*masses[j,0]/r
        pe[i] = pe_i
    return pe

if __name__ == "__main__":

    pass

