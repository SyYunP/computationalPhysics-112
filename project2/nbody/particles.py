import numpy as np
import matplotlib.pyplot as plt
import h5py

class Particles:
    """
    Particle class to store particle properties
    """
    def __init__(self,N:int=100):
        self.nparticles = N
        self.time = 0
        self._tags = np.arange(N)
        self._masses = np.ones((N,1))
        self._positions = np.zeros((N,3))
        self._velocities = np.zeros((N,3))
        self._accelerations = np.zeros((N,3))
        return
    
    def output(self,filename):
        header = "# masses,tags,x,y,z,vx,vy,vz,ax,ay,az"
        np.savetxt(filename,np.hstack((self._masses,
                                       self._tags.reshape(-1,1),
                                       self._positions,
                                       self._velocities,
                                       self._accelerations)),
                                       delimiter=",",header=header)
        return

    def draw(self,dim,pov = None):
        if dim == 2:
            if pov == 'xy':
                plt.scatter(self._positions[:,0],self._positions[:,1])
                plt.xlabel('x')
                plt.ylabel('y')
            elif pov == 'xz':
                plt.scatter(self._positions[:,0],self._positions[:,2])
                plt.xlabel('x')
                plt.ylabel('z')
            elif pov == None: 
                print('Point of view is not defined.')
            else:
                print('Point of view is not valid.')
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self._positions[:,0],self._positions[:,1],self._positions[:,2])
        else:
            print('Input is not valid.')
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


    pass

