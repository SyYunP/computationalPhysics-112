import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def load_files(header,pattern='[0-9][0-9][0-9][0-9][0-9][0-9]'):
    """
    Load the data from the output file

    :param header: string, the header of the output file
    """
    fns = 'data_' + header+'/'+header +'_'+ pattern+'.dat'
    fns = glob.glob(fns)
    fns.sort()
    return fns


def save_movie(fns, lengthscale=1.0, filename='movie.mp4',fps=30):

    scale = lengthscale

    fig, ax = plt.subplots()
    fig.set_linewidth(5)
    fig.set_size_inches(10, 10, forward=True)
    fig.set_dpi(72)
    line, = ax.plot([], [], '.', color='w', markersize=2)

    def init():
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_aspect('equal')
        ax.set_xlabel('X [code unit]', fontsize=18)
        ax.set_ylabel('Y [code unit]', fontsize=18)
        return line, 

    def update(frame):
        fn = fns[frame]
        x,y = np.loadtxt(fn,delimiter=',',unpack=True,usecols=(3,4))
        line.set_data(x, y)
        plt.title("Frame ="+str(frame),size=18)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init, blit=True)
    ani.save(filename, writer='ffmpeg', fps=fps)

    return

def plot_energy(fns,plot_ke=True,plot_pe = True,plot_total=True,total_label='Total energy',ke_label='Kinetic energy',pe_label='Potential energy'):
    ke = np.zeros(len(fns))
    pe = np.zeros(len(fns))
    time = np.zeros(len(fns))
    
    for i,fn in enumerate(fns):
        t,ke_i,pe_i = np.loadtxt(fn,delimiter=',',unpack=True,usecols=(0,12,13))
        ke[i] = np.sum(ke_i)
        pe[i] = np.sum(pe_i)/2
        time[i] = t[0]

    if plot_total:
        plt.plot(time,ke+pe,label = total_label)
    
    if plot_ke:
        plt.plot(time,ke,label = ke_label)
    
    if plot_pe:
        plt.plot(time,pe,label = pe_label)

    return

def plot_energy_dissipation(fns,label):
    ke = np.zeros(len(fns))
    pe = np.zeros(len(fns))
    time = np.zeros(len(fns))

    for i,fn in enumerate(fns):
        t,ke_i,pe_i = np.loadtxt(fn,delimiter=',',unpack=True,usecols=(0,12,13))
        ke[i] = np.sum(ke_i)
        pe[i] = np.sum(pe_i)/2
        time[i] = t[0]
    
    tt = ke+pe
    plt.plot(time,abs((tt-tt[0])/tt[0]),label = label)
    return 

    
    


    
