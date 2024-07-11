import jax.numpy as np
import numpy as onp
from erg_expl import ErgodicTrajectoryOpt
from gaussian import gaussian
from IPython.display import clear_output
import matplotlib.pyplot as plt

def main(num_agents, map_size, info_map=None, init_pos=None, plot=True):
    if info_map is None:
        info_map = sample_map(map_size)
    if init_pos is None:
        init_pos = sample_initpos(num_agents, map_size)
        
    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()

    traj_opt = ErgodicTrajectoryOpt(init_pos, info_map, num_agents, map_size)
    for k in range(100):
        traj_opt.solver.solve(max_iter=1000)
        sol = traj_opt.solver.get_solution()
        clear_output(wait=True)

    for i in range(num_agents):
        path_travelled[i][0].append(sol['x'][:,i][:,0]+(map_size/2))
        path_travelled[i][1].append(sol['x'][:,i][:,1]+(map_size/2))

    if plot == True:
        cmap = get_colormap(num_agents+1)
        fig, ax = plt.subplots()
        ax.imshow(info_map, origin="lower")
        for i in range(num_agents):
            ax.plot(np.array(path_travelled[i][0]).flatten(), np.array(path_travelled[i][1]).flatten(), c=cmap(i))
        plt.show()

    return path_travelled

### Helpers ###

def sample_map(size, peaks=3):
    pos = np.floor(onp.random.uniform(0, size, 2*peaks))
    pmap = gaussian(size, pos[0], pos[1], 10)
    peak_indices = [np.where(pmap>.1)]
    for i in range(1, peaks):
        new_peak = gaussian(size, pos[2*i], pos[2*i+1], 10)
        pmap += gaussian(size, pos[2*i], pos[2*i+1], 10)
        peak_indices.append(np.where(new_peak>.1))
    return pmap

def sample_initpos(agents, size):
    return onp.random.uniform(-size/2, size/2, (agents, 2))

def get_colormap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

if __name__ == "__main__":
    main(num_agents=3, map_size=100)
