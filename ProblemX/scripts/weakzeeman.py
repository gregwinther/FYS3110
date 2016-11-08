from matplotlib import pyplot as plt 
import numpy as np 

def total_energy(l, j, j_z, g_j, muBB, n=2):
	alpha = 1.0/137.036
	g_j = 1 + (j*(j+1) + (3/4) - l*(l + 1))/(2*j*(j + 1))
	return -(13.6/n**2) * (1 + (alpha/n**2) * ((n/(j + 0.5)) - (3/4))) \
			+ (muBB * j_z * g_j)

muBB = np.linspace(0,1)

plt.plot(muBB, total_energy(0, 0.5, +0.5, 2, muBB), 'r')
plt.plot(muBB, total_energy(0, 0.5, -0.5, 2, muBB), 'r')
plt.plot(muBB, total_energy(1, 0.5, +0.5, (2/3), muBB), 'b')
plt.plot(muBB, total_energy(1, 0.5, -0.5, (2/3), muBB), 'b')
plt.plot(muBB, total_energy(1, (3/2), -0.5, (4/3), muBB),'g')
plt.plot(muBB, total_energy(1, (3/2), +0.5, (4/3), muBB),'g')
plt.plot(muBB, total_energy(1, (3/2), +(3/2), (4/3), muBB),'y')
plt.plot(muBB, total_energy(1, (3/2), -(3/2), (4/3), muBB),'y')
plt.show()
