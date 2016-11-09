from matplotlib import pyplot as plt 
import numpy as np 

def total_energy(l, j, j_z, g_j, muBB, n=2):
	alpha = 1.0/137.036
	g_j = 1 + (j*(j+1) + (3/4) - l*(l + 1))/(2*j*(j + 1))
	return -(13.6/n**2) * (1 + (alpha/n**2) * ((n/(j + 0.5)) - (3/4))) \
			+ (muBB * j_z * g_j)

muBB = np.linspace(0,1)


plt.plot(muBB, total_energy(0, 0.5, +0.5, 2, muBB), 'r',\
	label=r"$l=0$, $j=\frac{1}{2}$, $j_z=\pm\frac{1}{2}$, $g_j=2$, slope=$\pm1$")
plt.plot(muBB, total_energy(0, 0.5, -0.5, 2, muBB), 'r')
plt.plot(muBB, total_energy(1, 0.5, +0.5, (2/3), muBB), 'b',\
	label=r"$l=1$, $j=\frac{1}{2}$, $j_z=\pm\frac{1}{2}$, $g_j=\frac{2}{3}$, slope=$\pm\frac{1}{3}$")
plt.plot(muBB, total_energy(1, 0.5, -0.5, (2/3), muBB), 'b')
plt.plot(muBB, total_energy(1, (3/2), -0.5, (4/3), muBB),'g',\
	label=r"$l=1$, $j=\frac{3}{2}$, $j_z=\pm\frac{1}{2}$, $g_j=\frac{4}{3}$, slope=$\pm\frac{2}{3}$")
plt.plot(muBB, total_energy(1, (3/2), +0.5, (4/3), muBB),'g')
plt.plot(muBB, total_energy(1, (3/2), +(3/2), (4/3), muBB),'y',\
	label=r"$l=1$, $j=\frac{3}{2}$, $j_z=\pm\frac{3}{2}$, $g_j=\frac{4}{3}$, slope=$\pm2$")
plt.plot(muBB, total_energy(1, (3/2), -(3/2), (4/3), muBB),'y')
plt.title(r"Weak Zeeman effect, $n=2$")
plt.legend(loc=2)
plt.ylim([-5.5, 0.0])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
plt.yticks([-3.4],[r"$\approx-3.408eV$"], rotation="vertical")
plt.ylabel("E", fontsize = 16, rotation="horizontal")
plt.xlabel(r"$\mu_B B_{ext}$", fontsize = 18)
plt.show()