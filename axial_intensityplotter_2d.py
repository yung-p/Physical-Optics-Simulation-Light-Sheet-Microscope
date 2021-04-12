from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from lmfit import Model


N = 4000
size = 15 * mm
wave_length = 690 * nm
r_laser_beam = 3 * mm
f1, f2, f3 = 200* mm, 200* mm, 100 * mm

w_0 = (1.9*10**-5)/2

b = 2*np.pi*w_0**2/wave_length
print(b)

x = np.linspace(-size/2,size/2,N)/mm
y = np.linspace(-size/2,size/2,N)/mm

z = np.linspace(f3-b*4,f3+b*4, 400)
N = 4000
size = 15 * mm
wave_length = 690 * nm
r_laser_beam = 3 * mm
f1, f2, f3 = 200* mm, 200* mm, 100 * mm

w_0 = 1.9*10**-5/2

x = np.linspace(-size/2,size/2,N)
y = np.linspace(-size/2,size/2,N)

b = 2*np.pi*w_0**2/wave_length
print(2*b/(1.9*10**-5))
b_line = np.zeros(len(z))+0.0001*0.6
b_line[0:175] = np.nan
b_line[225:-1] = np.nan


w_0 = 1.9*10**-5/2
b = 2*np.pi*w_0**2/wave_length













x_axial_intensities = np.array(pd.read_csv('x_axial_intensities.csv', header=None).iloc[:, :].astype(float))
y_axial_intensities = np.array(pd.read_csv('y_axial_intensities.csv', header=None).iloc[:, :].astype(float))


y_crop = np.transpose(y_axial_intensities[:,int(N/2-30):int(N/2+30)])
x_crop = np.transpose(x_axial_intensities[:,int(N/2-600):int(N/2+600)])

## vertical lines
y1 = np.ones(y_crop.shape[0])*z[100]
y2 = np.ones(y_crop.shape[0])*z[150]
y3 = np.ones(y_crop.shape[0])*z[200]
y4 = np.ones(y_crop.shape[0])*z[250]
y5 = np.ones(y_crop.shape[0])*z[250]


#plt.plot(x[int(N/2-600):int(N/2+600)]/mm,x_crop[:,200])#


print(y_crop.shape)
print(z.shape)
print(y.shape)

#fig1, axs = plt.subplots( figsize =(10,1) )
#axs.pcolormesh(z/mm, y/mm, y_crop, shading='auto', cmap='viridis')
#axs.plot(z/mm, b_line)

def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)

x = np.linspace(-size/2,size/2,N)
y = np.linspace(-size/2,size/2,N)


fig1, axs = plt.subplots( figsize =(10,3) )
#axs.plot(np.linspace(y[int(N/2-30)], y[int(N/2+30)],len(y1))/mm,y1)

axs.set_ylabel(' y [mm]', fontweight='bold')
axs.set_xlabel('z [mm]', fontweight='bold')
amice = axs.imshow(y_crop,extent=[np.amin(z)/mm, np.amax(z)/mm, y[int(N/2-30)]/mm, y[int(N/2+30)]/mm],aspect=6)

for i in range(0,3):
    axs.axvline(z[175+25*i]/mm,linestyle='--',color = 'r',linewidth=0.8)
    axs.text(z[175+2+25*i] / mm, -0.08,  str(-1+i), color='r')

cax = fig1.add_axes([axs.get_position().x1+0.005,axs.get_position().y0,0.015,axs.get_position().height])
plt.colorbar(amice,cax=cax, label= 'intensity [arb.]' )
axs.plot(z/mm, b_line/mm, color='white',linestyle='--', linewidth=0.8)
axs.text(z[int(400/2)]/mm, 0.0001*0.8/mm, r'$b = 2\pi w_0^2/\lambda \approx $'+ str(Round_To_n(b/mm,4))+'mm', color='white')


plt.savefig('yz_section_LSM.pdf',format='pdf',dpi=400)

x = np.linspace(-size/2,size/2,N)
y = np.linspace(-size/2,size/2,N)

fig2, axs2 = plt.subplots( figsize =(10,5) )
amice2= axs2.imshow(x_crop,extent=[np.amin(z)/mm, np.amax(z)/mm, x[int(N/2-500)]/mm, x[int(N/2+500)]/mm],aspect=0.7)
cax = fig2.add_axes([axs2.get_position().x1+0.005,axs2.get_position().y0,0.015,axs2.get_position().height])
plt.colorbar(amice2,cax=cax, label= 'intensity [arb.]' )
#axs.plot(z/mm, b_line/mm, color='white',linestyle='--', linewidth=0.8)
#axs.text(z[int(400/2)]/mm, 0.0001*0.8/mm, r'$b = 2\pi w_0^2/\lambda \approx $'+ str(Round_To_n(b/mm,4)), color='white')
axs2.set_ylabel(' x [mm]', fontweight='bold')
axs2.set_xlabel('z [mm]', fontweight='bold')
plt.savefig('xz_section_LSM.pdf',format='pdf',dpi=400)





#plt.plot(z,y_crop[int(y_crop.shape[0]/2),:])
#plt.plot(z,x_crop[int(x_crop.shape[0]/2),:])


#plt.text(x_4000[a4000index_min_ls-11]/mm, 0.65, r'$\delta_{sinc} = $' + str(Round_To_n(1.22*(wave_length*f1/r_laser_beam),5)), fontsize=15, color="C4")



plt.show()


#plt.pcolormesh(x,z,y_axial_intensities)

