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

w_0 = (1.9*10**-5/2)

b = 2*np.pi*w_0**2/wave_length

x = np.linspace(-size/2,size/2,N)/mm
y = np.linspace(-size/2,size/2,N)/mm

z = np.linspace(f3-b*4,f3+b*4, 400)
N = 4000
size = 15 * mm
wave_length = 690 * nm
r_laser_beam = 3 * mm
f1, f2, f3 = 200* mm, 200* mm, 100 * mm


x = np.linspace(-size/2,size/2,N)
y = np.linspace(-size/2,size/2,N)

b = 2*np.pi*w_0**2/wave_length

b_line = np.zeros(len(z))+0.0001*0.6
b_line[0:int(400/4)] = np.nan
b_line[int(3*400/4):-1] = np.nan







x_axial_intensities = np.array(pd.read_csv('x_axial_intensities.csv', header=None).iloc[:, :].astype(float))
y_axial_intensities = np.array(pd.read_csv('y_axial_intensities.csv', header=None).iloc[:, :].astype(float))


y_crop = np.transpose(y_axial_intensities[:,int(N/2-30):int(N/2+30)])
x_crop = np.transpose(x_axial_intensities[:,int(N/2-600):int(N/2+600)])

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


gmodel = Model(gaussian)
result = gmodel.fit(y_crop[:,200], x = y[int(N/2-30):int(N/2+30)], amp=np.amax(y_crop[:,200]), cen=y[2000], wid=0.5*w_0)

width_axial = np.zeros(len(z))

def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)



fig1, axs = plt.subplots(1,3,figsize=(20,3),constrained_layout = True)
for i in range (0,3):
    result = gmodel.fit(y_crop[:, 175+25*i], x=y[int(N / 2 - 30):int(N / 2 + 30)], amp=np.amax(y_crop[:, 150+25*i]), cen=y[2000],
                        wid=w_0)
    axs[i].plot(y[int(N/2-30):int(N/2+30)]/mm, (np.amax(y_crop[:,175+25*i])**-1)*y_crop[:,175+25*i], linewidth = 0.8)
    axs[i].plot(y[int(N/2-30):int(N/2+30)]/mm, (np.amax(result.best_fit)**-1)*result.best_fit,  label='Gauss fit',linewidth = 0.8)
    axs[i].text(0.02,0.8, r'$w_{'+str(-1+i)+'} = ' +  str(Round_To_n(1.7*result.values['wid']/mm,3)) + r'$mm',color='C1')
    axs[i].set_xlabel('z[mm]',fontweight='bold')
    axs[i].set_ylabel('y[mm]',fontweight='bold')


axs[0].set_title(r'$z=f_3-b$')
axs[1].set_title(r'$z=f_3$')
axs[2].set_title(r'$z=f_3+b$')


#plt.plot(z,1+np.zeros(len(z)))

import matplotlib.pyplot as plt


for i in range(0,len(z)):
    result = gmodel.fit(y_crop[:, i], x=y[int(N / 2 - 30):int(N / 2 + 30)], amp=np.amax(y_crop[:, i]), cen=y[2000],
                        wid=0.5 * w_0)
    #print(result.values['wid'])
    width_axial[i] = result.values['wid']

w_y = np.zeros(len(z))
bar1 = np.zeros(len(z))
bar2 = np.zeros(len(z))


print(width_axial.shape)
print(np.array(np.where((width_axial/width_axial[200]) < np.sqrt(2))))

w_y[170] = np.sqrt(2)
w_y[229] = np.sqrt(2)
w_y[w_y==0] = np.nan
bar1[172:-1] = np.nan
bar2[230:-1] = np.nan

fig2,  axs = plt.subplots()
axs.plot(z/mm, width_axial/width_axial[200], linewidth = 0.8, color='C3')

axs.plot(z/mm, bar2 + np.sqrt(2), linestyle='--',linewidth = 0.8, color='k')
axs.text(98.1,1.5 , r'$w(z)/w_0 \approx \sqrt{2}$')
axs.text(99.54,1.5 , r'$b^\prime = ' + str(Round_To_n(z[229]-z[170],3)/mm) +'$mm')

axs.plot(z/mm, w_y ,  linewidth = 0.8, marker='x', markersize=5, color='b')
axs.set_xlim(0.098/mm,0.102/mm)
axs.set_ylim(0,5)
#plt.yticks(np.arange(0, 1, step=0.2))
axs.set_xlabel('z[mm]',fontweight='bold')
axs.set_ylabel(r'$w(z)/w_0$',fontweight='bold')
print((z[229]-z[170])/mm)


plt.show()