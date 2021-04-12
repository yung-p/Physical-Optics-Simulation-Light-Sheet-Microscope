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




def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


x_crop[:,200]


fig1, axs = plt.subplots(1,3,figsize=(20,3),constrained_layout = True)
for i in range (0,3):
    #result = gmodel.fit(y_crop[:, 175+25*i], x=y[int(N / 2 - 30):int(N / 2 + 30)], amp=np.amax(y_crop[:, 150+25*i]), cen=y[2000],
                   #     wid=w_0)
    axs[i].plot(x[int(N/2-600):int(N/2+600)]/mm, x_crop[:,175+25*i], linewidth = 0.8, color='C3')
    #axs[i].plot(x[int(N/2-600):int(N/2+600)]/mm, (np.amax(result.best_fit)**-1)*result.best_fit,  label='Gauss fit',linewidth = 0.8)
    #axs[i].text(0.02,0.8, r'$w_{'+str(-1+i)+'} = ' +  str(Round_To_n(1.7*result.values['wid']/mm,3)) + r'$mm',color='C1')
    axs[i].set_xlabel('z[mm]',fontweight='bold')
    axs[i].set_ylabel('y[mm]',fontweight='bold')


axs[0].set_title(r'$z=f_3-b$')
axs[1].set_title(r'$z=f_3$')
axs[2].set_title(r'$z=f_3+b$')
plt.show()