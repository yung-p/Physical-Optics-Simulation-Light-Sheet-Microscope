from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import pandas as pd

N = 4000
size = 15 * mm
wave_length = 690 * nm
r_laser_beam = 3 * mm
f1, f2, f3 = 200* mm, 200* mm, 100 * mm
distances = np.array([ 2*f1, 2*f1 + 2*f2, 2*f1 + 2*f2 + 2*f3])/mm


zeros_space_sinc = wave_length*f1/r_laser_beam
min_N = size/(zeros_space_sinc/2)


def sinc_func_f1(r):
    w = r_laser_beam
    arg = (2 * np.pi * w * r) / (wave_length * f1)
    sinc_func = (np.sin(arg)/ arg) ** 2
    return sinc_func


def sinc_func_image_plane(r):
    w = r_laser_beam
    r_prime = f2/f3*r
    arg = (2 * np.pi * w * r_prime) / (wave_length * f1)
    sinc_func = (np.sin(arg)/ arg) ** 2
    return sinc_func


def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):

    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    A_Internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (np.sum(A_Internal)\
                + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))\
                + 0.25 * (A_ul + A_ur + A_dl + A_dr))



asm_cs_4000_x =  np.array(pd.read_csv('LSM_cs_x_direction4000.csv', header=None).iloc[:, :].astype(float))
asm_cs_4000_y =  np.array(pd.read_csv('LSM_cs_y_direction4000.csv', header=None).iloc[:, :].astype(float))

fr_cs_4000_x =  np.array(pd.read_csv('fr_LSM_cs_x_direction4000.csv', header=None).iloc[:, :].astype(float))
fr_cs_4000_y =  np.array(pd.read_csv('fr_LSM_cs_y_direction4000.csv', header=None).iloc[:, :].astype(float))
#print(np.array(np.where(asm_cs_4000_x == np.amax(asm_cs_4000_x)))[0])
#print(np.array(np.where(asm_cs_4000_y == np.amax(asm_cs_4000_y)))[0])
#print(np.array(np.where(fr_cs_4000_x == np.amax(fr_cs_4000_x)))[0])
#print(np.array(np.where(fr_cs_4000_y == np.amax(fr_cs_4000_y)))[0])

N_sinc = 50000
x_sinc = np.linspace(-size/2,size/2,N_sinc)
sinc = sinc_func_image_plane(x_sinc)

x_4000 = np.linspace(-size/2,size/2,4000)
y_4000 = np.linspace(-size/2,size/2,4000)


def width_of_peak_finder(cross_section,x_N ):
    index_max = int(np.array(np.where(cross_section == np.amax(cross_section)))[0])
    i = index_max
    while cross_section[i]-cross_section[i+1] > 0:
        i = i+1
        if cross_section[i]-cross_section[i+1] < 0:
            index_min_rs = i
    j = index_max
    while cross_section[j]-cross_section[j-1] > 0:
        j = j-1
        if cross_section[j]-cross_section[j-1] < 0:
            index_min_ls = j+1
    peak_for_plot = np.zeros(len(x_N))
    peak_for_plot[index_min_ls:index_min_rs] = 0.5
    width_peak = (x_N[index_min_rs] - x_N[index_min_ls])
    return width_peak, peak_for_plot, index_min_ls, index_min_rs, index_max

#asm_peak_width_x, asm_peak_for_plotting_x, asm_min_ls_x, asm_min_rs_x, index_max_asm_x = width_of_peak_finder(asm_cs_4000_x, x_4000 )
asm_peak_width_y, asm_peak_for_plotting_y, asm_min_ls_y, asm_min_rs_y, index_max_asm_y = width_of_peak_finder(asm_cs_4000_y, y_4000 )

fr_peak_width_x, fr_peak_for_plotting_x, fr_min_ls_x, fr_min_rs_x, index_max_fr_x = width_of_peak_finder(fr_cs_4000_x, y_4000 )
fr_peak_width_y, fr_peak_for_plotting_y, fr_min_ls_y, fr_min_rs_y, index_max_fr_y = width_of_peak_finder(fr_cs_4000_y, y_4000 )


def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)





fig, ax1 = plt.subplots( figsize=(7,7))
ax1.plot(y_4000/mm ,(np.amax(asm_cs_4000_y)**-1)*asm_cs_4000_y,linewidth=0.8,label=r'$ASM,N = 4000$',linestyle='--')
ax1.plot(y_4000/mm ,(np.amax(fr_cs_4000_y)**-1)*fr_cs_4000_y,linewidth=0.8,label=r'$Fresnel,N = 4000$',linestyle='--')
ax1.plot(x_sinc/mm,(np.amax(sinc) **-1) * sinc, linewidth=0.8, label=r'sinc$(kay/f)$')
ax1.set_xlim([ -8*((2 * np.pi * r_laser_beam  ) / (wave_length * f1))**-1/mm, 8*((2 * np.pi * r_laser_beam  ) / (wave_length * f1))**-1/mm])
ax1.set_ylim([ -0.05, 1.05])
ax1.set_ylabel(' Amplitude [arb.]', fontweight='bold')
ax1.set_xlabel('y [mm]', fontweight='bold')
#ax1.set_title('', fontweight='bold')
plt.text(x_4000[asm_min_ls_y-13]/mm, 0.75, r'$\delta_{F} = $' + str(Round_To_n((x_4000[asm_min_rs_y]-x_4000[asm_min_ls_y])/mm,5)) + ' mm', fontsize=13, color="C1")
plt.text(x_4000[asm_min_ls_y-13]/mm, 0.7, r'$\delta_{A} = $' + str(Round_To_n((x_4000[asm_min_rs_y]-x_4000[asm_min_ls_y])/mm,5)) + ' mm', fontsize=13, color="C0")
plt.text(x_4000[asm_min_ls_y-13]/mm, 0.65, r'$\delta_{sinc} = $' + str(Round_To_n((((2 * np.pi * r_laser_beam  ) *0.5/ (wave_length * f1))**-1)/mm,5)) + ' mm', fontsize=15, color="C2")
plt.legend()
plt.savefig('image_planeLSM_cross_sections.png', format = 'png', dpi=400)

plt.show()



print(print(np.where(asm_cs_4000_y==np.amax(asm_cs_4000_y))))
print(size/4000/2)
print(x_4000[1999])
#plt.savefig('2fplane_cross_sectionN[i]]is4000.png', dpi=1200)



