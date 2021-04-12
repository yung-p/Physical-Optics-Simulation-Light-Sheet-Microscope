
from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import pandas as pd

N = [700, 4000]
size = 15 * mm
wave_length = 690 * nm
r_laser_beam = 3 * mm
f1, f2 = 200* mm,  200* mm
distances = [0, 2*f1*1000,  4*f2*1000 ]
N_f = (r_laser_beam**2)/(wave_length*f1)

def asm_sim_4f_system(N, size, wave_length, r_laser_beam, f1, f2):
    focal_lengths = np.array((f1, f2))
    d_s = [focal_lengths[0], focal_lengths[0] + focal_lengths[1], focal_lengths[1]]

    n_d1 = 2
    n_d2 = 3
    n_d3 = 2

    d_1 = np.linspace(0, d_s[0], n_d1)
    d_2 = np.linspace(0, d_s[1], n_d2)
    d_3 = np.linspace(0, d_s[2], n_d3)


    # preallocation of fields
    fields_1 = np.zeros((N, N, n_d1)).astype(complex)
    fields_2 = np.zeros((N, N, n_d2)).astype(complex)
    fields_3 = np.zeros((N, N, n_d3)).astype(complex)

    F_0 = Begin(size, wave_length, N)  # initial field
   # F_0 = (F_0.field*np.exp( 1j *2*np.pi*delta_x/wave_length)).Field
    F_0 = CircAperture(F_0, r_laser_beam, x_shift=0, y_shift=0)

    for i in range(0, n_d1):
        fields_1[:, :, i] = Forvard(F_0, d_1[i]).field

    P_1 = Forvard(F_0, d_1[-1])
    L_1 = Lens(P_1, f1, x_shift=0.0, y_shift=0.0)  # second normal lens

    for i in range(0, n_d2):
        fields_2[:, :, i] = Forvard(L_1, d_2[i]).field

    P_2 = Forvard(L_1,d_2[-1])
    L_2 = Lens(P_2, f2, x_shift=0.0, y_shift=0.0)

    for i in range(0, n_d3):
        fields_3[:, :, i] = Forvard(L_2, d_3[i]).field

    intensity_1 = abs(fields_1) ** 2
    intensity_2 = abs(fields_2) ** 2
    intensity_3 = abs(fields_3) ** 2

    phases_1 = np.angle(fields_1)
    phases_2 = np.angle(fields_2)
    phases_3 = np.angle(fields_3)

    return intensity_1, intensity_2, intensity_3, phases_1, phases_2, phases_3

def fresnel_sim_4f_system(N, size, wave_length, r_laser_beam, f1, f2):
    focal_lengths = np.array((f1, f2))
    d_s = [focal_lengths[0], focal_lengths[0] + focal_lengths[1], focal_lengths[1]]

    n_d1 = 2
    n_d2 = 3
    n_d3 = 2

    d_1 = np.linspace(0, d_s[0], n_d1)
    d_2 = np.linspace(0, d_s[1], n_d2)
    d_3 = np.linspace(0, d_s[2], n_d3)


    # preallocation of fields
    fields_1 = np.zeros((N, N, n_d1)).astype(complex)
    fields_2 = np.zeros((N, N, n_d2)).astype(complex)
    fields_3 = np.zeros((N, N, n_d3)).astype(complex)

    F_0 = Begin(size, wave_length, N)  # initial field
    F_0 = CircAperture(F_0, r_laser_beam, x_shift=0.0, y_shift=0.0)

    for i in range(0, n_d1):
        fields_1[:, :, i] = Fresnel(F_0, d_1[i]).field

    P_1 = Fresnel(F_0, d_1[-1])
    L_1 = Lens(P_1, f1, x_shift=0.0, y_shift=0.0)  # second normal lens

    for i in range(0, n_d2):
        fields_2[:, :, i] = Fresnel(L_1, d_2[i]).field

    P_2 = Fresnel(L_1,d_2[-1])
    L_2 = Lens(P_2, f2, x_shift=0.0, y_shift=0.0)

    for i in range(0, n_d3):
        fields_3[:, :, i] = Fresnel(L_2, d_3[i]).field

    intensity_1 = abs(fields_1) ** 2
    intensity_2 = abs(fields_2) ** 2
    intensity_3 = abs(fields_3) ** 2

    phases_1 = np.angle(fields_1)
    phases_2 = np.angle(fields_2)
    phases_3 = np.angle(fields_3)

    return intensity_1, intensity_2, intensity_3, phases_1, phases_2, phases_3

def airy_cross_section(r):

    w = r_laser_beam
    arg= 2*np.pi*w/(wave_length*f1) * r
    airy_cutout =  (sp.jv(1, arg)/arg) ** 2

    return airy_cutout



asm_cs_700 =  np.array(pd.read_csv('asm_cross_section_Nis' + str(700) + '.csv', header=None).iloc[:, :].astype(float))
fresnel_cs_700 =  np.array(pd.read_csv('fresnel_cross_section_Nis' + str(700) + '.csv', header=None).iloc[:, :].astype(float))
asm_cs_4000 =  np.array(pd.read_csv('asm_cross_section_Nis' + str(4000) + '.csv', header=None).iloc[:, :].astype(float))
fresnel_cs_4000  =  np.array(pd.read_csv('fresnel_cross_section_Nis' + str(4000) + '.csv', header=None).iloc[:, :].astype(float))

N_jinc = 50000
x_jinc = np.linspace(-size/2,size/2,N_jinc)
jinc = airy_cross_section(x_jinc)

x_700 = np.linspace(-size/2,size/2,700)
x_4000 = np.linspace(-size/2,size/2,4000)

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

asm_width_peak700, asm_peak_for_plotting700, a700index_min_ls, a700index_min_rs, index_max_a700 = width_of_peak_finder(asm_cs_700, x_700 )
asm_width_peak4000, asm_peak_for_plotting4000, a4000index_min_ls, a4000index_min_rs, index_max_a4000 = width_of_peak_finder(asm_cs_4000, x_4000 )
fresnel_width_peak700, fresnel_peak_for_plotting700,  f700index_min_ls, f700index_min_rs, index_max_fr700 = width_of_peak_finder(fresnel_cs_700, x_700 )
fresnel_width_peak4000, fresnel_peak_for_plotting4000, f4000index_min_ls, f4000index_min_rs,index_max_fr4000 = width_of_peak_finder(fresnel_cs_4000, x_4000 )
def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


print(asm_cs_4000-fresnel_cs_4000)


fig, ax1 = plt.subplots( figsize=(9,9))
ax1.plot(x_700/mm ,(np.amax(asm_cs_700)**-1)*asm_cs_700,linewidth=0.8,label=r'ASM, $N = 700$',linestyle='--')
ax1.plot(x_700/mm,(np.amax(fresnel_cs_700)**-1)*fresnel_cs_700,linewidth=0.8,label=r'Fresnel, $N = 700$',linestyle='--')

#ax1[0].plot(x_700/mm  , asm_peak_for_plotting700 ,linewidth=0.8,label=r'asm width peak 700')
ax1.plot(x_4000/mm,(np.amax(asm_cs_4000)**-1)*asm_cs_4000,linewidth=0.8,label=r'ASM, $N = 4000$')
ax1.plot(x_4000/mm,(np.amax(fresnel_cs_4000)**-1)*fresnel_cs_4000,linewidth=0.8,label=r'Fresnel, $N = 4000$')
ax1.plot(x_jinc/mm,(np.amax(jinc) **-1) * jinc, linewidth=0.8, label=r'jinc$(kar/f)$')

print(x_4000[a4000index_min_ls])
#ax1.plot(np.ones(4000)*x_4000[a4000index_min_ls]/mm, np.linspace(-0.2,1.2,4000)  ,linewidth=0.8,label=r'asm width peak 4000', )
#ax1.plot(np.ones(4000)*x_4000[a4000index_min_rs]/mm, np.linspace(-0.2,1.2,4000)  ,linewidth=0.8,label=r'asm width peak 4000')


ax1.set_xlim([ -0.7*1.20583361e-04/mm, 0.7*1.20583361e-04/mm])
ax1.set_ylim([ -0.05, 1.05])
ax1.set_ylabel(' Amplitude [arb.]', fontweight='bold')
ax1.set_xlabel('x [mm]', fontweight='bold')

plt.text(x_4000[a4000index_min_ls-14]/mm, 0.75, r'$\delta_{F,4000} = $' + str(Round_To_n((x_4000[f4000index_min_rs]-x_4000[f4000index_min_ls])/mm,5)) + ' mm', fontsize=15, color="C3")
plt.text(x_4000[a4000index_min_ls-14]/mm, 0.7, r'$\delta_{A,4000} = $' + str(Round_To_n((x_4000[a4000index_min_rs]-x_4000[a4000index_min_ls])/mm,5))+ ' mm', fontsize=15, color="C2")
plt.text(x_4000[a4000index_min_ls-14]/mm, 0.65, r'$\delta_{jinc} = $' + str(Round_To_n(1.22*(wave_length*f1/r_laser_beam)/mm,5)) + ' mm', fontsize=15, color="C4")



plt.legend()

plt.savefig('2f_4fsystem_cross_sections.png', format = 'png', dpi=400)

#plt.savefig('2fplane_cross_sectionN[i]]is4000.png', dpi=1200)
plt.show()

