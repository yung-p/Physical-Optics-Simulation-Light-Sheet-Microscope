
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


    E_0f = double_Integral(-size/2, size/2, -size/2,size/2, N, N,intensity_1[:,:,0])
    E_2f = double_Integral(-size/2, size/2, -size/2,size/2, N, N, intensity_2[:,:,1])
    E_4f = double_Integral(-size/2, size/2, -size/2,size/2, N, N,intensity_3[:,:, -1])


    phases_1 = np.angle(fields_1)
    phases_2 = np.angle(fields_2)
    phases_3 = np.angle(fields_3)

    return intensity_1, intensity_2, intensity_3, phases_1, phases_2, phases_3, E_0f, E_2f, E_4f

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

    E_0f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_1[:, :, 0])
    E_2f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_2[:, :, 1])
    E_4f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_3[:, :, -1])

    phases_1 = np.angle(fields_1)
    phases_2 = np.angle(fields_2)
    phases_3 = np.angle(fields_3)

    return intensity_1, intensity_2, intensity_3, phases_1, phases_2, phases_3, E_0f, E_2f, E_4f

def airy_cross_section(r):

    w = r_laser_beam
    arg= 2*np.pi*w/(wave_length*f1) * r
    airy_cutout =  (sp.jv(1, arg)/arg) ** 2

    return airy_cutout


for i in range(0,len(N)):

    x = np.linspace(-size / 2, size / 2, N[i])
    y = np.linspace(-size / 2, size / 2, N[i])
    delta_x = size/N[i]

    intensity_1, intensity_2, intensity_3,\
    phases_1, phases_2, phases_3, E_0f, E_2f, E_4f= asm_sim_4f_system(N[i], size, wave_length, r_laser_beam, f1, f2)

    fig1, axs = plt.subplots(2, 3, sharex='col', gridspec_kw={'hspace': 0.05 ,'wspace':0.3}, figsize=(12, 7))
    axs[0, 0].pcolormesh(x/mm, y/mm, intensity_1[:, :, 0], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')
    axs[0, 1].pcolormesh(x/mm, y/mm, intensity_2[:, :, 1], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')
    axs[0, 1].set_xlim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])
    axs[0, 1].set_ylim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])
    amice1 = axs[0, 2].pcolormesh(x/mm, y/mm, intensity_3[:, :, -1], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')


    def Round_To_n(x, n):
        return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


    axs[0, 0].text(size/7/mm,size/2.23/mm, r'$E_{0f} / E_0 ='+str(Round_To_n(E_0f/E_0f,5))+ '$', color='white')
    axs[0, 1].text(0, 2*12*1.20583361e-04/mm / 2.3 , r'$E_{2f} / E_0 =' + str(Round_To_n(E_2f/E_0f,5)) + '$', color='white')
    axs[0, 2].text(0, size / 2.3 / mm, r'$E_{4f} / E_0 =' + str(Round_To_n(E_4f/E_0f,5)) + '$' ,color='white')

    #fig1.colorbar(amice1, ax=axs[0, 2], aspect=70)

    axs[1, 0].plot(x/mm, intensity_1[:, :, 0][int(N[i] / 2), :], linewidth=0.8)
    axs[1, 1].plot(x/mm,intensity_2[:, :, 1][int(N[i] / 2), :], linewidth=0.8)
    axs[1, 2].plot(x/mm, intensity_3[:, :, -1][int(N[i] / 2), :], linewidth=0.8)
    axs[0, 0].set_ylabel('Intensity Patterns \n y[mm]', fontweight='bold')
    print('ben nu hier')
    for j in range(0, 3):
        axs[0, j].set_title(str(distances[j]) + ' mm', fontsize=11)
        axs[1, j].set_xlabel('x [mm]', fontweight='bold');
    axs[1, 0].yaxis.set_visible(True)
    axs[1, 0].set_ylabel('Amplitude [arb.]', fontweight='bold')
    print('ben nu hier')
    #fig1.suptitle('4f system, Intensities (ASM) in planes z = 0, 400mm and 800mm,  \n N[i]] = ' + str(
        # N[i]]) + ', grid size = 15mm ', weight='bold', fontsize='12')
    plt.savefig('4f_asm_int_N'+ str(N[i]) +'.pdf', format='pdf', dpi=400)
    print('ben nu hier')

    np.savetxt('asm_cross_section_Nis' + str(N[i]) + '.csv', intensity_2[:, :, 1][int(N[i] / 2), :], delimiter=',')

    fig2, axs = plt.subplots(2, 3, sharex='col', gridspec_kw={'hspace': 0.05}, figsize=(12, 7),constrained_layout = True)
    axs[0, 0].pcolormesh(x/mm, y/mm, phases_1[:,:,0], shading='auto',
                         vmin=-np.pi, vmax=np.pi, cmap='rainbow')
    axs[0, 1].pcolormesh(x/mm, y/mm, phases_2[:,:,1], shading='auto',
                     vmin=-np.pi, vmax=np.pi, cmap='rainbow')
    axs[0, 1].set_xlim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])
    axs[0, 1].set_ylim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])

    amice2 = axs[0, 2].pcolormesh(x/mm, y/mm, phases_3[:,:,-1], shading='auto',
                     vmin=-np.pi, vmax=np.pi, cmap='rainbow')
    fig2.colorbar(amice2, ax=axs[0, 2], aspect = 60,label='phase [rad.]')



    axs[1, 0].plot(x/mm, phases_1[:,:,0][int(N[i] / 2), :], linewidth=0.8);
    axs[1, 1].plot(x/mm,  phases_2[:,:,1][int(N[i] / 2), :], linewidth=0.8)
    axs[1, 2].plot(x/mm, phases_3[:,:,-1][int(N[i] / 2), :], linewidth=0.8);
    axs[0, 0].set_ylabel('Phase patterns \n y[mm]', fontweight='bold')
    for j in range(0, 3):
        axs[0, j].set_title(str(distances[j]) + ' mm', fontsize=11)
    axs[1, j].set_xlabel('x [mm]', fontweight='bold');
    axs[1, 0].yaxis.set_visible(True)
    axs[1, 0].set_ylabel('Phase [rad.]', fontweight='bold')

    plt.savefig('4f_asm_phase_N'+ str(N[i]) +'.pdf', format='pdf', dpi=400)
    print('ben nu hier')

    intensity_1, intensity_2, intensity_3,\
    phases_1, phases_2, phases_3, E_0f, E_2f, E_4f = fresnel_sim_4f_system(N[i], size, wave_length, r_laser_beam, f1, f2)

    fig3, axs = plt.subplots(2, 3, sharex='col', gridspec_kw={'hspace': 0.05 ,'wspace':0.3}, figsize=(12, 7))
    axs[0, 0].pcolormesh(x/mm, y/mm, intensity_1[:, :, 0], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')
    axs[0, 1].pcolormesh(x/mm, y/mm, intensity_2[:, :, 1], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')
    axs[0, 1].set_xlim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])
    axs[0, 1].set_ylim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])
    amice3 = axs[0, 2].pcolormesh(x/mm, y/mm, intensity_3[:, :, -1], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')
    axs[0, 0].text(0, size / 2.3 / mm, r'$E_{0f} / E_0 =' + str(Round_To_n(E_0f/E_0f,5)) + '$',color='white')
    axs[0, 1].text(0, 2*12*1.20583361e-04/mm / 2.3, r'$E_{2f} / E_0 =' + str(Round_To_n(E_2f/E_0f,5)) + '$',color='white')
    axs[0, 2].text(0, size / 2.3 / mm, r'$E_{4f} / E_0 =' + str(Round_To_n(E_4f/E_0f,5)) + '$',color='white')

    #fig3.colorbar(amice3, ax=axs[0, 2],aspect=70)

    axs[1, 0].plot(x/mm, intensity_1[:, :, 0][int(N[i] / 2), :], linewidth=0.8)
    axs[1, 1].plot(x/mm,intensity_2[:, :, 1][int(N[i] / 2), :], linewidth=0.8)
    axs[1, 2].plot(x/mm, intensity_3[:, :, -1][int(N[i] / 2), :], linewidth=0.8)
    axs[0, 0].set_ylabel('Intensity Patterns \n y[mm]', fontweight='bold')
    print('ben nu hier')
    for j in range(0, 3):
        axs[0, j].set_title(str(distances[j]) + ' mm', fontsize=11)
        axs[1, j].set_xlabel('x [mm]', fontweight='bold');
    axs[1, 0].yaxis.set_visible(True)
    axs[1, 0].set_ylabel('Amplitude [arb.]', fontweight='bold')
    print('ben nu hier')
    #fig1.suptitle('4f system, Intensities (ASM) in planes z = 0, 400mm and 800mm,  \n N[i]] = ' + str(
        # N[i]]) + ', grid size = 15mm ', weight='bold', fontsize='12')
    plt.savefig('4f_fresnel_int_N'+ str(N[i]) +'.pdf', format='pdf', dpi=400)

    np.savetxt('fresnel_cross_section_Nis' + str(N[i]) + '.csv', intensity_2[:, :, 1][int(N[i] / 2), :], delimiter=',')

    fig4, axs = plt.subplots(2, 3, sharex='col', gridspec_kw={'hspace': 0.05}, figsize=(12, 7),constrained_layout = True)
    axs[0, 0].pcolormesh(x/mm, y/mm, phases_1[:,:,0], shading='auto',
                         vmin=-np.pi, vmax=np.pi, cmap='rainbow')
    axs[0, 1].pcolormesh(x/mm, y/mm, phases_2[:,:,1], shading='auto',
                     vmin=-np.pi, vmax=np.pi, cmap='rainbow')
    axs[0, 1].set_xlim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])
    axs[0, 1].set_ylim([ -12*1.20583361e-04/mm, 12*1.20583361e-04/mm])
    amice4 = axs[0, 2].pcolormesh(x/mm, y/mm, phases_3[:,:,-1], shading='auto',
                     vmin=-np.pi, vmax=np.pi, cmap='rainbow')
    fig4.colorbar(amice4, ax=axs[0, 2],aspect=60 ,label='phase [rad.]')

    axs[1, 0].plot(x/mm, phases_1[:,:,0][int(N[i] / 2), :], linewidth=0.8);
    axs[1, 1].plot(x/mm,  phases_2[:,:,1][int(N[i] / 2), :], linewidth=0.8)
    axs[1, 2].plot(x/mm, phases_3[:,:,-1][int(N[i] / 2), :], linewidth=0.8);
    axs[0, 0].set_ylabel('Phase patterns \n y[mm]', fontweight='bold')
    for j in range(0, 3):
        axs[0, j].set_title(str(distances[j]) + 'mm', fontsize=11)
    axs[1, j].set_xlabel('x [mm]', fontweight='bold');
    axs[1, 0].yaxis.set_visible(True)
    axs[1, 0].set_ylabel('Phase [rad.]', fontweight='bold')
    #fig2.suptitle('4f system, Phases (ASM) in planes z = 0, 400mm and 800,  \n N[i]] = ' + str(
                #    N[i]]) + ', grid size = 15mm ', weight='bold',
                #            fontsize='12')
    plt.savefig('4f_fresnel_phase_N'+ str(N[i]) +'.pdf', format='pdf', dpi=400)
    print('ben nu hier')



#plt.savefig('2fplane_cross_sectionN[i]]is4000.png', dpi=1200)

