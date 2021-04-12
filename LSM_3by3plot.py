from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

N = 4000
size = 15 * mm
wave_length = 690 * nm
r_laser_beam = 3 * mm
f1, f2, f3 = 200* mm, 200* mm, 100 * mm
distances = np.array([ 2*f1, 2*f1 + 2*f2, 2*f1 + 2*f2 + 2*f3])/mm
print(distances[-1])


zeros_space_sinc = wave_length*f1/r_laser_beam
min_N = size/(zeros_space_sinc/2)
print(zeros_space_sinc/2)
print(min_N)

def airy_cutout(r):
    w = r_laser_beam
    arg= 2*np.pi*w/(wave_length*f1) * r
    airy_cutout =(np.pi*w**2/(wave_length*f1)**2) * (2*sp.jv(1, arg)/arg) **2
    return airy_cutout

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


x_sinc = np.linspace(-size / 2, size / 2, 4*N)
y_sinc = np.linspace(-size / 2, size / 2, 4*N)
sinc_func_image_plane = sinc_func_image_plane(y_sinc)
x = np.linspace(-size / 2, size / 2, N)
y = np.linspace(-size / 2, size / 2, N)


def lsm_sim(N, size, wave_length, r_laser_beam, f1, f2,f3):
    focal_lengths = np.array((f1, f2, f3))
    d_s = [focal_lengths[0], focal_lengths[0] + focal_lengths[1], focal_lengths[1] + focal_lengths[2], focal_lengths[2] ]

    n_d1 = 2
    n_d2 = 3
    n_d3 = 3
    n_d4 = 2

    d_1 = np.linspace(0, d_s[0], n_d1)
    d_2 = np.linspace(0, d_s[1], n_d2)
    d_3 = np.linspace(0, d_s[2], n_d3)
    d_4 = np.linspace(0, d_s[3], n_d4)



    # preallocation of fields
    fields_1 = np.zeros((N, N, n_d1)).astype(complex)
    fields_2 = np.zeros((N, N, n_d2)).astype(complex)
    fields_3 = np.zeros((N, N, n_d3)).astype(complex)
    fields_4 = np.zeros((N, N, n_d4)).astype(complex)

    F_0 = Begin(size, wave_length, N)  # initial field
    F_0 = CircAperture(F_0, r_laser_beam, x_shift=0.0, y_shift=0.0)

    for i in range(0, n_d1):
        fields_1[:, :, i] = Forvard(F_0, d_1[i]).field

    P_1 = Forvard(F_0, d_1[-1])
    L_1 = CylindricalLens(P_1, f1, x_shift=0.0, y_shift=0.0, angle=90*deg)  # second normal lens

    for i in range(0, n_d2):
        fields_2[:, :, i] = Forvard(L_1, d_2[i]).field

    P_2 = Forvard(L_1,d_2[-1])
    L_2 = Lens(P_2, f2, x_shift=0.0, y_shift=0.0)

    for i in range(0, n_d3):
        fields_3[:, :, i] = Forvard(L_2, d_3[i]).field

    P_3 = Forvard(L_2, d_3[-1])
    L_3 = Lens(P_3, f3, x_shift=0.0, y_shift=0.0)

    for i in range(0,n_d4):
        fields_4[:,:,i] = Forvard(L_3,d_4[i]).field

    intensity_1 = abs(fields_1) ** 2
    intensity_2 = abs(fields_2) ** 2
    intensity_3 = abs(fields_3) ** 2
    intensity_4 = abs(fields_4) ** 2

    E_0f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_1[:, :, 0])
    E_2f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_2[:, :, 1])
    E_4f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_3[:, :, -1])
    E_6f =  double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_4[:, :, -1])

    OL_field = np.savetxt('OL_field_Nis' + str(N) + '.csv', intensity_4[:, :, -1], delimiter=',')

    phases_1 = np.angle(fields_1)
    phases_2 = np.angle(fields_2)
    phases_3 = np.angle(fields_3)
    phases_4 = np.angle(fields_4)


    #two_f_energy = print(Intensity(Forvard(L_1, d_2[1])).sum())

    return intensity_1, intensity_2, intensity_3,  intensity_4, phases_1, phases_2, phases_3, phases_4, E_0f, E_2f, E_4f, E_6f


def fresnel_lsm_sim(N, size, wave_length, r_laser_beam, f1, f2,f3):
    focal_lengths = np.array((f1, f2, f3))
    d_s = [focal_lengths[0], focal_lengths[0] + focal_lengths[1], focal_lengths[1] + focal_lengths[2], focal_lengths[2] ]

    n_d1 = 2
    n_d2 = 3
    n_d3 = 3
    n_d4 = 2

    d_1 = np.linspace(0, d_s[0], n_d1)
    d_2 = np.linspace(0, d_s[1], n_d2)
    d_3 = np.linspace(0, d_s[2], n_d3)
    d_4 = np.linspace(0, d_s[3], n_d4)



    # preallocation of fields
    fields_1 = np.zeros((N, N, n_d1)).astype(complex)
    fields_2 = np.zeros((N, N, n_d2)).astype(complex)
    fields_3 = np.zeros((N, N, n_d3)).astype(complex)
    fields_4 = np.zeros((N, N, n_d4)).astype(complex)

    F_0 = Begin(size, wave_length, N)  # initial field
    F_0 = CircAperture(F_0, r_laser_beam, x_shift=0.0, y_shift=0.0)

    for i in range(0, n_d1):
        fields_1[:, :, i] = Fresnel(F_0, d_1[i]).field

    P_1 = Fresnel(F_0, d_1[-1])
    L_1 = CylindricalLens(P_1, f1, x_shift=0.0, y_shift=0.0, angle=90*deg)  # second normal lens

    for i in range(0, n_d2):
        fields_2[:, :, i] = Fresnel(L_1, d_2[i]).field

    P_2 = Fresnel(L_1,d_2[-1])
    L_2 = Lens(P_2, f2, x_shift=0.0, y_shift=0.0)

    for i in range(0, n_d3):
        fields_3[:, :, i] = Fresnel(L_2, d_3[i]).field

    P_3 = Fresnel(L_2, d_3[-1])
    L_3 = Lens(P_3, f3, x_shift=0.0, y_shift=0.0)

    for i in range(0,n_d4):
        fields_4[:,:,i] = Fresnel(L_3,d_4[i]).field

    intensity_1 = abs(fields_1) ** 2
    intensity_2 = abs(fields_2) ** 2
    intensity_3 = abs(fields_3) ** 2
    intensity_4 = abs(fields_4) ** 2

    E_0f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_1[:, :, 0])
    E_2f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_2[:, :, 1])
    E_4f = double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_3[:, :, -1])
    E_6f =  double_Integral(-size / 2, size / 2, -size / 2, size / 2, N, N, intensity_4[:, :, -1])

    OL_field = np.savetxt('OL_field_Nis' + str(N) + '.csv', L_3.field, delimiter=',')

    phases_1 = np.angle(fields_1)
    phases_2 = np.angle(fields_2)
    phases_3 = np.angle(fields_3)
    phases_4 = np.angle(fields_4)


    #two_f_energy = print(Intensity(Forvard(L_1, d_2[1])).sum())

    return intensity_1, intensity_2, intensity_3,  intensity_4, phases_1, phases_2, phases_3, phases_4, E_0f, E_2f, E_4f, E_6f




def Round_To_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


intensity_1, intensity_2, intensity_3, intensity_4,\
phases_1, phases_2, phases_3, phases_4 , E_0f, E_2f, E_4f, E_6f = lsm_sim(N, size, wave_length, r_laser_beam, f1, f2, f3)


fig1, axs = plt.subplots(3, 3,  gridspec_kw={'hspace': 0.23, 'wspace':0.3},   figsize=(15, 15), constrained_layout = True)
axs[0, 0].pcolormesh(x/mm, y/mm, intensity_2[:,:,1], shading='auto',
               vmin=-1, vmax=1, cmap='twilight')
axs[0, 1].pcolormesh(x/mm, y/mm, intensity_3[:, :, 1], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')
axs[0, 2].pcolormesh(x/mm, y/mm, intensity_4[:, :, -1], shading='auto',
                     vmin=-1, vmax=1, cmap='twilight')
axs[0, 0].text(0, size / 2.23 / mm, r'$E_{2f} / E_0 =' + str(Round_To_n(E_2f / E_0f, 5)) + '$',color='white')

axs[0, 1].text(0, size / 2.3 / mm, r'$E_{4f} / E_0 =' + str(Round_To_n(E_4f / E_0f, 5)) + '$', color='white')
axs[0, 2].text(0, size / 2.3 / mm, r'$E_{4f} / E_0 =' + str(Round_To_n(E_6f / E_0f, 5)) + '$', color='white')

axs[1, 0].plot(x/mm, intensity_2[:, :, 1][int(N / 2), :], linewidth=0.8)
axs[1, 1].plot(x/mm, intensity_3[:, :, 1][int(N / 2), :], linewidth=0.8)
axs[1, 2].plot(x/mm, intensity_4[:, :, -1][int(N / 2), :], linewidth=0.8)

axs[2, 0].plot(x/mm, intensity_2[:, :, 1][:,int(N / 2)],color='r', linewidth=0.8)
axs[2, 0].set_xlim([ -700*wave_length*f1/mm, 700*wave_length*f1/mm])
axs[2, 1].plot(x/mm, intensity_3[:, :, 1][:,int(N / 2)],color='r', linewidth=0.8)
axs[2, 2].plot(x/mm, intensity_4[:, :, -1][:,int(N / 2)],color='r', linewidth=0.8)
axs[2, 2].set_xlim([ -700*wave_length*f1*(f2/f3)/mm, 700*wave_length*f1*(f2/f3)/mm])
axs[0, 0].set_ylabel('Intensity Patterns \n y[mm]', fontweight='bold')

np.savetxt('LSM_cs_y_direction' + str(N) + '.csv', intensity_4[:, :, -1][: ,int(N / 2)], delimiter=',')
np.savetxt('LSM_cs_x_direction' + str(N) + '.csv', intensity_4[:, :, -1][int(N / 2), :], delimiter=',')

for j in range(0, 3):
    axs[0, j].set_title(str(distances[j]) + ' mm', fontsize=11)
    axs[1, j].set_xlabel('x [mm]', fontweight='bold');
    axs[2, j].set_xlabel('y [mm]', fontweight='bold');
axs[1, 0].yaxis.set_visible(True)
axs[1, 0].set_ylabel('x Amplitude [arb.]', fontweight='bold')
axs[2, 0].set_ylabel('y Amplitude [arb.]', fontweight='bold')

print('ben nu hier')
#fig1.suptitle('4f system, Intensities (ASM) in planes z = 0, 400mm and 800mm,  \n N = ' + str(
   # N) + ', grid size = 15mm ', weight='bold', fontsize='12')
plt.savefig('LSM_intensity_N4000.pdf', format='pdf', dpi=400)
print('ben nu hier')

fig2, axs = plt.subplots(3, 3,  gridspec_kw={'hspace': 0.23, 'wspace':0.3},   figsize=(15, 15) ,constrained_layout = True)

axs[0, 0].pcolormesh(x/mm, y/mm, phases_2[:,:,1], shading='auto',
                     vmin=-np.pi, vmax=np.pi, cmap='rainbow')
axs[0, 1].pcolormesh(x/mm, y/mm, phases_3[:, :, 1], shading='auto',
                     vmin=-np.pi, vmax=np.pi, cmap='rainbow')
amice2 = axs[0, 2].pcolormesh(x/mm, y/mm, phases_4[:, :, -1], shading='auto',
                     vmin=-np.pi, vmax=np.pi, cmap='rainbow')
axs[0,2].set_xlim([-4,4])
axs[0,2].set_ylim([-4,4])

fig2.colorbar(amice2, ax=axs[0, 2], label='phase [rad.]')

axs[1, 0].plot(x/mm, phases_2[:, :, 1][int(N / 2), :], linewidth=0.8)
axs[1, 1].plot(x/mm, phases_3[:, :, 1][int(N / 2), :], linewidth=0.8)
axs[1, 1].set_xlim([ -2*900*wave_length*f1/mm,  2*900*wave_length*f1/mm])

axs[1, 2].plot(x/mm, phases_4[:, :, -1][int(N / 2), :], linewidth=0.8)

axs[2, 0].plot(x/mm, phases_2[:, :, 1][:,int(N / 2)],color='r', linewidth=0.8)
axs[2, 0].set_xlim([ -700*wave_length*f1/mm,  700*wave_length*f1/mm])
axs[2, 1].plot(x/mm, phases_3[:, :, 1][:,int(N / 2)],color='r', linewidth=0.8)
axs[2, 2].plot(x/mm, phases_4[:, :, -1][:,int(N / 2)],color='r', linewidth=0.8)
axs[2, 2].set_xlim([ -700*wave_length*f1*(f2/f3)/mm, 700*wave_length*f1*(f2/f3)/mm])
axs[0, 0].set_ylabel('Phase Patterns \n y[mm]', fontweight='bold')
print('ben nu hier')
for j in range(0, 3):
    axs[0, j].set_title(str(distances[j]) + ' mm', fontsize=11)
    axs[1, j].set_xlabel('x [mm]', fontweight='bold');
    axs[2, j].set_xlabel('y [mm]', fontweight='bold');
axs[1, 0].yaxis.set_visible(True)
axs[1, 0].set_ylabel('x phase [rad.]', fontweight='bold')
axs[2, 0].set_ylabel('y phase [rad.]', fontweight='bold')
plt.savefig('LSM_phase_N4000.pdf', format='pdf', dpi=400)

