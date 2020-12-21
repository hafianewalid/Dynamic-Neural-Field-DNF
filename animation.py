from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import DNF

dnf=DNF.DNF()
dnf_noise=DNF.DNF(noise=True)
dnf_Asencro=DNF.DNF()
dnf_Asencro_noise=DNF.DNF()
dnf_Asencro_noise.Input_Map=dnf_noise.Input_Map

dnf_rk=DNF.DNF()
dnf_noise_rk=DNF.DNF()
dnf_noise_rk.Input_Map=dnf_noise.Input_Map
dnf_Asencro_rk=DNF.DNF()
dnf_Asencro_noise_rk=DNF.DNF()
dnf_Asencro_noise_rk.Input_Map=dnf_noise.Input_Map


'''
def updatefig(*args):
    #im.set_array(np.random.random((45, 45)))
    #a = DNF.gaussian_distribution((0.1,0.1),(45,45),0.1)
    #a = DNF.gaussian_activity((0.1,0.5),(0.8,0.5),0.1)
    #a = DNF.gaussian_distribution_constant((0.1,0.1),(45,45),0.1)
    #a = DNF.generate_selection_kernel((45,45),100,0.03,100)

    print("up")
    dnf.syncronous_run()
    im.set_array(dnf.Map)
    return im,


'''
def updatefig(*args):
    dnf.syncronous_run()
    dnf_noise.syncronous_run()
    dnf_Asencro.Asynchronous_run()
    dnf_Asencro_noise.Asynchronous_run()

    dnf_rk.syncronous_run(update=1)
    dnf_noise_rk.syncronous_run(update=1)
    dnf_Asencro_rk.Asynchronous_run(update=1)
    dnf_Asencro_noise_rk.Asynchronous_run(update=1)

    im_input.set_array(dnf.Input_Map)
    im_noise.set_array(dnf_noise.Input_Map)
    im_kernel.set_array(dnf.K)

    im_dnf.set_array(dnf.Map)
    im_dnf_noise.set_array(dnf_noise.Map)
    im_dnfA.set_array(dnf_Asencro.Map)
    im_dnf_noiseA.set_array(dnf_Asencro_noise.Map)

    im_dnf_rk.set_array(dnf.Map)
    im_dnf_noise_rk.set_array(dnf_noise.Map)
    im_dnfA_rk.set_array(dnf_Asencro.Map)
    im_dnf_noiseA_rk.set_array(dnf_Asencro_noise.Map)

    return im_input, im_noise, im_kernel, im_dnf ,im_dnf_noise,im_dnfA ,im_dnf_noiseA ,im_dnf_rk ,im_dnf_noise_rk,im_dnfA_rk,im_dnf_noiseA_rk


if __name__ == '__main__':
    '''
    fig, axs = plt.subplots(1,3)

    axs[0].set_title('Input')
    axs[0].imshow(dnf.Map,cmap='hot')

    axs[1].set_title('Kernel')
    axs[1].imshow(dnf.Map_conv)

    axs[2].set_title('Potontial')
    a = np.random.random((45, 45))
    im = plt.imshow(a, cmap='hot', interpolation='nearest', animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True) # interval est le temps en ms entre chaque frame

    plt.show()

    '''

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)

    ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)

    plt.subplot(3,4,1)
    plt.title("Input")
    im_input= plt.imshow(dnf.Input_Map, cmap='hot', interpolation='nearest', animated=False)

    plt.subplot(3,4,2)
    plt.title("Input with noise")
    im_noise=plt.imshow(dnf_noise.Input_Map, cmap='hot', interpolation='nearest', animated=True)

    plt.subplot(3,4,3)
    plt.title("Kernel")
    im_kernel=plt.imshow(dnf.K, animated=False)
    
    #################################

    plt.subplot(3,4,5)
    plt.title("DNF/syncro")
    im_dnf = plt.imshow(dnf.Input_Map, cmap='hot', interpolation='nearest', animated=True)

    plt.subplot(3,4,6)
    plt.title("DNF/syncro/noise")
    im_dnf_noise = plt.imshow(dnf_noise.Input_Map, cmap='hot', interpolation='nearest', animated=True)

    plt.subplot(3, 4, 7)
    plt.title("DNF/Asyncro")
    im_dnfA = plt.imshow(dnf.Input_Map, cmap='hot', interpolation='nearest', animated=True)

    plt.subplot(3, 4, 8)
    plt.title("DNF/Asyncro/noise")
    im_dnf_noiseA = plt.imshow(dnf_noise.Input_Map, cmap='hot', interpolation='nearest', animated=True)
    
    
    #####################################

    plt.subplot(3, 4, 9)
    plt.title("RK/syncro")
    im_dnf_rk = plt.imshow(dnf.Input_Map, cmap='hot', interpolation='nearest', animated=True)

    plt.subplot(3, 4, 10)
    plt.title("RK/syncro/noise")
    im_dnf_noise_rk = plt.imshow(dnf_noise.Input_Map, cmap='hot', interpolation='nearest', animated=True)

    plt.subplot(3, 4, 11)
    plt.title("RK/Asyncro")
    im_dnfA_rk = plt.imshow(dnf.Input_Map, cmap='hot', interpolation='nearest', animated=True)

    plt.subplot(3, 4, 12)
    plt.title("RK/Asyncro/noise")
    im_dnf_noiseA_rk = plt.imshow(dnf_noise.Input_Map, cmap='hot', interpolation='nearest', animated=True)



    fig.tight_layout(pad=3.0)
    plt.show()

