# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:10:48 2022

@author: GabyPyka
"""
#First thing's first- importing the packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Defining the constants for later use
SPEED_OF_LIGHT = 299792458 #m/s
EMITTED_WAVELENGTH = 6.5628e-7 #m

#Defining function that will fit each plot; a combination of gaussian and linear
def fit_func(x, a, mu, sig, m, c):
    """
    Takes gaussian and linear parameters and returns the combination 
    """
    gaus = a * np.exp( - (x - mu) ** 2 / (2 * sig ** 2))
    line = m * x + c
    return gaus + line

#Reading both sets of data into python
h_alpha = np.loadtxt("Data_CP1\Halpha_spectral_data.csv", delimiter=',', skiprows=3)
distance_mpc = np.loadtxt("Data_CP1\Distance_Mpc.txt", skiprows=2)

#Create coordinate arrays for the velocity plot
velocity_arr = np.array([])
distance_arr = np.array([])

#Iterate through the galaxy observations
no_of_galaxy_observations = len(h_alpha[0]) 
figure_no = 1 #Figure is the number of the galaxy from 1-30x

for freq_col_no in range(0, no_of_galaxy_observations, 2):
    
    #Useful variables
    intensity_col_no = freq_col_no + 1
    galaxy_observation = h_alpha[0, freq_col_no]
    
    #Plot freq and intensity of galaxy observations
    freq = h_alpha[1:, freq_col_no] #This is frequency of the emitted wavelengths and is located in the first column of each galaxy.
    intensity= h_alpha[1:, intensity_col_no] #Intensity refers to the intensity of the emitted light from the galaxy and is the second column of each galaxy.
    plt.plot(freq,intensity) #Plotting intensity against frequency
    plt.ylabel(f"{intensity_col_no}: Intensity (arb unit)") #Next to the intensity label is the column number of the data
    plt.xlabel(f"{freq_col_no}: Frequency (Hz)") #Next to the freq label is the column number of the data
    plt.title(f"{figure_no}. Galaxy Observation Number: {int(galaxy_observation)}") 
    
    #Increment figure count number to ensure that all 30 galaxies have been plotted.
    figure_no = figure_no + 1

    #Find initial Guesses for m and c
    #To find m, we take the first point on the graph and the last point and use those for the gradient as to minimise the possible variation on the gradient.
    first_intensity_reading = h_alpha[1, intensity_col_no] # = y1
    last_intensity_reading = h_alpha[1000, intensity_col_no] # = y2
    first_frequency_reading = h_alpha[1, freq_col_no] # = x1
    last_frequency_reading = h_alpha[1000, freq_col_no] # = x2
    m_guess = (first_intensity_reading - last_intensity_reading) / (first_frequency_reading - last_frequency_reading) # m = (y1 - y2) / (x1 - x2)
    c_guess = first_intensity_reading - (m_guess * first_frequency_reading) #c_guess = y1 - (m* x1)
    
    #Testing if the linear fit is sensible on each plot.
    fitted_line = (m_guess * freq) +c_guess
    #plt.plot(freq, fitted_line) #This is not needed in the final code so it is commented out but was useful part of the process.
    
    
    #Find initial guesses for a, mu, sig   
    gaus_guess = intensity - ((m_guess * freq)+ c_guess) #Finding the residual to obtain a guess for a,mu, and sig
    a_guess = max(gaus_guess) # a is roughly the maximum value for the residual
    index_of_highest_gg = np.where(gaus_guess == a_guess) #mu is roughly the x-coordinate of a so we need to find the index of 'a' and then find the frequency of this 'a' value, this will be our mu_guess
    mu_guess = freq[index_of_highest_gg][0] #Finding the frequency in the 0th column and in the same row as the 'a' value.
    sig_guess = np.sqrt(sum((freq-mu_guess)**2)/len(freq)) #Using formula for standard deviation but using our mu_guess as the mean population.
    #plt.plot(freq,gaus_guess,'x', color='red') #This plotted the residual but wasn't needed in the final code.

    #Final fit of curve
    initial_guess = [a_guess, mu_guess, sig_guess, m_guess, c_guess] #Inputting all the intial guesses
    po,po_cov = curve_fit(fit_func, freq,intensity, initial_guess) #Finding the parameters and covariance matrix
    plt.plot(freq, fit_func(freq, po[0], po[1], po[2], po[3], po[4])) #Plotting the fit
    plt.show()
    
    #Find redshifted velocity for galaxies with good instrument response
    index_no = np.where(distance_mpc == galaxy_observation)[0] #Matching the galaxy observation numbers from the two documents
    if distance_mpc[index_no, 2] == 1: #IF the instrument response is 1
        observed_wavelength = SPEED_OF_LIGHT / po[1] 
        squared_wavelength_ratio = (observed_wavelength / EMITTED_WAVELENGTH) ** 2
        redshifted_velocity = (SPEED_OF_LIGHT * (squared_wavelength_ratio - 1)) / (1 + squared_wavelength_ratio) #Formula for velocity
        
        #Store velocity and distance for the later plot
        velocity_arr = np.append(velocity_arr, redshifted_velocity)
        distance_arr = np.append(distance_arr, distance_mpc[index_no, 1])
         
#Plot velocity against distance
fit_vel, cov_vel = np.polyfit(distance_arr, velocity_arr, 1, cov=True) 
final_line = np.poly1d(fit_vel)
plt.plot(distance_arr, final_line(distance_arr))

plt.plot(distance_arr, velocity_arr, 'x', color = 'darkgreen')
plt.title("Hubble's Law")
plt.ylabel("Velocity (m/s)")
plt.xlabel("Distance (Mpc)")
plt.grid()
plt.show()

#Calculating final estimate for hubble's constant (gradient of v-d graph) and the uncertainty.
uncertainty = np.sqrt(cov_vel[0,0])
print("H0 =", fit_vel[0],"+/-", uncertainty, "m/s/Mpc")
