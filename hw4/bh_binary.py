#! /usr/bin/env python


#Supermassive Blackhole Binary Computation
#Prof Tinker

import numpy as np 
import matplotlib.pyplot as plt 
from astropy import units as u
from mymodules import odes 

distance_unit = 100 * u.pc
r_s = 1.e-7 *distance_unit


#Constants
m_sun = 1.989e33 * u.g 
m_bh = 1.e8 * m_sun
G = 6.67e-8 * (u.cm**3/(u.g*u.s**2))

def vdf(v, a, b):
    """
    The Dynamical Friction Force formula
    used to compute the EoM for BH binaries
    
    INPUTS:
    v -- velocity dispersion: array_like 
    a -- free paramater (physics), const 
    b -- free parameter (physics), const
    
    returns
    f -- The dynamical friction force, array_like
    """
    v = np.array(v)
    mag_v = np.sqrt(v.dot(v))
    return - a * v/(mag_v**3 + b)

def f(r,t, friction = True, a = 0, b =0):
    """
    The 2nd DE describing the black hole
    trajectory
    """
    #Assuming planer orbit
    #Turning the two (x,y) 2nd order
    #odes into 4 Ordinary DEs
    x = r[0]
    vx = r[1]
    y = r[2]
    vy = r[3]
    
    mag_r = np.sqrt(x**2 + y**2)
    
    v_tot = np.array([vx, vy], float)
    fx = vx 
    fy = vy
    if friction:
        fvx = -x/(4*mag_r**3) + vdf(v_tot, a, b)[0]
        fvy = -y/(4*mag_r**3) + vdf(v_tot, a, b)[1]
    else:
        fvx = -x/(4*mag_r**3)
        fvy = -y/(4*mag_r**3)
        
    return np.array([vx, fvx, vy, fvy], float)

#Part A: Solving for delta
def part_a(user_delta):
    #r_peri = 1.e-7
    #v_init = 1/np.sqrt(4*r_peri)

    delta = user_delta
    #Initial Conditions
    x_0 = 1
    y_0 = 0 
    v_0 = 0.8 *np.sqrt(1/(4))

    vx_0 = 0
    vy_0 = 1.e-3
    r = np.array([x_0,vx_0, y_0, vy_0])

    #initial step size
    #N = 1e5
    h = 1.e-3
    t_0 = 0

    x_arr =[]
    y_arr = []
    
    r_val = 1
    while r_val > 1.e-7:
        x_arr.append(r[0])
        y_arr.append(r[2])
        
        solution, h, t = odes.adapt_rk4(f, r, t_0, h, delta, friction=False)
        
        
        r_val = np.sqrt(solution[0]**2 + solution[2]**2)
        
    print("r_min", r_val)
    print("step_size:", h)
    print("times:", t)
    
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    
    plt.plot(x_arr, y_arr)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('circ.pdf')
    plt.show()

#Part B

def part_b():
    #Initial Conditions
    x_0 = 1
    y_0 = 0 
    v_0 = 0.8 *np.sqrt(1/(4))
    delta = 1.e-4
    h = 1.e-7

    vx_0 = 0
    vy_0 = v_0
    r = np.array([x_0,vx_0, y_0, vy_0])

    #arrays for storage
    tpoints = []
    xpoints = []
    ypoints = []

    vxpoints = []
    vypoints = []
    t = 0

    r_val = 1
    while r_val > 1.e-7:
        tpoints.append(t)
        xpoints.append(r[0])
        vxpoints.append(r[1])
        ypoints.append(r[2])
        vypoints.append(r[3])
        
        dr, h, t = odes.adapt_rk4(f, r, t, h, delta, friction=True, a=1, b=1)
        r += dr
        r_val = np.sqrt(r[0]**2 + r[2]**2)

        
    xpoints = np.array(xpoints)*distance_unit
    ypoints = np.array(ypoints)*distance_unit
    tpoints = np.array(tpoints)
    vxpoints = np.array(vxpoints)
    vypoints = np.array(vypoints)

    r_l = np.sqrt(xpoints**2 + ypoints**2)
    v_tot = np.sqrt(vxpoints**2 + vypoints**2)*np.sqrt(G*m_bh/(4*distance_unit.to(u.cm)))

    #Convert time units
    tru_t = (r_l.to(u.cm))/(v_tot)
    tru_t = tru_t.to(u.yr)

    #get rid of r units for log
    rs = r_l/distance_unit

    #plot the trajectory
    fig, axs = plt.subplots(1, 2, figsize=(10,15))
    axs[0].plot(xpoints.value, ypoints.value)
    axs[0].set_xlabel('x [pc]', fontsize=12)
    axs[0].set_ylabel('y [pc]', fontsize=12)

    #print('True time:', tru_t)
    #print("Time points:", tpoints)
    #zzz = input('')

    axs[1].semilogx(tpoints*tru_t[0].value, np.log10(rs))
    axs[1].set_xlabel('time [yr]', fontsize=12)
    axs[1].set_ylabel('log (r)', fontsize=12)
    axs[1].set_xlim(1.e5, 1.e8)

    save_fig = input('Is the result good enough to save? [y/n]')
    if save_fig == 'y':
        plt.savefig('Bh_trajectory_xy.pdf')
    else:
        pass

    plt.show()


#Part C

def part_c():
    #Initial Conditions
    x_0 = 1
    y_0 = 0 
    v_0 = 0.8 *np.sqrt(1/(4))
    delta = 1.e-4
    h = 1.e-7
    
    # A, B arrays
    a = np.arange(0.5, 10)
    b = np.arange(0.5, 10)

    #Initialze dicts to store 
    #values corresponding the the (a,b)
    #parameters
    ab_rat = []
    x_dict = {}
    y_dict = {}
    t_dict = {}
    vx_dict = {}
    vy_dict = {}
    for b_val in b:
        x_dict[b_val] = {}
        y_dict[b_val] = {}
        t_dict[b_val] = {}
        vx_dict[b_val] = {}
        vy_dict[b_val] = {}
        for a_val in a:
            #Change the initial conditions
            vx_0 = 0
            vy_0 = v_0
            r = np.array([x_0,vx_0, y_0, vy_0])
            
            #arrays for storage
            tpoints = []
            xpoints = []
            ypoints = []
            vxpoints = []
            vypoints = []
            t = 0
            r_val = 1
            while r_val > 1.e-7:
                tpoints.append(t)
                xpoints.append(r[0])
                vxpoints.append(r[1])
                ypoints.append(r[2])
                vypoints.append(r[3])
                
                dr, h, t = odes.adapt_rk4(f, r, t, h, delta, friction=True, a=a_val, b=b_val)
                r += dr
                r_val = np.sqrt(r[0]**2 + r[2]**2)
                
            #Add the relevant solutions to their respective
            #dictionary positions
            x_dict[b_val][a_val] = np.array(xpoints)*distance_unit
            y_dict[b_val][a_val] = np.array(ypoints)*distance_unit
            t_dict[b_val][a_val] = np.array(tpoints)
            vx_dict[b_val][a_val] = np.array(vxpoints)
            vy_dict[b_val][a_val] = np.array(vypoints)
            
            #add the A/B ratio into array
            ab_rat.append(b_val/a_val)

    #Define S time
    ts = []
    ab_rat = np.array(ab_rat)
    for key, val in t_dict.keys():
        for a_key, val in t_dict[key].keys():
            
            r_l = np.sqrt(x_dict[key][a_key]**2 + 
                        y_dict[key][a_key]**2)
            
            v_tot = (np.sqrt(vx_dict[key][a_key]**2 + vy_dict[key][a_key]**2)*
                    np.sqrt(G*m_bh/(4*distance_unit.to(u.cm))))

            #Convert time units
            tru_t = (r_l.to(u.cm))/(v_tot)
            tru_t = tru_t.to(u.yr)
            
            #Get time value at r_s and scale to right units
            ts.append(t_dict[key][a_key][-1]*tru_t[0].value)
    
    #ts in Myr
    ts = np.array(ts)/1.e6 
            
    #Plot Stuff
    plt.semilogx(ts, ab_rat)
    plt.xlabel('T [Myr]', fontsize=12)
    plt.ylabel('B/A', fontsize=12)
    plt.savefig('ratio.pdf')

which_function = input('which part do you want to compute? [a, b, c]')

if which_function == 'a':
    u_delta = input('Input a guess value for delta:')
    u_delta = float(u_delta)
    part_a(u_delta)
elif which_function == 'b':
    part_b()
elif which_function == 'c':
    part_c()
else:
    raise ValueError('Please choose either lowercase a, b, or c.')