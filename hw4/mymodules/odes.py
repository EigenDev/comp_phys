"""
ODEs is a module used for solving Linear Ordinary Differential Equations
through numerical methods

Functions:
rk4 -- The Runge Kutta Method of Solving ODEs (robust)
adapt_bulirsch_stoer -- The adaptive timestep method for the Bulsirsch Stoer method

Marcus Dupont
New York University
Department of Physics
November 2019
"""

import numpy as np 
import collections

def rk4(f, r, t, h=0.1, *args, **kwargs):
    """
    Returns the solution to a 4th order differential equation using the Runge
    Kutta method

    INPUTS:

    f -- the desired N-Dimensional function to be used for ODE evaluation
    r -- an N-dimensional vector
    t -- the time arrays

    args -- Optional parameters for function f

    OUTPUT:
     p -- an N-Dimensional solution to the ode defined as f
    """

    #Check if the time value is an array or scalar
    if not isinstance(t, (collections.Sequence, np.ndarray)):

        k1 = h*f(r, t, *args, **kwargs)
        k2 = h*f(r+0.5*k1,t+0.5*h, *args, **kwargs)
        k3 = h*f(r+0.5*k2, t+0.5*h, *args, **kwargs)
        k4 = h*f(r+k3, t+h, *args, **kwargs)

        return (k1+2*k2+2*k2 +k4)/6

    else:
        t = np.array(t)
    
        w = t.size

        p = np.array([r]*w, float)

        for i in range(w - 1):

            k1 = h*f(p[i], t[i], *args)
            k2 = h*f(p[i]+0.5*k1,t[i]+0.5*h, *args)
            k3 = h*f(p[i]+0.5*k2, t[i]+0.5*h, *args)
            k4 = h*f(p[i]+k3, t[i]+h, *args)

            #print(k1)

            p[i+1] = p[i] + (k1 + 2*k2 +2*k3 +k4)/6

    
        return p
def adapt_rk4(f, r, t, h, delta, *args, **kwargs):
    """
    The adaptive Runge-Kutta method with
    varying time steps
    """
    
    #Perform two successive steps of size h
    dx1 = rk4(f, r, t, h, *args, **kwargs)
    dx2 = rk4(f, r+dx1, t+h, h, *args, **kwargs)
    
    dr1 = dx1 + dx2 
    
    #Now take a step of size 2h
    dr2 = rk4(f, r, t, 2*h, *args, **kwargs)
    
    #Separate Errors
    err_x1 = dr1[0]
    err_x2 = dr2[0]
    err_y1 = dr1[2]
    err_y2 = dr2[2]
    
    err = np.sqrt((err_x1-err_x2)**2 + (err_y1 - err_y2)**2)/30
    
    #Compute rho
    rho = (h*delta)/err
    
    c = rho**(1/4)
    if rho >= 1:
        t = t+2*h 
        
        if c > 2:
            h *= 2
        else:
            h *= c
            
        #print(dx1)
        # Use local extrapolation to better our estimate of the positions
        dr1[0] += (err_x1 - err_x2) / 15
        dr1[2] += (err_y1 - err_y2) / 15
        return dr1, h, t
    else:
        h_prime = h*c
        return adapt_rk4(f,r, t, h_prime,delta, *args, **kwargs)
    
def adapt_bulirsch_stoer(f,r, t, H, delta, *args):
    """
    Returns the solution to a Bulirsch-Stoer adaptive ODE
    solving method

    INPUTS:

    f -- the desired N-Dimensional function to be used for ODE evaluation
    r -- an N-dimensional vector
    t -- the time arrays
    H -- size of big steps if not defined by the user 
    delta -- accuracy in terms of time steps 
    N -- Number of bif steps 
    
    args -- Optional parameters for function f

    OUTPUT:
     p -- an N-Dimensional solution to the ode defined as f
    """
    #Check if the time value is an array or scalar
    times = np.array(t)

    #Initialize dynamics time array
    tpoints = [times[0]]
    xpoints = [r[0]]
    ypoints = [r[1]]
  
    def step(r, t, H):
        """
        Computes the nth Richardson extrapolation until the target accuracy is reached
        INPUTS:

        r -- the N-dimensional vector (x1, x2, x3,...)
        H -- Step size
        t -- the time array
        
        return r: vector at time t+H
        """
        
        def mod_midpoint_step(r, n):
            """
            returns the vector after perfoming
            the modified midpoint step formula
            """
            #preserve the vector r
            r = np.copy(r)
            
            h = H/n 
            r1 = r + 0.5*h*f(r, *args)
            r2 = r + h*f(r1, *args)
            
            for i in range(n-1):
                r1 += h*f(r2, *args)
                r2 += h*f(r1, *args)
                
            R1 = 0.5 * (r1 + r2 + 0.5*h*f(r2, *args))
            #print(R1)
            #zzz = input('')
            
            return R1
        
        def calc_table(R1, n):
            """
            Computes the nearest row from the Richardson Extrapolation
            table.
           
            INPUTS:
            R -- R1
            n -- the nth row
            """
            def R_column(m):
               """
               Calculate the row and column of the 
               Richardson table and vectorize it. Indices
               are shifted down one to account for the n = 1
               term later.
               """
               return R2[m-2] + (R2[m-2] - R1[m-2])/ ((n/(n-1))**(2*(m-1)) - 1)
           
            #Check for max value of n
            if n > 8:
                #print(n)
                r1 = step(r, t, H / 2)
                r2 = step(r1, t + H / 2, H / 2)
                return r2
            else:
                # Compute R_n,1
                R2 = [mod_midpoint_step(r, n)]
                # Compute the rest of the row
                for m in range(2, n + 1):
                    R2.append(R_column(m))

                # Flatten it into an array the calc error
                R2 = np.array(R2, float)
                error_vector = (R2[n - 2] - R1[n - 2]) / ((n / (n - 1)) ** (2 * (n - 1)) - 1)
                error = np.sqrt(error_vector[0] ** 2 + error_vector[1] ** 2)

                # If error is smaller than accuracy, calculation terminates, else repeat with 1 more step
                target_accuracy = H * delta
                #print("Accuracy", target_accuracy)
                #print(error)
                if error < target_accuracy:
                    tpoints.append(t + H)
                    xpoints.append(R2[n - 1][0])
                    ypoints.append(R2[n - 1][1])
                    #print(xpoints)
                    #zzz= input('')
                    #zzz = input('')
                    return R2[n - 1]
                else:
                    return calc_table(R2, n + 1) 
            
        return calc_table(np.array([mod_midpoint_step(r, 1)], float), 2)

    step(r, t, H)  
    #print("xpoints:", xpoints)  
    return xpoints,ypoints, tpoints