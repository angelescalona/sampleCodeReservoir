import fiona
import mdp
from scipy.optimize import fsolve
from pydelay import dde23
import math
from euler_laser31 import *
import sys

#******************************************************************
#   This script is part of python module FIONA for the fast 
#   implementation of reservoir computing. It works as the core 
#   of the reservoir computer to transfor the data. 
#
#
#   Created by Miguel Escalona-Moran
#   Version 1.2, November 15 2011
#******************************************************************

class Reservoir_SNLND(mdp.Node):
    """
    A Single NonLinear Node with Delay (SNLND) reservoir.
    """
    
    def __init__(self, eta, gamma, p, tau, theta, dynamics='mg_new', \
    integrator='euler', dt=mdp.numx.float64(0.2), s_rate=1, input_dim=1, output_dim=None,\
    spectral_radius=mdp.numx.float64(0.9), bias_scaling=0,\
    input_scaling=mdp.numx.float64(1.0), dtype='float64', _instance=0, \
    w_connect=None, offset=0.0, w=None, w_bias=True, init_reservoir_at=1):
        """ Initializes and constructs an SNLND reservoir.
        Parameters are:
            - eta, gamma, p, tau, and theta are parameters of the nonlinear function.
            - dynamics: the nonlinear function
            - integrator: the method to integrate the nonlinear function
            - dt: step of integration
            - s_rate: sampling rate of results
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - w_connect: connection matrix (connection among neurons)
            - offset: specify if there is an offset from the steady point of the nonlinearity
            - init_reservoir_at: should the reservoir start at a steady point?
        
        Weight matrices are either generated randomly or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created randomly:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        
        OUTPUT: When the reservoir is executed, it returns the state matrix of the input. 
        """
        super(Reservoir_SNLND, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)

        # Set all object attributes
        # Scaling for input weight matrix
        self.input_scaling = input_scaling
        # Scaling for bias weight matrix
        self.bias_scaling = bias_scaling
        # Spectral radius scaling
        self.spectral_radius = spectral_radius
        # Instance ID, used for making different reservoir instantiations with the same parameters
        # Can be ranged over to simulate different 'runs'
        self._instance = _instance
        # Non-linear function
        self.nonlin_func = nonlin_func
        #Other
        self.dt = dt
        self.eta = eta
        self.gamma = gamma
        self.p = p
        self.tau = tau
        self.theta = theta
        self.s_rate = s_rate
        self.dynamics = dynamics
        self.dynamics_dict = {"mg_old":mg_old, "mg_new": mg_new, "mg_another": mg_another, \
                              "mg_another_fixed_point": mg_another_fixed_point, \
                              "mg_original": mg_original, "ikeda": ikeda}
        self.integrator = integrator
        self.offset = offset
        self._init_reservoir_at = init_reservoir_at
        self.initial_state = mdp.numx.zeros((1, self.output_dim),dtype='float64')        
        
        # Store any externally passed initialization values for w, w_in and w_bias
        self.w_connect_initial = w_connect
        self.w_initial = w
        self.w_bias_initial = w_bias
        
        # Fields for allocating reservoir weight matrix w, input weight matrix w_in
        # and bias weight matrix w_bias
        self.w_connect = []
        self.w = []
        self.w_bias = []
        
        # Call the initialize function to create the weight matrices
        self.initialize()

    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def initialize(self):
        """ Initialize the weight matrices of the reservoir node. If no 
        arguments for w, w_in and w_bias matrices were given at construction
        time, they will be created as follows:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        """
                
        #Connectivity matrix
        if self.w_connect_initial is None:
            self.w_connect = mdp.numx.random.uniform(-1,1, self.output_dim*self.input_dim)
            self.w_connect.shape = (self.output_dim, self.input_dim)

        else:
            if callable(self.w_connect_initial):
                self.w_connect = self.input_scaling * self.w_connect_initial() # If it is a function, call it
            else:
                self.w_connect = self.input_scaling * self.w_connect_initial # else just copy it

        if self.w_connect.shape != (self.output_dim, self.input_dim):
            self.w_connect = self.w_connect[:self.output_dim,:self.input_dim]
            exception_str = 'Shape of given w_connect does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_connect: ' + str(self.w_connect.shape)
            raise mdp.NodeException(exception_str)
        
    
            
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        steps = x.shape[0]

        # Pre-allocate the state vector, adding the initial state
        states = mdp.numx.concatenate((self.initial_state, mdp.numx.zeros((steps, self.output_dim))))
        t = mdp.numx.float64(0.0)
        if (self._init_reservoir_at == 1):
            y_out =  mdp.numx.float64(fsolve(self.dynamics_dict[self.dynamics+'_fixed_point'],\
                                             0.2 , args=(self.eta, self.p))[0])  #condicion inicial del MG
        else:
            y_out = 0.0

        #setting virtual nodes vn
        v_n = mdp.numx.zeros(self.output_dim, dtype='float64') #Virtual nodes
        aux = y_out*mdp.numx.ones(self.s_rate*self.output_dim, dtype='float64')  #Virtual nodes auxiliar array DELAYS
        #Preparing input matrix J
        J = (mdp.utils.matmult(self.w_connect,x, trans_b=1))        
        
        count =0

        time_series = []
        #NOTE: due to the delay of the nonlinear node, this integrator might be managed using nested for's. Look for improvements.
        if self.integrator=='euler':
            for n in range(steps):
                index = 0
                for node in range(self.output_dim):

                    for inter in range(self.s_rate):
                        #Integrate function defined at self.dynamics
                        y_out = fiona.utils.euler(t, self.dt, y_out, self.dynamics_dict[self.dynamics], \
                                                aux[index] , self.eta, self.gamma, self.p, \
                                                (J.T[n][node]+ self.offset))
                        t += self.dt
                        aux[index] = y_out #keep track of all values for each n, this is: the delayed values
                        index += 1
                        time_series.append(y_out) #time_series contains the outputs of the nonlinear node (at full resolution.)
                        
                    v_n[node] = y_out

                #####Assign the corresponding states to the reservoir
                states[n+1,:] = v_n[:]


        elif self.integrator=='ko_euler':
            #function deleted from this sample
            pass
        
        elif self.integrator=='heun_laser':
            #function deleted from this sample
            pass


        elif self.integrator=='pydelay':
            #function deleted from this sample
            pass
        else:
            sys.exit("Integration method not understood, it should be 'euler', 'heun_laser' or 'pydelay'.")

        # Return the whole state matrix except the initial state
        return states[1:,:]


     
    def _post_update_hook(self, states, input, timestep):
        """ Hook which gets executed after the state update equation for every timestep. Do not use this to change the state of the 
            reservoir (e.g. to train internal weights) if you want to use parallellization - use the TrainableReservoirNode in that case.
        """
        pass


# Function to integrate
def mg_old(t,y, tau, a, b, c, d):
#   a = eta,    b = gamma,    c = p,    d=J
    parenthesis = tau + d
    return ( ((a * ( parenthesis ))/( mdp.numx.float64(1.0) + (b ** c)*( ( parenthesis ) ** c ))) - y )

def mg_new(t,y, tau, a, b, c, d,e):
#   a = eta,    b = gamma,    c = p, d=J
    parenthesis = tau + b*d
    if (c<0.0): print "Warning: A negative Mackey-Glass exponent can result in singularities of the function"
    else: return ( (( a * ( parenthesis )) / ( mdp.numx.float64(1.0) + ( parenthesis ) ** c )) - y )


def mg_another(t,y, tau, a, b, c, d, e):
    parenthesis = tau + b*d + e
    return ( (( a * ( parenthesis )) / ( mdp.numx.float64(1.0) + ( parenthesis ) ** c )) - y )
    
def mg_another_fixed_point(y,a,c,e):
    return ( a * (y + e) / (mdp.numx.float64(1.0) + (y + e) ** c ) - y)

def mg_original(t,y, tau, a, b, c, d): #Units in Volts
#   a = eta,    b = gamma,    c = p, d=J
    parenthesis = a*tau + b*d
    return ( (( mdp.numx.float64(1.33) * ( parenthesis )) / ( mdp.numx.float64(1.0) + ( mdp.numx.float64(0.4)*parenthesis ) ** c )) - y )

def ikeda(t,y, tau, a, b, c, d):
#   a = beta,    b = phi,    c = rho, d=J
    b_rad = b * math.pi
    mu = 1
    argument = mu * tau + c * d + b_rad
    return (a * math.sin(argument) ** (2.0) - y)



class FeedbackReservoirNode_SNLND(Reservoir_SNLND):
    """This is a reservoir node that can be used for setups that use output 
    feedback. Note that because state needs to be stored in the Node object,
    this Node type is not parallelizable using threads.
    """
    
    def __init__(self, reset_states=True, **kwargs):
        super(FeedbackReservoirNode_SNLND, self).__init__(**kwargs)
        self.reset_states = reset_states
        self.states = mdp.numx.zeros((1, self.output_dim))


    def _execute(self, x):        
        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            self.initial_state = mdp.numx.zeros((1, self.output_dim))
        else:
            self.initial_state = mdp.numx.atleast_2d(self.states[-1, :])

        self.states = super(FeedbackReservoirNode_SNLND, self)._execute(x)
        return self.states



