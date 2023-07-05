import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import os
import polyscope as ps
import polyscope.imgui as psim

import utils
import config

class rigid2d:

    @staticmethod
    def construct(problem_name):

        '''

        Basic philosophy:
            We define the system via two objects, a object instance ('system'), which can
            hold pretty much anything (strings, function pointers, etc), and a dictionary 
            ('system_def') which holds only jnp.array objects.

            The reason for this split is the JAX JIT system. Data stored in the `system`
            is fixed after construction, so it can be anything. Data stored in `system_def`
            can potentially be modified after the system is constructed, so it must consist
            only of JAX arrays to make JAX's JIT engine happy.


        The fields which MUST be populated are:
       
            = System name
            system.system_name --> str
            The name of the system (e.g. "neohookean")
            

            = Problem name
            system.problem_name --> str
            The name of the problem (e.g. "trussbar2")
       

            = Dimension
            system.dim --> int
                The dimension of the configuration space for the system. When we
                learn subspaces that map from `R^d --> R^n`, this is `n`. If the 
                system internally expands the configuration space e.g. to append 
                additional pinned vertex positions, those should NOT be counted 
                here.


            = Initial position
            system_def['init_pos'] --> jnp.array float dimension: (n,)
                An initial position, used to set the initial state in the GUI
                It does not necessarily need to be the rest pose of the system.
            

            = Conditional parameters dimension
            system.cond_dim --> int
                The dimension of the conditional parameter space for the system. 
                If there are no conditional parameters, use 0.
            
            = Conditional parameters value
            system_def['cond_param'] --> jnp.array float dimension: (c,)
                A vector of conditional paramters for the system, defining its current
                state. If there are no conditional parameters, use a length-0 vector.
            

            = External forces
            system_def['external_forces'] --> dictionary of anything
                Data defining external forces which can be adjusted at runtime
                The meaning of this dictionary is totally system-dependent. If unused, 
                leave as an empty dictionary. Fill the dictionary with arrays or other 
                data needed to evaluate external forces.


            = Interesting states
            system_def['interesting_states'] --> jnp.array float dimension (i,n)
                A collection of `i` configuration state vectors which we want to explicitly track
                and preserve.
                If there are no interesting states (i=0), then this should just be a size (0,n) array.


            The dictionary may also be populated with any other user-values which are useful in 
            defining the system.

            NOTE: When you add a new system class, also add it to the registry in config.py 
        '''

        system_def = {}
        system=rigid2d()


        # Example values:
        # (dummy data)
        system_def['system_name'] = "rigid2d"
        system_def['problem_name'] = problem_name
        system_def['cond_params'] = jnp.zeros((0,)) # a length-0 array
        system_def['external_forces'] = {}

        bodies = []
        joint_list = []
        numBodiesFixed = 0

        if problem_name == 'simple_pendulum':
            config_dim = 2
            
            system_def['mass'] = 5
            system_def['length'] = 10
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))
            system_def['x0'] = jnp.array([0, 0])
            system_def['external_forces']['torque_strength_minmax'] = (-15, 15) # in m/s
            system_def['external_forces']['torque_strength'] = 0.0

            # Define the external forces
            system_def['gravity'] = jnp.array([0.0, 0.0, -9.8])
        
        elif problem_name == 'problem_B':

            # and so on....
            config_dim = 334
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some other values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))

        else:
            raise ValueError("could not parse problem name: " + str(problem_name))

        pos  = system_def['x0']
        system.dim = pos.size
        system.cond_dim = 0
        system_def['interesting_states'] = system_def['init_pos'][None,:]


        return system, system_def
  
    # ===========================================
    # === Energy functions 
    # ===========================================

    # These define the core physics of our system

    def potential_energy(self, system_def, q):
        qR = q.reshape(-1,2,1)
        q_pos = jnp.take(qR, jnp.array([0]))
        lengthR = system_def['length']

        height = lengthR * jnp.cos(q_pos)
        massR = system_def['mass']
        gravity = system_def['gravity']  

        gravity_energy = height * massR * gravity[None,2]

        return gravity_energy
    
    def real_kineticenergy(self, system_def, q):
        qR = q.reshape(-1,2,1)
        q_vel = jnp.take(qR, jnp.array([1]))
        massR = system_def['mass']
        lengthR = system_def['length']

        inertia = massR * jnp.square(lengthR)
        ke = 1/2*inertia*jnp.square(q_vel)

        return ke
    
    def dissipation_fnc(self, n, c, q):
        qR = q.reshape(-1,2,1)
        q_vel = jnp.take(qR, jnp.array([0]))
        D = 1/(n+1)*c*q_vel
        return D

    def action(self, system_def, q):
        external_force = system_def['external_forces']['torque_strength']
        
        PE = self.potential_energy(system_def, q)
        KE = self.real_kineticenergy(system_def, q)
        dis_extforce = self.dissipation_fnc(0, external_force, q)

        return PE+KE+dis_extforce

    def kinetic_energy(self, system_def, q_dot):
        q_dotR = q_dot.reshape(-1,2,1)
        q_pos = jnp.take(q_dotR, jnp.array([0]))
        q_vel = jnp.take(q_dotR, jnp.array([1]))
        error_arr = jnp.array([jnp.square(q_pos), jnp.square(q_vel)])
        error = 1/2*jnp.sum(error_arr)
        return error

    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(self, system_def, rngkey, rho=1.):
        return jnp.zeros((0,))

    # ===========================================
    # === Visualization routines
    # ===========================================
    
    def build_system_ui(system_def):
        # Construct a Polyscope gui to tweak parameters of the system
        # Make psim.InputFloat etc calls here

        # If appliciable, the cond_params values and external_forces values should be 
        # made editable in this function.
         if psim.TreeNode("system UI"):
            psim.TextUnformatted("External forces:")

            if "torque_strength" in system_def["external_forces"]:
                low, high = system_def['external_forces']['torque_strength_minmax']
                _, new_val = psim.SliderFloat("torque_strength", float(system_def['external_forces']['torque_strength']), low, high)
                system_def['external_forces']['torque_strength'] = jnp.array(new_val)
                
                psim.TreePop()

    def visualize(self, system_def, x, name="rigid2d", prefix='', transparency=1.):

        print(x)
        

    def visualize_set_nice_view(system_def, q):
        # Set a Polyscope camera view which nicely looks at the scene
        
        # Example:
        # (could also be dependent on x)
        # ps.look_at((2., 1., 2.), (0., 0., 0.))

        ps.look_at((1.5, 1.5, 1.5), (0., -.2, 0.))
    
        pass
