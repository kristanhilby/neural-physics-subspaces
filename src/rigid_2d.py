import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import os
import polyscope as ps
import polyscope.imgui as psim

try:
    import igl
finally:
    print("WARNING: igl bindings not available")

import utils
import config

# Define general functions for defining and making bodies
def make_body(file, density, scale):

    v, f = igl.read_triangle_mesh(file)
    v = scale*v

    vol = igl.massmatrix(v,f).data
    vol = np.nan_to_num(vol) # massmatrix returns Nans in some stewart meshes

    c = np.sum( vol[:,None]*v, axis=0 ) / np.sum(vol) 
    v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    mass = np.matmul(W.T, vol[:,None]*W) * density

    x0 = jnp.array( [c[0], 0, c[1], 0, c[2], 0, 0, 0, 0, 0, 0, 0] )

    body = {'v': v, 'f': f, 'W':W, 'x0': x0, 'mass': mass }
    return body

def make_joint( b0, b1, bodies, joint_pos_world, joint_vec_world ):
    # Creates a joint between the specified bodies, assumes the bodies have zero rotation and are properly aligned in the world
    # TODO: Use rotation for joint initialization
    pb0 = joint_pos_world
    vb0 = joint_vec_world
    if b0 != -1:
        c0 = bodies[b0]['x0'][3,:]
        pb0 = pb0 - c0
    pb1 = joint_pos_world
    vb1 = joint_vec_world
    if b1 != -1:
        c1 = bodies[b1]['x0'][3,:]
        pb1 = pb1 - c1
    joint = {'body_id0': b0, 'body_id1': b1, 'pos_body0': pb0, 'pos_body1': pb1, 'vec_body0': vb0, 'vec_body1': vb1}
    return joint

def bodiesToStructOfArrays(bodies):
    v_arr = []
    f_arr = []
    W_arr = []
    x0_arr = []
    mass_arr = []
    for b in bodies:
        v_arr.append(b['v'])
        f_arr.append(b['f'])
        W_arr.append(b['W'])
        x0_arr.append(b['x0'])
        mass_arr.append(b['mass'])
    
    out_struct = {
        'v'     : jnp.stack(v_arr, axis=0),
        'f'     : jnp.stack(f_arr, axis=0),
        'W'     : jnp.stack(W_arr, axis=0),
        'x0'    : jnp.stack(x0_arr, axis=0),
        'mass'  : jnp.stack(mass_arr, axis=0),
    }

    n_bodies = len(v_arr)

    return out_struct, n_bodies

class rigid2d:

    @staticmethod
    def construct(problem_name):
        system_def = {}
        system=rigid2d()

        # Example values:
        # (dummy data)
        system_def['system_name'] = "rigid2d"
        system.problem_name = str(problem_name)
        
        # Set some default parameters
        system_def['external_forces'] = {}
        system_def['cond_param'] = jnp.zeros((0,))
        system_def["contact_stiffness"] = 1000000.0
        system.cond_dim = 0
        system.body_ID = None

        bodies = []
        joint_list = []
        numBodiesFixed = 0

        if problem_name == 'simple_pendulum':
            # Declate number of fixed bodies
            numBodiesFixed = 0

            # Include the relevant bodies
            bodies.append( make_body( os.path.join(".", "data", "simplependulum_pendulum.obj"), 1000, 1.0) )
            
            # Create the joints between the bodies
            joint_list.append( make_joint( 0, -1, bodies, jnp.array([ 12.5, 0.0 ,12.5 ]), jnp.array([ 0, 1.0, 0.0 ]) ) )
            
            system_def['length'] = 50

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

        posFixed  = jnp.array( np.array([ body['x0']   for body in bodies[0:numBodiesFixed] ]).flatten() )
        pos  = jnp.array( np.array([ body['x0']   for body in bodies[numBodiesFixed:] ]).flatten() )

        mass = jnp.array( np.array([ body['mass'] for body in bodies[numBodiesFixed:] ]).flatten() )
        
        #
        system.dim = pos.size

        system.bodiesRen = bodies
        system.n_bodies = len(bodies)
        
        #
        system.joints = joint_list

        system_def['fixed_pos'] = posFixed
        system_def['rest_pos'] = pos
        system_def['init_pos'] = pos
        system_def['mass'] = mass
        system_def['dim'] = pos.size

        system_def['interesting_states'] = system_def['init_pos'][None,:]

        return system, system_def

  
    # ===========================================
    # === Energy functions 
    # ===========================================

    # These define the core physics of our system

    def potential_energy(self, system_def, q):
        # Joint contact energy
        qRFull = jnp.concatenate((system_def['fixed_pos'],q)).reshape(-1,12,1)
        qr_linpos = jnp.array([qRFull[:,0,1], qRFull[:,2,1], qRFull[:,4,1]])

        ###########

        joint_energy = 0.0

        for j in self.joints:

            pb0 = j['pos_body0']
            vb0 = j['vec_body0']
            
            b0id = j['body_id0']
            if b0id != -1:
                # transform point on body to point in world
                pb0 = jnp.matmul(jnp.append(pb0,1), qr_linpos[b0id])
                vb0 = jnp.matmul(jnp.append(vb0,0), qr_linpos[b0id])

            pb1 = j['pos_body1']
            vb1 = j['vec_body1']

            b1id = j['body_id1']
            if b1id != -1:
                # transform point on body to point in world
                pb1 = jnp.matmul(jnp.append(pb1,1), qr_linpos[b1id])
                vb1 = jnp.matmul(jnp.append(vb1,0), qr_linpos[b1id])

            d = pb1 - pb0
            dist_squared = jnp.sum(d*d)
            joint_stiffness = 300000.0 

            align = 1.0-jnp.sum(vb0*vb1)
            align_stiffness = 500.0

            joint_energy += 0.5 * joint_stiffness * dist_squared + 0.5 * align_stiffness * align

        ###########

        contact_energy = 0.0
        ext_force_energy = 0.0

        external_forces = system_def['external_forces']
        forcedBodyId = 23 // 2

        # Gravity
        
        massR = system_def['mass'].reshape(-1,4,4)

        gravity = system_def["gravity"]
        c_weighted = massR[:,3,3][:,None]*qr_linpos

        gravity_energy = -jnp.sum(c_weighted * gravity[None,:])

        return joint_energy + gravity_energy
    
    def real_kineticenergy(self, system_def, q):
        qRFull = jnp.concatenate((system_def['fixed_pos'],q)).reshape(-1,12,1)
        qr_vel = jnp.array([qRFull[:,1,1], qRFull[:,3,1], qRFull[:,5,1]])
        
        massR = system_def['mass'].reshape(-1,4,4)
        
        ke = 1/2*massR[:,3,3][:,None]*jnp.square(qr_vel)

        return ke
    
    def dissipation_fnc(self, n, c, q):
        qRFull = q.reshape(-1,12,1)
        qr_vel = jnp.array([qRFull[:,1,1], qRFull[:,3,1], qRFull[:,5,1]])
        
        D = 1/(n+1)*c*qr_vel
        return D

    def action(self, system_def, q):
        external_force = system_def['external_forces']['torque_strength']
        
        PE = self.potential_energy(system_def, q)
        KE = self.real_kineticenergy(system_def, q)
        dis_extforce = self.dissipation_fnc(0, external_force, q)

        return PE+KE+dis_extforce

    def kinetic_energy(self, system_def, q_dot):
        q_dotR = q_dot.reshape(-1,12,1)
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

        xr = jnp.concatenate((system_def['fixed_pos'],x)).reshape(-1,4,3)

        for bid in range(self.n_bodies):
            v = np.array(jnp.matmul(self.bodiesRen[bid]['W'], xr[bid]))
            f = np.array(self.bodiesRen[bid]['f'])

            ps_body = ps.register_surface_mesh("body" + prefix + str(bid), v, f)
            if transparency < 1.:
                ps_body.set_transparency(transparency)

            transform = np.identity(4)
            ps_body.set_transform( transform )
        
        return ps_body # not clear that anything needs to be returned

    def export(self, system_def, x, prefix=""):
        pass        

    def visualize_set_nice_view(system_def, q):
        # Set a Polyscope camera view which nicely looks at the scene
        
        # Example:
        # (could also be dependent on x)
        # ps.look_at((2., 1., 2.), (0., 0., 0.))

        ps.look_at((1.5, 1.5, 1.5), (0., -.2, 0.))
    
        pass
