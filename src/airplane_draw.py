import aerosandbox as asb
import numpy as np

# Define airplane parameters for aero calcs
name_airplane = "stoprotor_vtol"
airfoil_airplane = "clarky"
symmetry = True

# # Declare the airplane
# wing_airfoil = asb.Airfoil(airfoil_airplane)
# airplane = asb.Airplane(name = name_airplane,
#                                     xyz_ref = [0, 0, 0], # Cg location
#                                     wings=[
#                                         asb.Wing(
#                                             name = "Right Wing",
#                                             symmetric = False,
#                                             xsecs = [
#                                                 asb.WingXSec(
#                                                     xyz_le = [0.04, 0.05, -0.085],
#                                                     chord = -0.16,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [0.018, 0.170, -0.085],
#                                                     chord = -0.112,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [-0.0025, 0.3, -0.085],
#                                                     chord = -0.064,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                             ]
#                                         ),
#                                         asb.Wing(
#                                             name = "Left Wing",
#                                             symmetric = False,
#                                             xsecs = [
#                                                 asb.WingXSec(
#                                                     xyz_le = [-0.04, -0.05, -0.085],
#                                                     chord = 0.16,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [-0.018, -0.170, -0.085],
#                                                     chord = 0.112,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                                 asb.WingXSec(
#                                                     xyz_le = [0.0025, -0.3, -0.085],
#                                                     chord = 0.064,
#                                                     twist = 0,
#                                                     airfoil = wing_airfoil
#                                                 ),
#                                             ]
#                                         ),
#                                     ],
#                                     fuselages = [
#                                         asb.Fuselage(
#                                             name="Fuselage",
#                                             xsecs = [
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0, 0, 0],
#                                                     width = 0.5,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0.084, 0, 0],
#                                                     width = 0.5,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.084, 0, 0],
#                                                     width = 0.5,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.084, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.02, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0.084, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0.025, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[0, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.24,
#                                                     shape = 100, 
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.025, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
#                                                 asb.FuselageXSec(
#                                                     xyz_c=[-0.084, 0, 0],
#                                                     width = 0.1,
#                                                     height = 0.12, 
#                                                     shape = 100,
#                                                 ),
                                                
                                        
                                                
#                                             ]
#                                         )
#                                     ]
#                                 )

# Declare the airplane
wing_airfoil = asb.Airfoil(airfoil_airplane)
airplane = asb.Airplane(name = name_airplane,
                                    xyz_ref = [0, 0, 0], # Cg location
                                    wings=[
                                        asb.Wing(
                                            name = "Right Wing",
                                            symmetric = True,
                                            xsecs = [
                                                asb.WingXSec(
                                                    xyz_le = [0.04, 0.05, -0.085],
                                                    chord = -0.16,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [0.018, 0.170, -0.085],
                                                    chord = -0.112,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                                asb.WingXSec(
                                                    xyz_le = [-0.0025, 0.3, -0.085],
                                                    chord = -0.064,
                                                    twist = 0,
                                                    airfoil = wing_airfoil
                                                ),
                                            ]
                                        ),
                                        
                                    ],
                                    fuselages = [
                                        asb.Fuselage(
                                            name="Fuselage",
                                            xsecs = [
                                                asb.FuselageXSec(
                                                    xyz_c=[0, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.084, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.5,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.02, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0.025, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[0, 0, 0],
                                                    width = 0.1,
                                                    height = 0.24,
                                                    shape = 100, 
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.025, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                asb.FuselageXSec(
                                                    xyz_c=[-0.084, 0, 0],
                                                    width = 0.1,
                                                    height = 0.12, 
                                                    shape = 100,
                                                ),
                                                
                                        
                                                
                                            ]
                                        )
                                    ]
                                )

v = np.array([0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
vwind = np.zeros((3,))

vel_array = np.array([v[1], v[3], v[5]])
vel_relative = np.add(vel_array, vwind)
velocity = np.linalg.norm(vel_relative) # m/s
alpha = np.arctanh(np.true_divide(vel_relative[2], vel_relative[0])) # deg
beta = np.arctanh(np.true_divide(vel_relative[1], vel_relative[0])) # deg
vlm = asb.VortexLatticeMethod(airplane=airplane, 
                                op_point=asb.OperatingPoint(
                                    velocity = velocity, # in m/s
                                    alpha = alpha, # in degrees
                                    beta = beta, 
                                    p = v[7],
                                    q = v[9],
                                    r = v[11],
                                )
)

#airplane.draw()

aero = vlm.run()
vlm.draw()

body_force = aero['F_b']
body_moment = aero['M_b']

print(body_force)
print(body_moment)