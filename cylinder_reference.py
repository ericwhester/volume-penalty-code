import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
import dedalus.public as de
import matplotlib.pyplot as plt
import os
from scipy.special import erf
import time
import logging
root = logging.root
for h in root.handlers: h.setLevel("INFO")
logger = logging.getLogger(__name__)

# Simulation parameters
Re = 100
N = [256,128]
boundaries = [1,4]
iterations, wall_time = 1000+1, 1*60*60
dt = 1e-3
print_freq = 10
sim_name = 'cylinder-reference'
data_dir = os.path.join('runs',sim_name)
if rank==0 and not os.path.isdir(data_dir): os.makedirs(data_dir)

# Create problem bases and domain
θbasis = de.Fourier('θ', N[0], interval=(-np.pi,np.pi), dealias=3/2)
rbasis = de.Chebyshev('r',N[1],interval=boundaries, dealias=3/2)
domain = de.Domain([θbasis,rbasis], grid_dtype=np.float64)
θ, r = domain.grids(domain.dealias)
θθ,rr = np.meshgrid(θ,r,indexing='ij')

# Boundary condition functions
from dedalus.core.operators import GeneralFunction
# Define GeneralFunction subclass for time dependent boundary conditions
class ConstantFunction(GeneralFunction):
    def __init__(self, domain, layout, func, args=[], kw={}, out=None,):
        super().__init__(domain, layout, func, args=[], kw={}, out=None,)

    def meta_constant(self, axis):
        return True

def normalized_mask(x): return np.piecewise(x, [x<=-1,(x>-1)&(x<1),x>=1],
                                           [lambda x:1,
                                            lambda x:(1-erf(np.sqrt(np.pi)*x/np.sqrt(1-x**2)))/2,
                                            lambda x:0])
def bc_func(solver): return normalized_mask(1-solver.sim_time/2)
def oscillation_func(solver): return np.sin(2*solver.sim_time/np.pi)

bc = ConstantFunction(domain, layout='g', func=bc_func)
oscillation = ConstantFunction(domain, layout='g', func=oscillation_func)

disk = de.IVP(domain, variables=['u','v','p','q'], ncc_cutoff=1e-10)
disk.meta[:]['r']['dirichlet'] = True

# Parameters
params = [1,10,Re,bc,np.pi] + N
param_names = ['R0','R1','Re','bc','π','Nθ','Nr']
for param, param_name in zip(params, param_names): 
    disk.parameters[param_name] = param

disk.substitutions['pr'] = "p - 0.5*(u*u+v*v)"
disk.substitutions['qr'] = "q/r"
disk.substitutions['c'] = "cos(θ)"
disk.substitutions['s'] = "sin(θ)"
disk.substitutions['fpr'] = "-pr"
disk.substitutions['fpθ'] = "0"
disk.substitutions['fvr'] = "0"
disk.substitutions['fvθ'] = "(dθ(u) + r*dr(v) - v)/(Re*r)"
disk.substitutions['φ'] = "(π/2)*(1-cos(2*t/π))"
disk.substitutions['ω'] = "sin(2*t/π)"
disk.substitutions['α'] = "(2/π)*cos(2*t/π)"

disk.add_equation("dr(r*u) + dθ(v) = 0")
disk.add_equation("r*r*dt(u) + (1/Re)*dθ(q) +       r*r*dr(p) =  r*v*q")
disk.add_equation("r*r*dt(v) - (1/Re)*(r*dr(q) - q) + r*dθ(p) = -r*u*q")
disk.add_equation("q - dr(r*v) + dθ(u) = 0")

# Boundary conditions                                                       
disk.add_bc("left(u)  = 0")
disk.add_bc("left(v)  = ω")
disk.add_bc("right(u) = bc*cos(θ)", condition="(nθ != 0)")
disk.add_bc("right(v) =-bc*sin(θ)")
disk.add_bc("right(p) = 0", condition="(nθ == 0)")

# Build timestepper and solver
ts = de.timesteppers.SBDF3
solver = disk.build_solver(ts)
solver.stop_sim_time, solver.stop_wall_time, solver.stop_iteration = np.inf, wall_time, iterations

# Initialize variables
bc.original_args = bc.args = [solver]
oscillation.original_args = oscillation.args = [solver]
u, v, p, q = (solver.state[name] for name in disk.variables)
for field in [u,v,p,q]: 
    field.set_scales(domain.dealias)
    field['g'] = 0

# Save state variables
analysis = solver.evaluator.add_file_handler('{}/data-{}'.format(data_dir,sim_name), 
                                             iter=10, max_writes=200,mode='overwrite')
for task in disk.variables: analysis.add_task(task)
analysis.add_task("ω")
analysis.add_task("φ")
analysis.add_task("α")

# Save force calcs
forces = solver.evaluator.add_file_handler('{}/force-{}'.format(data_dir,sim_name), 
                                           iter=1, max_writes=iterations,mode='overwrite')
forces.add_task("integ(interp((c*fpr-s*fpθ)*r,r='left'),'θ')",name='Fpx')
forces.add_task("integ(interp((c*fvr-s*fvθ)*r,r='left'),'θ')",name='Fvx')
forces.add_task("integ(interp((s*fpr+c*fpθ)*r,r='left'),'θ')",name='Fpy')
forces.add_task("integ(interp((s*fvr+c*fvθ)*r,r='left'),'θ')",name='Fvy')
forces.add_task("integ(interp(fvθ*r*r,r='left'),'θ')",name='Tv')
forces.add_task("ω")
forces.add_task("φ")
forces.add_task("α")

# Save simulation parameters   
parameters = solver.evaluator.add_file_handler('{}/parameters-{}'.format(data_dir,sim_name), iter=np.inf, max_writes=np.inf,mode='overwrite')
for param_name in param_names: parameters.add_task(param_name)

# Run the simulation
start_time = time.time()

while solver.ok:
    solver.step(dt)
    if solver.iteration % print_freq == 0:
        logger.info('It:{:0>5d}, Time:{:.2f}'.format(solver.iteration, (time.time()-start_time)/60))
        if np.isnan(u['g'].max()): break
