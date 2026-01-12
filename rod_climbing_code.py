# This script solves Rod Climbing problem for the J-S-G model A and B.
# The script generates own mesh and initialize an instance of a class that
# defines properties of the fluid, solver and more Firedrake needs.
# Ready to be run!

import os
import sys
import petsc4py
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from firedrake.__future__ import interpolate
import argparse

# MPI setting
comm = fd.COMM_WORLD
rank = fd.COMM_WORLD.rank
_print = print

# Parsing arguments
parser = argparse.ArgumentParser(description="Run solver with configurable FE degree")
parser.add_argument(
    "--fe_polynomial_degree",
    type=int,
    default=4,
    help="Finite element polynomial degree"
)
args = parser.parse_args()
fe_polynomial_degree = args.fe_polynomial_degree
print(f"Using FE polynomial degree = {fe_polynomial_degree}")



def print(x):
    if rank == 0:
        _print(x, flush=True)


def piecewise_polynomial_from_CGk_dofs(xs, vals, degree, samples_per_element=50):
    # Build a piecewise-polynomial curve of arbitrary CG(k) degree data.

    xs = np.asarray(xs).ravel()
    vals = np.asarray(vals).ravel()

    order = np.argsort(xs)
    xs = xs[order]
    vals = vals[order]

    k = degree
    dofs_per_elem = k + 1

    n = len(xs)
    # For N elements, node count = N*k + 1
    if (n - 1) % k != 0:
        print("Warning: DOF count does not match a CG(k) structure.")

    num_elems = (n - 1) // k

    x_fine_all = []
    y_fine_all = []

    # Loop over elements:
    for e in range(num_elems):
        idx0 = e * k
        idx1 = idx0 + dofs_per_elem
        x_nodes = xs[idx0:idx1]
        u_nodes = vals[idx0:idx1]

        if len(x_nodes) != dofs_per_elem:
            raise RuntimeError("Inconsistent DOF layout for CG(k).")

        # Build Lagrange basis at sample points.
        ts = np.linspace(0.0, 1.0, samples_per_element, endpoint=(e == num_elems - 1))

        # Convert reference t to physical x using Lagrange interpolation of coordinates.
        t_nodes = np.linspace(0.0, 1.0, dofs_per_elem)

        # Precompute Lagrange basis at ts
        L = np.zeros((len(ts), dofs_per_elem))
        for j in range(dofs_per_elem):
            numer = np.ones_like(ts)
            denom = 1.0
            for m in range(dofs_per_elem):
                if m != j:
                    numer *= ts - t_nodes[m]
                    denom *= t_nodes[j] - t_nodes[m]
            L[:, j] = numer / denom

        # Interpolate values and coordinates
        xe = L @ x_nodes
        ue = L @ u_nodes

        x_fine_all.append(xe)
        y_fine_all.append(ue)

    return np.concatenate(x_fine_all), np.concatenate(y_fine_all)


# Min/max cell diameters
def get_cell_diameter(mesh):
    import numpy as np

    # Compute something equivalent to cell diameter
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    dim = mesh.topological_dimension()
    h_ = fd.TestFunction(DG0)
    _h_sizes = fd.assemble(fd.Constant(1) * h_ * fd.dx)
    h_sizes = fd.Function(DG0)
    h_sizes.dat.data_wo[:] = _h_sizes.dat.data_ro
    np.power(h_sizes.dat.data, (1 / dim), out=h_sizes.dat.data)
    return h_sizes


# MIN MAX Functions
def get_global_min(fun):
    import numpy as np

    # Compute global h_min - for higher order mesh.
    h_sizes = fun.dat.data_ro
    local_min_size = np.min(h_sizes)
    mesh = fun.function_space().mesh()
    min_size = mesh.comm.allreduce(local_min_size, op=MPI.MIN)
    return min_size


def get_global_max(fun):
    import numpy as np

    # Compute global h_max - for higher order mesh.
    h_sizes = fun.dat.data_ro
    local_max_size = np.max(h_sizes)
    mesh = fun.function_space().mesh()
    max_size = mesh.comm.allreduce(local_max_size, op=MPI.MAX)
    return max_size


class Fluid(object):
    def __init__(self, name, mesh, t=0.0, dt=0.5, model="A", *args, **kwargs):
        self.mesh = mesh
        (x, z) = fd.SpatialCoordinate(mesh)  # x[0] = x (r), x[1] = z
        self.x = x
        self.k_degree = fe_polynomial_degree
        fd.dx = fd.dx(degree=self.k_degree * 4)

        # Build function spaces (Taylor-Hood)
        Ev = fd.VectorFunctionSpace(mesh, "CG", self.k_degree, dim=3)  # velocity
        Ep = fd.FunctionSpace(mesh, "DG", self.k_degree - 1)  # pressure
        Eb = fd.TensorFunctionSpace(
            mesh, "CG", self.k_degree - 1, shape=(3, 3)
        )  # left Cauchy-Green
        Eu = fd.FunctionSpace(mesh, "CG", self.k_degree)  # deformation
        W = fd.MixedFunctionSpace([Ev, Ep, Eb, Eu])
        self.W = W

        # Boundary conditions -------- everything no-slip, except cylinder and free surface
        noslip = fd.Constant(0)
        # Container wall
        bc1_r = fd.DirichletBC(W.sub(0).sub(0), noslip, 1)
        bc1_phi = fd.DirichletBC(W.sub(0).sub(1), noslip, 1)
        bc1_z = fd.DirichletBC(W.sub(0).sub(2), noslip, 1)
        # Container bottom
        bc2_r = fd.DirichletBC(W.sub(0).sub(0), noslip, 2)
        bc2_phi = fd.DirichletBC(W.sub(0).sub(1), noslip, 2)
        bc2_z = fd.DirichletBC(W.sub(0).sub(2), noslip, 2)
        # Symmetry wall
        bc3_r = fd.DirichletBC(W.sub(0).sub(0), noslip, 3)
        bc3_phi = fd.DirichletBC(W.sub(0).sub(1), noslip, 3)
        # Cylinder mantle
        self.vc_phi = v_mantle * x / radius
        self.vc_phi_ramping = fd.Constant(1.0)
        bc5_r = fd.DirichletBC(W.sub(0).sub(0), noslip, 5)
        bc5_phi = fd.DirichletBC(W.sub(0).sub(1), self.vc_phi * self.vc_phi_ramping, 5)
        # Cylinder bottom
        bc6_r = fd.DirichletBC(W.sub(0).sub(0), noslip, 6)
        bc6_phi = fd.DirichletBC(W.sub(0).sub(1), self.vc_phi * self.vc_phi_ramping, 6)
        bc6_z = fd.DirichletBC(W.sub(0).sub(2), noslip, 6)
        # Corresponding mesh BC
        bc1_mesh_z = fd.DirichletBC(W.sub(3), noslip, 1)
        bc2_mesh_z = fd.DirichletBC(W.sub(3), noslip, 2)
        bc6_mesh_z = fd.DirichletBC(W.sub(3), noslip, 6)

        self.bcs = [
            bc1_r,
            bc1_phi,
            bc1_z,
            bc1_mesh_z,
            bc2_r,
            bc2_phi,
            bc2_z,
            bc2_mesh_z,
            bc3_r,
            bc3_phi,
            bc5_r,
            bc5_phi,
            bc6_r,
            bc6_phi,
            bc6_z,
            bc6_mesh_z,
        ]

        # Facet normal, identity tensor and boundary measure
        n_2D = fd.FacetNormal(mesh)
        n = fd.as_vector([n_2D[0], 0, n_2D[1]])
        I = fd.Identity(3)

        # Define test function(s)
        (v_, p_, B_, mesh_u_temp_) = fd.TestFunctions(W)

        # current unknown at time step t
        w = fd.Function(W)
        (v, p, B, mesh_u_temp) = fd.split(w)

        # previous known time step solution
        w0 = fd.Function(W)
        (v0, p0, B0, mesh_u_temp0) = fd.split(w0)

        # Initial data
        w.subfunctions[0].interpolate(fd.Constant((0, 0, 0)))
        w.subfunctions[1].interpolate(fd.Constant(0))
        w.subfunctions[2].interpolate(I)
        w.subfunctions[3].interpolate(fd.Constant(0))
        w0.assign(w)

        mesh_u = fd.as_vector([0, 0.0, mesh_u_temp])
        mesh_u_ = fd.as_vector([0, 0.0, mesh_u_temp_])
        mesh_u0 = fd.as_vector([0, 0.0, mesh_u_temp0])

        # Benchmark parameters
        self.alpha = fd.Constant(0.0)  # alpha = 0 Oldroyd-B like model
        self.a = fd.Constant(1.0)  # a = 1 Oldroyd upper convected derivative
        delta1 = fd.Constant((1.0 - self.alpha) / We)
        delta2 = fd.Constant(self.alpha / We)

        # Define cylindrical gradient for scalar, vector a 2nd order tensorial field
        def L_scal(p):  # p = p(r,z)
            return fd.as_vector([p.dx(0), 0.0, p.dx(1)])

        def L_vec(v):
            cylindrical_second_column = fd.as_vector([-v[1] / x, v[0] / x, 0.0])
            cylindrical_nabla_vector = fd.as_tensor(
                [v.dx(0), cylindrical_second_column, v.dx(1)]
            ).T
            return cylindrical_nabla_vector

        def L_ten(B):
            cylindrical_second_matrix = (
                -fd.as_tensor(
                    [
                        [2 * B[0][1], B[1][1] - B[0][0], B[1][2]],
                        [B[1][1] - B[0][0], -2 * B[0][1], -B[0][2]],
                        [B[1][2], -B[0][2], 0.0],
                    ]
                )
                / x
            )
            cylindrical_nabla_tensor = fd.as_tensor(
                [B.dx(0), cylindrical_second_matrix, B.dx(1)]
            )
            return cylindrical_nabla_tensor

        def L_tenOnVec(B, v):
            first_matrix = fd.as_tensor(
                [
                    [
                        fd.dot(L_scal(B[0][0]), v),
                        fd.dot(L_scal(B[0][1]), v),
                        fd.dot(L_scal(B[0][2]), v),
                    ],
                    [
                        fd.dot(L_scal(B[0][1]), v),
                        fd.dot(L_scal(B[1][1]), v),
                        fd.dot(L_scal(B[1][2]), v),
                    ],
                    [
                        fd.dot(L_scal(B[0][2]), v),
                        fd.dot(L_scal(B[1][2]), v),
                        fd.dot(L_scal(B[2][2]), v),
                    ],
                ]
            )
            cylindrical_second_matrix = (
                -fd.as_tensor(
                    [
                        [2 * B[0][1], B[1][1] - B[0][0], B[1][2]],
                        [B[1][1] - B[0][0], -2 * B[0][1], -B[0][2]],
                        [B[1][2], -B[0][2], 0.0],
                    ]
                )
                / x
                * v[1]
            )
            return first_matrix + cylindrical_second_matrix

        # Define auxiliary variables for variational form
        def F_hat(mesh_u):
            return I + L_vec(mesh_u)

        def J_hat(mesh_u):
            return fd.det(F_hat(mesh_u))

        def F_hat_inv(mesh_u):
            return fd.inv(F_hat(mesh_u))

        def Dv(v, mesh_u):
            return 0.5 * (
                L_vec(v) * F_hat_inv(mesh_u) + F_hat_inv(mesh_u).T * L_vec(v).T
            )

        def Wv(v, mesh_u):
            return 0.5 * (
                L_vec(v) * F_hat_inv(mesh_u) - F_hat_inv(mesh_u).T * L_vec(v).T
            )

        # Define objective time derivatives a timestepping params
        self.t = t
        self.theta = 0.29289321881345
        self.dt = dt
        self.k = fd.Constant(1.0 / (dt * self.theta))

        def gordon_schowalter_objective_derivative_NOtime(a, v, mesh_u, B):
            return -a * (Dv(v, mesh_u) * B + B * Dv(v, mesh_u)) - (
                Wv(v, mesh_u) * B - B * Wv(v, mesh_u)
            )

        def time_der_vec(rho, mesh_u, mesh_u0, v, v0, v_):
            return (
                x * rho * J_hat(mesh_u) * self.k * fd.inner((v - v0), v_) * fd.dx
                + x
                * rho
                * J_hat(mesh_u)
                * fd.inner(
                    L_vec(v)
                    * fd.dot(F_hat_inv(mesh_u), v - (mesh_u - mesh_u0) * self.k),
                    v_,
                )
                * fd.dx
            )

        def time_der_ten(rho, mesh_u, mesh_u0, B, B0, B_):
            return (
                x * rho * J_hat(mesh_u) * self.k * fd.inner((B - B0), B_) * fd.dx
                + x
                * rho
                * J_hat(mesh_u)
                * fd.inner(
                    L_tenOnVec(
                        B, F_hat_inv(mesh_u) * (v - (mesh_u - mesh_u0) * self.k)
                    ),
                    B_,
                )
                * fd.dx
            )

        # Define RHS
        force = fd.Constant((0.0, 0.0, -1))
        n_hat = F_hat_inv(mesh_u).T * n
        n_hat_norm = fd.sqrt(fd.inner(n_hat, n_hat))
        n_hat = n_hat / n_hat_norm

        # Define Cauchy stress
        if model == "A":

            def T(p, v, mesh_u, B):
                return (
                    -p * I
                    + 2.0 * mu_beta * Dv(v, mesh_u)
                    + self.a * (1 - mu_beta) / We * (B - I)
                )

        elif model == "B":

            def T(p, v, mesh_u, B):
                return (
                    -p * I
                    + 2.0 * mu_beta * Dv(v, mesh_u)
                    + (1 - mu_beta) / We * (B - I)
                )

        else:
            raise ValueError(f"Unknown model: {model}")

        # Variational form without time derivatives
        def EQ(v, p, B, mesh_u, v_, p_, B_, mesh_u_):
            Eq1 = x * J_hat(mesh_u) * fd.tr(L_vec(v) * F_hat_inv(mesh_u)) * p_ * fd.dx
            Eq2 = (
                x
                * J_hat(mesh_u)
                * fd.inner(T(p, v, mesh_u, B) * F_hat_inv(mesh_u).T, L_vec(v_))
                * fd.dx
                - x * J_hat(mesh_u) / St * fd.inner(force, v_) * fd.dx
                + x
                * n_hat_norm
                / Ca
                * fd.inner(I - fd.outer(n_hat, n_hat), L_vec(v_) * F_hat_inv(mesh_u))
                * fd.ds(4, degree=self.k_degree * 4)
            )
            if model == "A":
                Eq3 = (
                    x
                    * J_hat(mesh_u)
                    * fd.inner(
                        gordon_schowalter_objective_derivative_NOtime(
                            self.a, v, mesh_u, B
                        )
                        + delta1 * (B - I)
                        + delta2 * (B - I) * B,
                        B_,
                    )
                    * fd.dx
                )
            elif model == "B":
                Eq3 = (
                    x
                    * J_hat(mesh_u)
                    * fd.inner(
                        gordon_schowalter_objective_derivative_NOtime(
                            self.a, v, mesh_u, B
                        )
                        + delta1 * (B - I)
                        + delta2 * (B - I) * B
                        - 2 * (1 - self.a) * Dv(v, mesh_u),
                        B_,
                    )
                    * fd.dx
                )
            else:
                raise ValueError(f"Unknown model: {model}")
            Eq4 = (
                x * fd.inner(mesh_u[2].dx(1), mesh_u_[2].dx(1)) * fd.dx
                - x * fd.inner(mesh_u[2].dx(1), mesh_u_[2]) * fd.ds(4)
                - x
                * fd.inner(
                    mesh_u[2] - mesh_u0[2] + (v[0] * mesh_u[2].dx(0) - v[2]) / self.k,
                    mesh_u_[2].dx(1)
                    - fd.Constant(1000.0) / fd.Constant(h_min) * mesh_u_[2],
                )
                * fd.ds(4)
            )
            return Eq1 + Eq2 + Eq3 + Eq4

        # combine variational forms with time derivative
        #
        #  dw/dt + F(t) = 0 is approximated using
        #  Implicit Glowinski three-step scheme (GL)
        #
        F = (
            time_der_vec(Re, mesh_u, mesh_u0, v, v0, v_)
            + time_der_ten(fd.Constant(1.0), mesh_u, mesh_u0, B, B0, B_)
            + EQ(v, p, B, mesh_u, v_, p_, B_, mesh_u_)
        )
        self.F = F

        # Prepare solver
        self.w = w
        self.w0 = w0
        print(f"Solving problem of size: {W.dim()}")
        self.fluid_problem = fd.NonlinearVariationalProblem(F, self.w, self.bcs)

        # Set solver parameters
        sp_lu = {
            "mat_type": "aij",
            "snes_type": "newtonls",
            # "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_max_it": 200,
            "snes_rtol": 1e-11,
            "snes_atol": 5e-10,
            "snes_linesearch_type": "basic",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            # Reuse Jacobian --- good tactics but not robust implementation
            "snes_lag_jacobian": 50,
            "snes_lag_jacobian_persists": True,
            "snes_lag_preconditioner": 50,  # reuse preconditioner too
            "snes_lag_preconditioner_persists": True,
            # Reuse Jacobian end
        }
        self.fluid_solver = fd.NonlinearVariationalSolver(
            self.fluid_problem, solver_parameters=sp_lu
        )

        # Prepare for the volume check
        self.VOL = 2 * 3.1415 * x * J_hat(mesh_u) * fd.dx
        self.Vref = fd.assemble(self.VOL)

        # Prepare for div(v) check
        self.divv_expr = x * J_hat(mesh_u) * fd.tr(L_vec(v) * F_hat_inv(mesh_u))

        # Create files for storing solution
        self.outfile = fd.VTKFile(f"results_{name}/fluid_data.pvd")

        # Net data
        self.height_data = []
        self.t_data = []

    def solve_step_GL(self):
        # solve the flow equations
        ## solve first step
        self.fluid_solver.solve()
        ## actualize solution
        q = fd.Constant((1.0 - self.theta) / self.theta)
        q0 = fd.Constant((2 * self.theta - 1.0) / self.theta)
        self.w0.assign(q * self.w + q0 * self.w0)
        self.w.assign(q * self.w + q0 * self.w0)
        ## solve second step
        self.fluid_solver.solve()

        # Move to next time step
        self.t += self.dt
        self.w0.assign(self.w)

    def save(self):
        # Extract solutions:
        (v, p, B, mesh_u) = self.w.subfunctions
        # Make vectors that are usefull for Paraview
        v_2D = fd.as_vector([v[0], v[2]])
        mesh_u_2D = fd.as_vector([0, mesh_u])

        # Save to file
        v_2D_projected = fd.assemble(
            interpolate(v_2D, fd.VectorFunctionSpace(mesh_f, "CG", 2))
        )
        v_2D_projected.rename("v", "velocity")
        mesh_u_2D_projected = fd.assemble(
            interpolate(mesh_u_2D, fd.VectorFunctionSpace(mesh_f, "CG", 2))
        )
        mesh_u_2D_projected.rename("u", "deformation")
        vphi_projected = fd.assemble(
            interpolate(v[1], fd.FunctionSpace(mesh_f, "CG", 2))
        )
        vphi_projected.rename("v_phi", "velocity")
        p.rename("p", "pressure")
        self.outfile.write(
            v_2D_projected, mesh_u_2D_projected, vphi_projected, p, time=self.t * Tref
        )

    def report(self):
        (v, p, B, mesh_u) = self.w.subfunctions
        # Print most important piece of information
        self.max_u = get_global_max(mesh_u)
        self.min_u = get_global_min(mesh_u)
        print(
            f"Maximal mesh deformation in z: H = {self.max_u}, minimal in z: H = {self.min_u}"
        )
        self.V = fd.assemble(self.VOL)
        print(
            f"Volume: {self.V}, Referential Volume {self.Vref}, ratio {self.V / self.Vref}"
        )

        # check div(v) condition
        divv = fd.sqrt(fd.assemble(self.divv_expr**2*fd.dx))
        print(f"||div(v)||_L2(\Omega_\chi): {divv}")

        # Store volume data
        self.height_data.append(self.max_u)
        self.t_data.append(self.t * Tref)

    def get_surface_shape2(self):
        (v, p, B, mesh_u_temp) = self.w.subfunctions
        g = mesh_u_temp
        x, y = fd.SpatialCoordinate(self.mesh)
        V = fd.FunctionSpace(self.mesh, "CG", self.k_degree)
        xf = fd.Function(V).interpolate(x)
        nodes = np.array(fd.DirichletBC(V, 1, 4).nodes, dtype=int)
        # DOF coordinates and function values at those DOFs
        xs = xf.dat.data_ro[nodes]
        vals = g.dat.data_ro[nodes]
        return (xs, vals)

    def get_surface_shape(self):
        (v, p, B, mesh_u_temp) = self.w.subfunctions
        g = mesh_u_temp
        x, y = fd.SpatialCoordinate(self.mesh)
        V = fd.FunctionSpace(self.mesh, "CG", self.k_degree)
        xf = fd.Function(V).interpolate(x)

        # nodes of boundary (or any DirichletBC)
        nodes = np.array(fd.DirichletBC(V, 1, 4).nodes, dtype=int)

        # Use data_ro_with_halos to safely access local + ghost DOFs
        xs_local = xf.dat.data_ro_with_halos[nodes]
        vals_local = g.dat.data_ro_with_halos[nodes]

        # MPI gather
        xs_all = comm.gather(xs_local, root=0)
        vals_all = comm.gather(vals_local, root=0)

        if comm.rank == 0:
            # flatten the lists
            xs_all = np.concatenate(xs_all)
            vals_all = np.concatenate(vals_all)

            # remove duplicates (ghost nodes from MPI splitting) based on xs
            _, inv = np.unique(xs_all, return_index=True)
            xs_all = xs_all[inv]
            vals_all = vals_all[inv]

            return xs_all, vals_all
        else:
            return None, None


############################## Main Part ####################################
pi = np.pi


# Generate Mesh and prepare boundaries
def generate_ngmesh_spline(lenght, height, radius, cyl_bot_height, maxh):
    from netgen.geom2d import SplineGeometry
    import netgen

    if fd.COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        fine_dining = 1
        tr = radius * fine_dining
        pnts = [
            (0.0, 0.0),
            (lenght, 0),
            (lenght, height),
            (radius + tr, height),
            (radius, height),
            (radius, height - tr),
            (radius, cyl_bot_height),
            (0, cyl_bot_height),
        ]
        p1, p2, p3, p4, p5, p6, p7, p8 = [geo.AppendPoint(*pnt) for pnt in pnts]
        lines = [
            [["line", p1, p2], 2, False],  # container bottom wall
            [["line", p2, p3], 1, False],  # container outer wall
            [["line", p3, p4], 4, False],  # container free surface
            [["line", p4, p5], 4, True],  # container free surface FINE
            [["line", p5, p6], 5, True],  # rotating cylinder mantle FINE
            [["line", p6, p7], 5, False],  # rotating cylinder mantle
            [["line", p7, p8], 6, False],  # rotating cylinder bottom
            [["line", p8, p1], 3, False],  # symmetry wall
        ]

        for geom, bc, fine in lines:
            if fine:
                geo.Append(geom, bc=bc, maxh=maxh / 50.0)
            else:
                geo.Append(geom, bc=bc)

        ngmesh = geo.GenerateMesh(maxh=maxh)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
    return ngmesh


# Set name for the output directory
name = "Oldroyd"
output_dir = f"results_{name}"
if rank == 0:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
comm.Barrier()

# Define end time of the simulation
t_end_dim = 5
radius_dim = 0.00635
lenght_dim = 23 * radius_dim
height_dim = 12 * radius_dim
maxh_dim = lenght_dim / 25.0

# Rescale!
Rref = radius_dim

# Set dimensions of the rod climbing problem
radius = 0.00635 / Rref
lenght = 23 * radius
height = 12 * radius
cyl_bot_height = 2 * radius
maxh = lenght / 25
ngmesh = generate_ngmesh_spline(lenght, height, radius, cyl_bot_height, maxh)
mesh_f = fd.Mesh(ngmesh, comm=fd.COMM_WORLD)

h_min = get_global_min(get_cell_diameter(mesh_f))
h_max = get_global_max(get_cell_diameter(mesh_f))
print(f"h_min = {h_min} \t h_max = {h_max}")
print("Mesh's ready and waiting")
mesh_id = "M1"

RPS_list = [0.5, 1.0, 1.3, 1.5, 1.7, 2.1, 2.6, 2.9]

if rank == 0:
    plt.figure()
    records = []


for RPS_step, RPS_dim in enumerate(RPS_list, start=1):
    print(
        f"---------------- This is omega = {np.round(RPS_dim,2)}, step {RPS_step} out of {len(RPS_list)} -------------"
    )
    # Recompute \omega and velocity on the rod
    omega_dim = 2.0 * pi * RPS_dim
    v_mantle_dim = omega_dim * radius_dim
    # Create physical parameters as global constants
    mu = fd.Constant(14.6)
    mu_ratio = 1 / 9
    mu_s = fd.Constant(mu / (1 + 1 / mu_ratio))
    print(f"solvent viscosity = {float(mu_s)}")
    mu_p = mu - mu_s
    print(f"polymer viscosity = {float(mu_p)}")
    print(f"ratio viscosity = {float(mu_s)/float(mu_p)}")
    mu_beta = mu_s / mu
    relax_time = 0.0162 / (1 - mu_ratio)
    We = fd.Constant(relax_time * omega_dim)
    print(f"Weissenberg number = {float(We)}")
    rho = fd.Constant(890.0)
    Re = fd.Constant(float(rho) / float(mu) * radius_dim * v_mantle_dim)
    print(f"Reynolds number = {float(Re)}")
    Gref = 9.81
    St = fd.Constant(float(mu) * omega_dim / float(rho) / Gref / radius_dim)
    print(f"Stokes number = {float(St)}")
    sigma = fd.Constant(0.0309)
    Ca = fd.Constant(mu * radius_dim * omega_dim / sigma)
    print(f"Cappilarity number = {float(Ca)}")
    print(f"Bonding number = {float(Ca/St)}")

    # Dimensionless time
    Vref = v_mantle_dim
    Tref = relax_time

    omega = omega_dim * Tref
    v_mantle = fd.Constant(v_mantle_dim / Vref)
    t_end = t_end_dim / Tref

    # Initializing an instance of a class that defines properties of the fluid, solver, etc.
    fluid = Fluid(name, mesh_f, t=0.0, dt=0.01 / Tref, model="A")
    fluid.a.assign(1.0)
    fluid.alpha.assign(0.0)

    # Timestepping loop
    time_step = 1
    while fluid.t <= t_end:
        print(f"t dim = {np.round(fluid.t*Tref,3)}, (progress {int(round(100 * fluid.t * Tref / t_end_dim))} %)")
        ramping_value = min(1, fluid.t * Tref) # ramping of rotations in the beggining
        fluid.vc_phi_ramping.assign(ramping_value)
        print(f"ramping assigning {np.round(ramping_value,2)}")
        fluid.solve_step_GL()
        if time_step % 10 == 0:
            fluid.save()
        fluid.report()
        time_step += 1

    (xs, vals) = fluid.get_surface_shape()
    if rank == 0:
        # Save to file first
        record = {
            "xs": xs,
            "vals": vals,
            "mesh_id": mesh_id,
            "k_degree": fluid.k_degree,
            "RPS": RPS_dim,
            "a": float(fluid.a),
        }
        records.append(record)

        # Line plot of k_degree polynomial
        xs_fine, vals_fine = piecewise_polynomial_from_CGk_dofs(
            xs, vals, fluid.k_degree
        )
        plt.plot(xs_fine, vals_fine, linewidth=5, label=f"RPS = {RPS_dim}", zorder=3)

        # Scatter plot of DoF
        vertex_mask = np.zeros(len(xs), dtype=bool)
        vertex_mask[:: fluid.k_degree] = True
        interior_mask = ~vertex_mask
        plt.scatter(
            xs[vertex_mask],
            vals[vertex_mask],
            marker="x",
            linewidths=0.2,
            s=20,
            color="black",
            zorder=7,
        )
        plt.scatter(
            xs[interior_mask],
            vals[interior_mask],
            marker=".",
            linewidths=0.2,
            s=10,
            color="black",
            zorder=5,
        )

    # At the end of the simulation plot max climbing height mesh_u_z.at(radius, height)
    if rank == 0:
        plt.figure()
        plt.title(f"RPS = {RPS_dim}")
        plt.xlabel("t [s]")
        plt.ylabel("H [1]")
        plt.plot(fluid.t_data, fluid.height_data)
        plt.savefig(
            f"{output_dir}/climb_height_omega{RPS_dim}_a{np.round(float(fluid.a),3)}_alpha{np.round(float(fluid.alpha),3)}.pdf",
            bbox_inches="tight",
        )
        plt.close()

if rank == 0:
    # Finish plot and file saving
    np.savez(
        f"{output_dir}/surface_shapes_all_a{np.round(float(fluid.a),3)}_alpha{np.round(float(fluid.alpha),3)}.npz",
        records=np.array(records, dtype=object),
    )
    print(
        f"Compiled results saved to {output_dir}/surface_shapes_all_a{np.round(float(fluid.a),3)}_alpha{np.round(float(fluid.alpha),3)}.npz"
    )

    plt.xlim(1.0, 2.0)
    plt.xlabel("r [1]")
    plt.ylabel("h(r,t=inf) [1]")
    plt.grid(True, zorder=1)
    plt.legend()
    plt.savefig(
        f"{output_dir}/surfaces_a{np.round(float(fluid.a),3)}_alpha{np.round(float(fluid.alpha),3)}.pdf"
    )
    plt.close()
    print(
        f"Surfaces plot saved to {output_dir}/surfaces_a{np.round(float(fluid.a),3)}_alpha{np.round(float(fluid.alpha),3)}.pdf"
    )
