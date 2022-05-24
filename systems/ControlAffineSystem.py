import torch
from abc import ABC, abstractmethod
import sys
from systems.utils import get_mesh_pos
# from data.trained_controller import model_CAR
from systems.utils import jacobian
import math
from plots.plot_functions import plot_scatter, plot_density_heatmap, plot_ref
import numpy as np

class ControlAffineSystem(ABC):
    """
    Represents an abstract control-affine dynamical system.

    A control-affine dynamical system is one where the state derivatives are affine in
    the control input, e.g.:

        dx/dt = a(x) + b(x) u

    These can be used to represent a wide range of dynamical systems, and they have some
    useful properties when it comes to designing controllers.
    """

    def __init__(self, systemname):
        self.controller = self.controller_wrapper(system=systemname)
        self.systemname = systemname


    def u_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        """
        Return the contracting control input

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x 1 tensor of reference states
        uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x 1 tensor of reference controls

        Returns
        -------
        u: torch.Tensor
            batch_size x self.DIM_U x 1 tensor of contracting control inputs
        """

        u = self.controller(x, xref, uref)
        return u

    def f_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        """
        Return the dynamics at the states x
            \dot{x} = f(x) = a(x) + b(x) * u(x, xref, uref)

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x 1 tensor of reference states
        uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x 1 tensor of reference controls

        Returns
        -------
        f: torch.Tensor
            batch_size x self.DIM_U x 1 tensor of dynamics at x
        """

        f = self.a_func(x) + torch.bmm(self.b_func(x), self.u_func(x, xref, uref))
        return f

    def dudx_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        """
        Return the Jacobian of the input at the states x
            du(x, xref, uref) / dx

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x 1 tensor of reference states
        uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x 1 tensor of reference controls

        Returns
        -------
        f: torch.Tensor
            batch_size x self.DIM_U x self.DIM_X tensor of the Jacobian of u at x
        """

        if x.requires_grad:
            x = x.detach()
        x.requires_grad = True
        u = self.controller(x, xref, uref)
        dudx = jacobian(u, x)
        x.requires_grad = False
        return dudx

    def dfdx_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        """
        Return the Jacobian of the dynamics f at states x
            df/dx = da(x)/dx + db(x)/dx u(x) + b(x) du(x)/dx

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        u: torch.Tensor
            batch_size x self.DIM_U x 1 tensor of controls

        Returns
        -------
        dfdx: torch.Tensor
            batch_size x self.DIM_X x self.DIM_X tensor of Jacobians at x
        """

        bs = x.shape[0]
        u = self.controller(x, xref, uref)
        dbdx_u = torch.zeros(bs, self.DIM_X, self.DIM_X)
        dbdx = self.dbdx_func(x)
        for i in range(self.DIM_X):
            dbdx_u[:, :, [i]] = torch.bmm(dbdx[:, :, :, i], u)
        dfdx = self.dadx_func(x) + dbdx_u + torch.bmm(self.b_func(x), self.dudx_func(x, xref, uref))
        return dfdx

    def divf_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        dfdx = self.dfdx_func(x, xref, uref)
        div_f = dfdx.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        return div_f

    def get_next_x(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor, dt) -> torch.Tensor:
        return x + self.f_func(x, xref, uref) * dt

    def get_next_xref(self, xref: torch.Tensor, uref: torch.Tensor, dt) -> torch.Tensor:
        return xref + (self.a_func(xref) + torch.bmm(self.b_func(xref), uref)) * dt

    def get_next_rho(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor, rho: torch.Tensor,
                     dt: int) -> torch.Tensor:
        divf = self.divf_func(x, xref, uref)
        drhodt = -divf * rho
        with torch.no_grad():
            rho = rho + drhodt * dt
        return rho

    def cut_xref_traj(self, xref_traj: torch.Tensor, uref_traj: torch.Tensor):
        """
        Cut xref and uref trajectories at the first time step when xref leaves the admissible state space

        Parameters
        ----------
        xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x args.N_sim tensor of reference state trajectories (assumed constant along first dimension)
        uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x args.N_sim tensor of reference control trajectories (assumed constant along first dimension)

        Returns
        -------
        xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x N_sim_cut tensor of shortened reference state trajectories
        uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x N_sim_cut tensor of shortened reference control trajectories
        """

        limits_exceeded = xref_traj.shape[2] * torch.ones(2 * self.DIM_X + 1)
        for j in range(self.DIM_X):
            N1 = ((xref_traj[0, j, :] > self.X_MAX[0, j, 0]).nonzero(as_tuple=True)[
                0])  # indices where xref > xmax for state j
            N2 = ((xref_traj[0, j, :] < self.X_MIN[0, j, 0]).nonzero(as_tuple=True)[
                0])  # indices where xref < xmin for state j

            # save first time step where state limits are exceeded
            if N1.shape[0] > 0:
                limits_exceeded[2 * j] = N1[0]
            if N2.shape[0] > 0:
                limits_exceeded[2 * j + 1] = N2[0]

        # cut trajectories at minimum time where state limits are exceeded
        N_sim_cut = int(limits_exceeded.min())
        uref_traj = uref_traj[:, :, :N_sim_cut]
        xref_traj = xref_traj[:, :, :N_sim_cut]
        return xref_traj, uref_traj

    def cut_x_rho(self, x, rho, pos):
        """
        Remove state trajectories which leave the admissible state space at specified time point

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.DIM_X x args.N_sim tensor of state trajectories
        rho: torch.Tensor
            batch_size x 1 x args.N_sim tensor of density trajectories
        pos: integer
            index / timestep of trajectories which is tested

        Returns
        -------
        x: torch.Tensor
            cut_batch_size x self.DIM_X x args.N_sim tensor of remaining state trajectories
        rho: torch.Tensor
            cut_batch_size x 1 x args.N_sim tensor of remaining density trajectories
        """

        # for j in range(self.DIM_X):
        #     mask = (x[:, j, pos] <= self.X_MAX[0, j, 0])  # indices where x < xmax for state j
        #     x = x[mask, :, :]
        #     rho = rho[mask, :, :]
        #     mask = (x[:, j, pos] >= self.X_MIN[0, j, 0])  # indices where x > xmin for state j
        #     x = x[mask, :, :]
        #     rho = rho[mask, :, :]
        mask = (rho[:, 0, pos] > 1e30)  # indices where rho infinite
        rho[mask, :, pos] = 1e30
        return x, rho

    def sample_uref_traj(self, args, u_params=None):
        N_sim = args.N_sim

        # parametrization by discretized input signals
        if args.input_type == "discr10":
            N_u = 10
            length_u = N_sim // N_u #length of each input signal
            if u_params is None:
                uref_traj = torch.zeros(1, self.DIM_U, N_sim)
                for j in range(self.DIM_U):
                    N_nonZero = torch.randint(0, N_u, (1,)) # number of non-zero input signals
                    indizes_nonZero = torch.randint(0, N_u, (N_nonZero,))
                    indizes_nonZero = list(set(indizes_nonZero.tolist()))
                    for i in indizes_nonZero:
                        interv_end = min((i + 1) * length_u, N_sim)
                        uref_traj[0, j, i * length_u: interv_end] = (self.UREF_MAX[0,j,0] - self.UREF_MIN[0,j,0]) * torch.rand((1,)) + self.UREF_MIN[0,j,0]
                u_params = uref_traj[0, :, :-1:length_u]
            else:
                uref_traj = torch.repeat_interleave(u_params, length_u, dim=2)

        # parametrization by polynomials of degree 3
        elif args.input_type == "polyn3":
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim).reshape(1, 1, -1).repeat(1, self.DIM_U, 1)
            if u_params is None:
                u_params = ((self.UPOLYN_MAX - self.UPOLYN_MIN) * torch.rand(4, self.DIM_U) + self.UPOLYN_MIN).reshape(
                    1, 2, 1, -1)  # .repeat(1, 1, t.shape[2], 1)
            else:
                raise(NotImplementedError)
            # u_params[1, 2, 1, args.input_params_zero] = 0
            uref_traj = u_params[:, :, :, 0] * torch.ones_like(t) + u_params[:, :, :, 1] * t + \
                        u_params[:, :, :, 2] * t ** 2 + u_params[:, :, :, 3] * t ** 3

            for j in range(self.DIM_U):
                if torch.any(uref_traj[[0], [j], :] > self.UREF_MAX[0, j, 0]):
                    uref_traj[0, j, uref_traj[0, j, :] > self.UREF_MAX[0, j, 0]] = self.UREF_MAX[0, j, 0]
                if torch.any(uref_traj[[0], [j], :] < self.UREF_MIN[0, j, 0]):
                    uref_traj[0, j, uref_traj[0, j, :] < self.UREF_MIN[0, j, 0]] = self.UREF_MIN[0, j, 0]

        elif args.input_type == "sins5":
            num_sins = 5
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim)
            T_end = t[-1]
            if u_params is None:
                u_params = (self.UREF_MAX - self.UREF_MIN)[0, :, :] * torch.rand(self.DIM_U, num_sins) + self.UREF_MIN[0,:,:]
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            for i in range(num_sins):
                uref_traj[0,:,:] += u_params[:, [i]] * (torch.sin((i+1) * t / T_end * 2 * np.pi)).repeat(2, 1) #+ u_params[i+num_sins, :] * torch.cos((i+1) * t / T_end * 2 * np.pi)
            uref_traj = uref_traj.clip(self.UREF_MIN, self.UREF_MAX)

        elif args.input_type == "sincos5":
            num_sins = 5
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim)
            T_end = t[-1]
            if u_params is None:
                u_params = (self.UREF_MAX - self.UREF_MIN)[0, :, :] * torch.rand(self.DIM_U, 2 * num_sins) + self.UREF_MIN[0,:,:]
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            for i in range(num_sins):
                uref_traj[0,:,:] += u_params[:, [i]] * (torch.sin((i+1) * t / T_end * 2 * np.pi)).repeat(2, 1) \
                                    + u_params[:, [i+num_sins]] * torch.cos((i+1) * t / T_end * 2 * np.pi)
            uref_traj[0, :, :] = 0.5* (uref_traj[0,:,:] - uref_traj[0,:,[0]])
            uref_traj = uref_traj.clip(self.UREF_MIN, self.UREF_MAX)

        elif args.input_type == "sin1":
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim)
            if u_params is None:
                u_params = torch.rand(4, self.DIM_U)
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            start = torch.round(N_sim * u_params[0, :])
            length = torch.round((N_sim - start) * u_params[1, :])
            amplitude = (2 * u_params[2, :] - 1) * self.USIN_AMPL
            wide = u_params[3, :] * self.USIN_WIDE
            for j in range(self.DIM_U):
                uref_traj[0, j, int(start[j]):int(start[j] + length[j])] = amplitude[j] * torch.sin(
                    wide[j] * t[:int(length[j])])
                if torch.any(uref_traj[[0], [j], :] > self.UREF_MAX[0, j, 0]):
                    uref_traj[0, j, uref_traj[0, j, :] > self.UREF_MAX[0, j, 0]] = self.UREF_MAX[0, j, 0]
                if torch.any(uref_traj[[0], [j], :] < self.UREF_MIN[0, j, 0]):
                    uref_traj[0, j, uref_traj[0, j, :] < self.UREF_MIN[0, j, 0]] = self.UREF_MIN[0, j, 0]

        elif "cust" in args.input_type:
            if args.input_type == "cust1":
                number = 1
            elif args.input_type == "cust2":
                number = 2
            elif args.input_type == "cust3":
                number = 3
            elif args.input_type == "cust4":
                number = 4
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim)
            if u_params is None:
                u_params = torch.rand(number, 3, self.DIM_U)
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            for i in range(number):
                start = torch.round(N_sim * u_params[i, 0, :])
                length = torch.round((N_sim - start) * u_params[i, 1, :])
                amplitude = (self.UREF_MAX - self.UREF_MIN).flatten() * u_params[i, 2, :] + self.UREF_MIN.flatten()
                for j in range(self.DIM_U):
                    uref_traj[0, j, int(start[j]):int(start[j] + length[j])] = amplitude[j]
        return uref_traj, u_params

    def sample_xref0(self):
        return (self.XREF0_MAX - self.XREF0_MIN) * torch.rand(1, self.DIM_X, 1) + self.XREF0_MIN

    def compute_xref_traj(self, xref0: torch.Tensor, uref_traj: torch.Tensor, args) -> torch.Tensor:
        N_sim = args.N_sim
        dt = args.dt_sim
        xref_traj = xref0.repeat(1, 1, N_sim)

        for i in range(N_sim - 1):
            xref_traj[:, :, [i + 1]] = self.get_next_xref(xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
        #xref_traj = self.project_angle(xref_traj)
        return xref_traj

    def u_params2xref_traj(self, xref0: torch.Tensor, u_params, args) -> torch.Tensor:
        uref_traj, _ = self.sample_uref_traj(self, args, u_params=u_params)
        xref_traj = self.compute_xref_traj(self, xref0, uref_traj, args)
        return xref_traj[:, :, ::args.factor_pred]


    def sample_x0(self, xref0, sample_size):
        xe0_max = torch.minimum(self.X_MAX - xref0, self.XE0_MAX)
        xe0_min = torch.maximum(self.X_MIN - xref0, self.XE0_MIN)
        if isinstance(sample_size, int):
            xe0 = torch.rand(sample_size, self.DIM_X, 1) * (xe0_max - xe0_min) + xe0_min
        else:
            xe0 = get_mesh_pos(sample_size).unsqueeze(-1) * (xe0_max - xe0_min) + xe0_min
        return xe0 + xref0

    def compute_density(self, x0, xref, uref, rho0, N_sim, dt, cutting=True, computeDensity=True):
        """
        Get the density rho(x) starting at x0 with rho(x0)

        Parameters
        ----------
        x0: torch.Tensor
            batch_size x self.DIM_X x 1: tensor of initial states
        xref: torch.Tensor
            batch_size x self.DIM_U x N: tensor of reference states over N time steps
        uref: torch.Tensor
            batch_size x self.DIM_U x N: tensor of controls
        rho0: torch.Tensor
            batch_size x 1 x 1: tensor of the density at the initial states
        N_sim:
            number of integration steps
        dt:
            time step for integration

        Returns
        -------
        x_traj: torch.Tensor
            batch_size x self.DIM_X x N_sim: tensor of state trajectories
        rho_traj: torch.Tensor
            batch_size x 1 x N_sim: tensor of the densities at the corresponding states
        """

        x_traj = x0.repeat(1, 1, N_sim)
        rho_traj = rho0.repeat(1, 1, N_sim)

        for i in range(uref.shape[2]-1):
            if computeDensity:
                rho_traj[:, 0, i + 1] = self.get_next_rho(x_traj[:, :, [i]], xref[:, :, [i]], uref[:, :, [i]],
                                                      rho_traj[:, 0, i], dt)
            with torch.no_grad():
                x_traj[:, :, [i + 1]] = self.get_next_x(x_traj[:, :, [i]], xref[:, :, [i]], uref[:, :, [i]], dt)
                if cutting:
                    x_traj, rho_traj = self.cut_x_rho(x_traj[:, :, :], rho_traj[:, [0], :], i + 1)
                    if x_traj.shape[0] < 2:
                       return x_traj[:, :, :i + 2], rho_traj[:, [0], :i + 2]
        #x_traj = self.project_angle(x_traj)
        return x_traj, rho_traj

    def get_valid_trajectories(self, sample_size, args, plot=False):
        # get random input trajectory and compute corresponding state trajectory
        valid = False
        while not valid:
            uref_traj, u_params = self.sample_uref_traj(args)  # get random input trajectory
            xref0 = self.sample_xref0()  # sample random xref
            xref_traj = self.compute_xref_traj(xref0, uref_traj, args)  # compute corresponding xref trajectory
            xref_traj, uref_traj = self.cut_xref_traj(xref_traj,
                                                        uref_traj)  # cut trajectory where state limits are exceeded
            if xref_traj.shape[2] < 0.8 * args.N_sim:  # start again if reference trajectory is shorter than 0.9 * N_sim
                continue

            # compute corresponding  state and density trajectories
            x0 = self.sample_x0(xref0, sample_size)  # get random initial states
            rho0 = torch.ones(x0.shape[0], 1, 1)  # equal initial density
            x_traj, rho_traj = self.compute_density(x0, xref_traj, uref_traj, rho0,
                                                      xref_traj.shape[2], args.dt_sim)  # compute x and rho trajectories
            if rho_traj.dim() < 2 or x_traj.shape[2] < 0.8 * args.N_sim:  # start again if x trajectories shorter than N_sim
                continue
            valid = True

        # save the results
        xref_traj = xref_traj[[0], :, :x_traj.shape[2]]
        uref_traj = uref_traj[[0], :, :x_traj.shape[2]]
        xe_traj = x_traj - xref_traj
        t = args.dt_sim * torch.arange(0, x_traj.shape[2])
        if plot:
            plot_ref(xref_traj, uref_traj, 'test', args, self, x_traj=xe_traj + xref_traj, t=t, include_date=True)
        return xref_traj[:, :, ::args.factor_pred], rho_traj[:, :, ::args.factor_pred], uref_traj[:, :, ::args.factor_pred], \
               u_params, xe_traj[:, :, ::args.factor_pred], t[::args.factor_pred]

    @abstractmethod
    def a_func(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dadx_func(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def b_func(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dbdx_func(self, x: torch.Tensor) -> torch.Tensor:
        pass
