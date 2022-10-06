import torch
from abc import ABC, abstractmethod
from systems.utils import get_mesh_pos
from systems.utils import jacobian
from plots.plot_functions import plot_ref
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

        :param x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        :param xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x 1 tensor of reference states
        :param uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x 1 tensor of reference controls

        :return: u: torch.Tensor
            batch_size x self.DIM_U x 1 tensor of contracting control inputs
        """
        u = self.controller(x, xref, uref)
        return u.type(torch.FloatTensor)

    def f_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor, noise=False) -> torch.Tensor:
        """
        Return the dynamics at the states x
            \dot{x} = f(x) = a(x) + b(x) * u(x, xref, uref)

        :param x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        :param xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x 1 tensor of reference states
        :param uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x 1 tensor of reference controls

        :return: f: torch.Tensor
            batch_size x self.DIM_U x 1 tensor of dynamics at x
        """

        f = self.a_func(x) + torch.bmm(self.b_func(x), self.u_func(x, xref, uref))
        if noise:
            noise_matrix = self.DIST.repeat(x.shape[0], 1, 1)
            noise = torch.bmm(noise_matrix, torch.randn(x.shape[0], self.DIST.shape[2], 1))
            f = f + noise
        return f.type(torch.FloatTensor)

    def fref_func(self, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        return self.a_func(xref) + torch.bmm(self.b_func(xref), uref.type(torch.FloatTensor))

    def dudx_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        """
        Return the Jacobian of the input at the states x
            du(x, xref, uref) / dx

        :param x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        :param xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x 1 tensor of reference states
        :param uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x 1 tensor of reference controls

        :return: f: torch.Tensor
            batch_size x self.DIM_U x self.DIM_X tensor of the Jacobian of u at x
        """

        if x.requires_grad:
            x = x.detach()
        x.requires_grad = True
        u = self.u_func(x, xref, uref)
        dudx = jacobian(u, x)
        x.requires_grad = False
        return dudx.type(torch.FloatTensor)

    def dfdx_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        """
        Return the Jacobian of the dynamics f at states x
            df/dx = da(x)/dx + db(x)/dx u(x) + b(x) du(x)/dx

        :param x: torch.Tensor
            batch_size x self.DIM_X x 1 tensor of state
        :param u: torch.Tensor
            batch_size x self.DIM_U x 1 tensor of controls

        :return: dfdx: torch.Tensor
            batch_size x self.DIM_X x self.DIM_X tensor of Jacobians at x
        """

        bs = x.shape[0]
        u = self.controller(x, xref, uref).type(torch.FloatTensor)
        dbdx_u = torch.zeros(bs, self.DIM_X, self.DIM_X)
        dbdx = self.dbdx_func(x)
        for i in range(self.DIM_X):
            dbdx_u[:, :, [i]] = torch.bmm(dbdx[:, :, :, i], u)
        dfdx = self.dadx_func(x) + dbdx_u + torch.bmm(self.b_func(x), self.dudx_func(x, xref, uref))
        return dfdx.type(torch.FloatTensor)

    def divf_func(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        """
        compute the divergence
        """
        dfdx = self.dfdx_func(x, xref, uref)
        div_f = dfdx.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        return div_f

    def get_next_x(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor, dt) -> torch.Tensor:
        """
        compute the next state
        """
        return x + self.f_func(x, xref, uref) * dt

    def get_next_xref(self, xref: torch.Tensor, uref: torch.Tensor, dt) -> torch.Tensor:
        """
        compute the next reference stat
        """
        return xref + self.fref_func(xref, uref) * dt

    def get_next_rho(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor, rho: torch.Tensor,
                     dt: int) -> torch.Tensor:
        """
        compute the next density value with LE
        """
        divf = self.divf_func(x, xref, uref)
        drhodt = -divf * rho
        with torch.no_grad():
            rho = rho + drhodt * dt
        return rho

    def get_next_rholog(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor, rholog: torch.Tensor,
                     dt: int) -> torch.Tensor:
        """
        compute the next log-density value with LE
        """
        divf = self.divf_func(x, xref, uref)
        drholog = torch.log(1 - divf * dt)
        #old update: drholog = - divf * dt (less accurate)
        with torch.no_grad():
            rholog = rholog + drholog
        return rholog

    def cut_xref_traj(self, xref_traj: torch.Tensor, uref_traj: torch.Tensor):
        """
        Cut xref and uref trajectories at the first time step when xref leaves the admissible state space

        :param xref: torch.Tensor
            batch_size (or 1) x self.DIM_X x args.N_sim tensor of reference state trajectories (assumed constant along first dimension)
        :param uref: torch.Tensor
            batch_size (or 1) x self.DIM_U x args.N_sim tensor of reference control trajectories (assumed constant along first dimension)

        :return:    xref: torch.Tensor
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
        uref_traj = uref_traj[:, :, :N_sim_cut-1]
        xref_traj = xref_traj[:, :, :N_sim_cut]
        return xref_traj, uref_traj

    def cut_x_rho(self, x, rho, pos):
        """
        Remove state trajectories which leave the admissible state space at specified time point

        :param x: torch.Tensor
            batch_size x self.DIM_X x args.N_sim tensor of state trajectories
        :param rho: torch.Tensor
            batch_size x 1 x args.N_sim tensor of density trajectories
        :param pos: integer
            index / timestep of trajectories which is tested

        :return:    x: torch.Tensor
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

    def sample_uref_traj(self, args, up=None):
        """
        sample random input parameters
        """

        N_sim = args.N_sim

        # parametrization by discretized input signals
        if "discr" in args.input_type:
            if args.input_type == "discr10":
                N_u = 10
            elif args.input_type == "discr5":
                N_u = 5
            length_u = args.N_sim_max // N_u #length of each input signal
            if up is None:
                up = 2 * torch.randn((1, self.DIM_U, N_u))
                up = up.clamp(self.UREF_MIN, self.UREF_MAX)
            elif up.dim() == 2:
                up = up.unsqueeze(0)
            uref_traj = torch.repeat_interleave(up, length_u, dim=2)

        # parametrization by polynomials of degree 3
        elif args.input_type == "polyn3":
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim).reshape(1, 1, -1).repeat(1, self.DIM_U, 1)
            if up is None:
                up = ((self.upOLYN_MAX - self.upOLYN_MIN) * torch.rand(4, self.DIM_U) + self.upOLYN_MIN).reshape(
                    1, 2, 1, -1)  # .repeat(1, 1, t.shape[2], 1)
            else:
                raise(NotImplementedError)
            # up[1, 2, 1, args.input_params_zero] = 0
            uref_traj = up[:, :, :, 0] * torch.ones_like(t) + up[:, :, :, 1] * t + \
                        up[:, :, :, 2] * t ** 2 + up[:, :, :, 3] * t ** 3

            for j in range(self.DIM_U):
                if torch.any(uref_traj[[0], [j], :] > self.UREF_MAX[0, j, 0]):
                    uref_traj[0, j, uref_traj[0, j, :] > self.UREF_MAX[0, j, 0]] = self.UREF_MAX[0, j, 0]
                if torch.any(uref_traj[[0], [j], :] < self.UREF_MIN[0, j, 0]):
                    uref_traj[0, j, uref_traj[0, j, :] < self.UREF_MIN[0, j, 0]] = self.UREF_MIN[0, j, 0]

        elif args.input_type == "sins5":
            num_sins = 5
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim)
            T_end = args.dt_sim * (args.N_sim_max - 1)
            if up is None:
                up = (self.UREF_MAX - self.UREF_MIN)[0, :, :] * torch.rand(self.DIM_U, num_sins) + self.UREF_MIN[0,:,:]
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            for i in range(num_sins):
                uref_traj[0,:,:] += up[:, [i]] * (torch.sin((i+1) * t / T_end * 2 * np.pi)).repeat(2, 1) #+ up[i+num_sins, :] * torch.cos((i+1) * t / T_end * 2 * np.pi)
            uref_traj = uref_traj.clip(self.UREF_MIN, self.UREF_MAX)

        elif "sincos" in args.input_type:
            if args.input_type == "sincos5":
                num_sins = 5
            elif args.input_type == "sincos3":
                num_sins = 3
            elif args.input_type == "sincos2":
                num_sins = 2
            t = torch.arange(0, args.dt_sim * N_sim - 0.001, args.dt_sim)
            T_end = args.dt_sim * (args.N_sim_max - 1)
            if up is None:
                up = (self.UREF_MAX - self.UREF_MIN)[0, :, :] * torch.rand(self.DIM_U, 2 * num_sins) + self.UREF_MIN[0,:,:]
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            for i in range(num_sins):
                uref_traj[0,:,:] += up[:, [i]] * (torch.sin((i+1) * t / T_end * 2 * np.pi)).repeat(2, 1) \
                                    + up[:, [i+num_sins]] * torch.cos((i+1) * t / T_end * 2 * np.pi)
            uref_traj[0, :, :] = 0.5 * (uref_traj[0,:,:] - uref_traj[0,:,[0]])
            uref_traj = uref_traj.clip(self.UREF_MIN, self.UREF_MAX)

        elif args.input_type == "sin1":
            t = torch.arange(0, args.dt_sim * N_sim, args.dt_sim)
            if up is None:
                up = torch.rand(4, self.DIM_U)
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            start = torch.round(args.N_sim_max * up[0, :])
            length = torch.round((args.N_sim_max - start) * up[1, :])
            amplitude = (2 * up[2, :] - 1) * self.USIN_AMPL
            wide = up[3, :] * self.USIN_WIDE
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
            if up is None:
                up = torch.rand(number, 3, self.DIM_U)
            else:
                raise(NotImplementedError)
            uref_traj = torch.zeros(1, self.DIM_U, N_sim)
            for i in range(number):
                start = torch.round(args.N_sim_max * up[i, 0, :])
                length = torch.round((args.N_sim_max - start) * up[i, 1, :])
                amplitude = (self.UREF_MAX - self.UREF_MIN).flatten() * up[i, 2, :] + self.UREF_MIN.flatten()
                for j in range(self.DIM_U):
                    uref_traj[0, j, int(start[j]):int(start[j] + length[j])] = amplitude[j]
        return uref_traj[:, :, :N_sim-1], up

    def sample_xref0(self):
        """
        sample initial reference state
        """
        return (self.XREF0_MAX - self.XREF0_MIN) * torch.rand(1, self.DIM_X, 1) + self.XREF0_MIN

    def compute_xref_traj(self, xref0: torch.Tensor, uref_traj: torch.Tensor, args, short=False) -> torch.Tensor:
        """
        compute the reference trajectory
        """
        N_sim = min(args.N_sim, uref_traj.shape[2]+1)
        dt = args.dt_sim
        xref_traj = xref0.repeat(1, 1, N_sim)

        for i in range(N_sim - 1):
            xref_traj[:, :, [i + 1]] = self.get_next_xref(xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
        #xref_traj = self.project_angle(xref_traj)
        if short:
            return xref_traj[:, :, ::args.factor_pred]
        return xref_traj

    def extend_xref_traj(self, xref_traj: torch.Tensor, uref_traj: torch.Tensor, dt) -> torch.Tensor:
        """
        extend reference trajectory
        """
        N_sim = uref_traj.shape[2]
        N_start = xref_traj.shape[2] - 1
        xref_traj = torch.cat((xref_traj, torch.zeros(1, xref_traj.shape[1], N_sim-N_start)), dim=2)

        for i in range(N_start, N_sim):
            xref_traj[:, :, [i + 1]] = self.get_next_xref(xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
            if torch.any(xref_traj[0, :, i+1] > self.X_MAX[0, :, 0]) or torch.any(xref_traj[0, :, i+1] < self.X_MIN[0, :, 0]):
                return None
        return xref_traj

    def up2ref_traj(self, xref0, up, args, short=True):
        """
        compute the reference trajectory from the input parameters
        """
        uref_traj, _ = self.sample_uref_traj(args, up=up)
        xref_traj = self.compute_xref_traj(xref0, uref_traj, args)
        if short:
            return uref_traj[:, :, ::args.factor_pred], xref_traj[:, :, ::args.factor_pred]
        else:
            return uref_traj, xref_traj

    def sample_xe(self, param):
        """
        sample the deviations of the reference trajectory
        """
        if isinstance(param, int):
            xe = torch.rand(param, self.DIM_X, 1) * (self.XE_MAX - self.XE_MIN) + self.XE_MIN
        return xe

    def sample_xe0(self, param):
        """
        sample the initial deviations of the reference trajectory
        """
        if isinstance(param, int):
            xe = torch.rand(param, self.DIM_X, 1) * (self.XE0_MAX - self.XE0_MIN) + self.XE0_MIN
        return xe

    def sample_x0(self, xref0, sample_size):
        """
        sample the initial states
        """
        xe0_max = torch.minimum(self.X_MAX - xref0, self.XE0_MAX)
        xe0_min = torch.maximum(self.X_MIN - xref0, self.XE0_MIN)
        if isinstance(sample_size, int):
            xe0 = torch.rand(sample_size, self.DIM_X, 1) * (xe0_max - xe0_min) + xe0_min
        else:
            xe0 = get_mesh_pos(sample_size).unsqueeze(-1) * (xe0_max - xe0_min) + xe0_min
        return xe0 + xref0

    def compute_density(self, xe0, xref_traj, uref_traj, dt, rho0=None, cutting=True, compute_density=True, log_density=False):
        """
        Get the density rho(x) starting at x0 with rho(x0)

        :param xe0: torch.Tensor
            batch_size x self.DIM_X x 1: tensor of initial error states
        :param xref_traj: torch.Tensor
            batch_size x self.DIM_U x N: tensor of reference states over N time steps
        :param uref_traj: torch.Tensor
            batch_size x self.DIM_U x N: tensor of controls
        :param rho0: torch.Tensor
            batch_size x 1 x 1: tensor of the density at the initial states
        :param dt:
            time step for integration

        :return:    xe_traj: torch.Tensor
            batch_size x self.DIM_X x N_sim: tensor of error state trajectories
                    rho_traj: torch.Tensor
            batch_size x 1 x N_sim: tensor of the densities at the corresponding states
        """

        x0 = xe0 + xref_traj[:, :, [0]]
        x_traj = x0.repeat(1, 1, uref_traj.shape[2]+1)
        if compute_density:
            if rho0 is None:
                if log_density:
                    rho0 = torch.zeros(x0.shape[0], 1, 1)  # equal initial density
                else:
                    rho0 = torch.ones(x0.shape[0], 1, 1)
            rho_traj = rho0.repeat(1, 1, uref_traj.shape[2]+1)
        else:
            rho_traj = None
        for i in range(uref_traj.shape[2]):
            if compute_density:
                if log_density:
                    rho_traj[:, 0, i + 1] = self.get_next_rholog(x_traj[:, :, [i]], xref_traj[:, :, [i]], uref_traj[:, :, [i]],
                                                      rho_traj[:, 0, i], dt)
                else:
                    rho_traj[:, 0, i + 1] = self.get_next_rho(x_traj[:, :, [i]], xref_traj[:, :, [i]], uref_traj[:, :, [i]],
                                                      rho_traj[:, 0, i], dt)
            with torch.no_grad():
                x_traj[:, :, [i + 1]] = self.get_next_x(x_traj[:, :, [i]], xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
        if compute_density and cutting:
            if log_density:
                if torch.any(rho_traj > 1e30):
                    print("clamp rho_traj to 1e30 (log density)")
                    rho_traj = rho_traj.clamp(max=1e30)
            else:
                if torch.any(rho_traj > 1e30) or torch.any(rho_traj < 0):
                    print("clamp rho_traj between 0 and 1e30 (no log density)")
                    rho_traj = rho_traj.clamp(min=0, max=1e30)
            if torch.any(rho_traj.isnan()):
                print("set nan in rho_traj to 1e30")
                rho_traj[rho_traj.isnan()] = 1e30
        return x_traj-xref_traj, rho_traj

    def get_valid_ref(self, args):
        """
        compute valid reference trajectory
        """
        while True:
            uref_traj, up = self.sample_uref_traj(args)  # get random input trajectory
            xref0 = self.sample_xref0()  # sample random xref
            xref_traj = self.compute_xref_traj(xref0, uref_traj, args)  # compute corresponding xref trajectory
            xref_traj, uref_traj = self.cut_xref_traj(xref_traj, uref_traj)  # cut trajectory where state limits are exceeded
            if xref_traj.shape[2] > 0.99 * args.N_sim:  # start again if reference trajectory is shorter than 0.9 * N_sim
                return up, uref_traj, xref_traj

    def get_valid_trajectories(self, sample_size, args, plot=False, log_density=True, compute_density=True):
        """
        compute valid trajectories
        """
        # get random input trajectory and compute corresponding state trajectory
        up, uref_traj, xref_traj = self.get_valid_ref(args)

        # compute corresponding  state and density trajectories
        xe0 = self.sample_xe0(sample_size)  # get random initial states
        xe_traj, rho_traj = self.compute_density(xe0, xref_traj, uref_traj, args.dt_sim,
                                                cutting=True, log_density=log_density,
                                                compute_density=compute_density)  # compute x and rho trajectories

        # save the results
        t = args.dt_sim * torch.arange(0, xe_traj.shape[2])
        if plot:
            plot_ref(xref_traj, uref_traj, 'test', args, self, x_traj=xe_traj + xref_traj, t=t, include_date=True)
        return xref_traj[:, :, ::args.factor_pred], rho_traj[:, :, ::args.factor_pred], uref_traj[:, :, ::args.factor_pred], \
               up, xe_traj[:, :, ::args.factor_pred], t[::args.factor_pred]

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
