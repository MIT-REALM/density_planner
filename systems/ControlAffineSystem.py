import torch
from abc import ABC, abstractmethod, abstractproperty
import sys
from utils import get_grid
# from data.trained_controller import model_CAR
from utils import jacobian
import math


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

    @staticmethod
    def controller_wrapper(system: str):
        """
        Return neural contration controller

        Parameters
        ----------
        system: ControlAffineSystem

        Returns
        -------
        controller: function with needs the inputs x, xref and uref and outputs the contracting control input
        """

        _controller = ControlAffineSystem.load_controller(system)

        def controller(x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
            xe = x - xref
            u = _controller(x, xe, uref)  # .squeeze(0)
            return u

        return controller

    @staticmethod
    def load_controller(system: str):
        """
        load the pretrained neural controller

        Parameters
        ----------
        system: ControlAffineSystem

        Returns
        -------
        neural contraction controller
        """

        sys.path.append('data/trained_controller')
        controller_path = 'data/trained_controller/controller_' + system + '.pth.tar'
        _controller = torch.load(controller_path, map_location=torch.device('cpu'))
        _controller.cpu()
        return _controller

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
            N1 = ((xref_traj[0, j, :] > self.X_MAX[0, j, 0]).nonzero(as_tuple=True)[0]) # indices where xref > xmax for state j
            N2 = ((xref_traj[0, j, :] < self.X_MIN[0, j, 0]).nonzero(as_tuple=True)[0]) # indices where xref < xmin for state j

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

        for j in range(self.DIM_X):
            mask = (x[:, j, pos] <= self.X_MAX[0, j, 0]) # indices where xref > xmax for state j
            # if mask.dim() > 1:
            #     mask = mask.squeeze()
            x = x[mask, :, :]
            rho = rho[mask, :, :]
            mask = (x[:, j, pos] >= self.X_MIN[0, j, 0])# indices where xref > xmax for state j
            # if mask.dim() > 1:
            #     mask = mask.squeeze()
            x = x[mask, :, :]
            rho = rho[mask, :, :]
        mask = (rho[:, 0, pos].isfinite())# indices where rho infinite
        # if mask.dim() > 1:
        #     mask = mask.squeeze()

        x = x[mask, :, :]
        rho = rho[mask, :, :]
        return x, rho

    # def exceed_state_limits(self, x, rho=None):
    #     """
    #     Remove state trajectories which leave the admissible state space at some time point
    #
    #     Parameters
    #     ----------
    #     x: torch.Tensor
    #         batch_size x self.DIM_X x args.N_sim tensor of state trajectories
    #     rho: torch.Tensor
    #         batch_size x 1 x args.N_sim tensor of density trajectories
    #
    #     Returns
    #     -------
    #     x: torch.Tensor
    #         cut_batch_size x self.DIM_X x args.N_sim tensor of remaining state trajectories
    #     rho: torch.Tensor
    #         cut_batch_size x 1 x args.N_sim tensor of remaining density trajectories
    #     """
    #     for j in range(self.DIM_X):
    #         if (torch.any(x[:, [j], [0]] > self.X_MAX[0, j, 0]) or
    #                 torch.any(x[:, [j], [0]] < self.X_MIN[0, j, 0])):
    #             return True
    #         if rho is not None and torch.any(rho.isinf()):
    #             return True
    #     return False

    def sample_uref_traj(self, N_sim, N_u=None):
        if N_u is None:
            return (self.UREF_MAX - self.UREF_MIN) * torch.rand(1, self.DIM_U, N_sim) + self.UREF_MIN
        else:
            u = torch.ones(1, self.DIM_U, N_sim)
            for i in range(math.ceil(N_sim / N_u)):
                interv_end = min((i+1)*N_u, N_sim)
                u[0, :, i*N_u : interv_end] = (self.UREF_MAX - self.UREF_MIN) * torch.rand(1, self.DIM_U, 1) + self.UREF_MIN
            return u

    def sample_xref0(self):
        return (self.XREF0_MAX - self.XREF0_MIN) * torch.rand(1, self.DIM_X, 1) + self.XREF0_MIN

    def compute_xref_traj(self, xref0: torch.Tensor, uref_traj: torch.Tensor, N_sim, dt) -> torch.Tensor:
        xref_traj = xref0.repeat(1, 1, N_sim)

        for i in range(N_sim - 1):
            xref_traj[:, :, [i + 1]] = self.get_next_xref(xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
        return xref_traj

    def sample_ref_traj(self, xref0: torch.Tensor, N_sim, dt):
        xref_traj = xref0.repeat(1, 1, N_sim)
        uref_traj = self.sample_uref_traj(N_sim)

        for i in range(N_sim - 1):
            xref_traj[:, :, [i + 1]] = self.get_next_xref(xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
            rep = 0
            while self.exceed_state_limits(xref_traj[:, :, [i + 1]]):
                uref_traj[:, :, [i]] = self.sample_uref_traj(1)
                xref_traj[:, :, [i + 1]] = self.get_next_xref(xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
                rep += 1
                if rep >= 10:
                    return xref_traj[:, :, :i + 1], uref_traj[:, :, :i + 1]
        return xref_traj[:, :, :i + 1], uref_traj[:, :, :i + 1]

    def sample_x0(self, xref0, sample_size):
        xe0_max = torch.minimum(self.X_MAX - xref0, self.XE0_MAX)
        xe0_min = torch.maximum(self.X_MIN - xref0, self.XE0_MIN)
        if isinstance(sample_size, int):
            xe0 = torch.rand(sample_size, self.DIM_X, 1) * (xe0_max - xe0_min) + xe0_min
        else:
            xe0 = get_grid(sample_size) * (xe0_max - xe0_min) + xe0_min
        return xe0 + xref0

    def compute_density(self, x0, xref, uref, rho0, N_sim, dt, cutting=True):
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

        for i in range(N_sim-1):
            rho_traj[:, 0, i + 1] = self.get_next_rho(x_traj[:, :, [i]], xref[:, :, [i]], uref[:, :, [i]], rho_traj[:, 0, i], dt)
            with torch.no_grad():
                x_traj[:, :, [i + 1]] = self.get_next_x(x_traj[:, :, [i]], xref[:, :, [i]], uref[:, :, [i]], dt)
                if cutting:
                    x_traj, rho_traj = self.cut_x_rho(x_traj[:, :, :], rho_traj[:, [0], :], i+1)
                    if x_traj.shape[0] < 2:
                        return x_traj[:, :, :i + 2], rho_traj[:, [0], :i + 2]
        return x_traj, rho_traj

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
