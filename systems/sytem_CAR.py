import torch
from .ControlAffineSystem import ControlAffineSystem
import numpy as np
import sys

class Controller:
    def __init__(self, _controller, u_min, u_max):
        self.controller = _controller
        self.dist_state = 2
        self.U_MIN = u_min
        self.U_MAX = u_max

    def __call__(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor):
        xe = x[:, :4, :] - xref[:, :4, :]
        xe[:, self.dist_state, :] += x[:, 4, :]
        u = self.controller(x[:, :4, :], xe, uref)  # .squeeze(0)
        return u.clip(self.U_MIN, self.U_MAX)


class Car(ControlAffineSystem):
    DIM_X = 5
    DIM_U = 2

    def __init__(self, args=None):
        self.init_limits(args)
        self.controller = self.controller_wrapper()
        self.systemname = "CAR"

    def init_limits(self, args):
        dist_max = np.pi / 8

        # limits for controller and trajectory validity
        self.X_MIN = torch.tensor([-20., -20., -np.pi + 0.2, 0, -dist_max]).reshape(1, -1, 1)
        self.X_MAX = torch.tensor([20., 20., 3 * np.pi - 0.2, 10, dist_max]).reshape(1, -1, 1)
        self.XE_MIN = torch.tensor([-2, -2, -1, -1, -dist_max]).reshape(1, -1, 1)
        self.XE_MAX = torch.tensor([2, 2, 1, 1, dist_max]).reshape(1, -1, 1)

        # actuator limits
        self.UREF_MIN = torch.tensor([-3., -3.]).reshape(1, -1, 1)
        self.UREF_MAX = torch.tensor([3., 3.]).reshape(1, -1, 1)
        self.U_MIN = torch.tensor([-3., -3.]).reshape(1, -1, 1)
        self.U_MAX = torch.tensor([3., 3.]).reshape(1, -1, 1)

        # for sampling of initial position
        if args is not None:
            self.XREF0_MIN = torch.tensor([args.environment_size[0], args.environment_size[2], 0, 1., 0]).reshape(1, -1, 1)
            self.XREF0_MAX = torch.tensor([args.environment_size[1], args.environment_size[3], 2 * np.pi, 8., 0]).reshape(1, -1, 1)
        else:
            self.XREF0_MIN = torch.tensor([-10., -30., 0, 1, 0]).reshape(1, -1, 1)
            self.XREF0_MAX = torch.tensor([10., 30., 2 * np.pi, 8, 0]).reshape(1, -1, 1)
        self.XE0_MIN = 0.5 * self.XE_MIN
        self.XE0_MAX = 0.5 * self.XE_MAX

        # limits for trajectory validity for motion planning
        self.X_MIN_MP = torch.tensor([-9.9, -29.9, -np.pi + 0.2, 0.1, -np.inf]).reshape(1, -1, 1)
        self.X_MAX_MP = torch.tensor([9.9, 9.9, 3 * np.pi - 0.2, 9.9, np.inf]).reshape(1, -1, 1)


    def controller_wrapper(self):
        """
        Return neural contration controller

        Parameters
        ----------
        system: ControlAffineSystem

        Returns
        -------
        controller: function with needs the inputs x, xref and uref and outputs the contracting control input
        """

        _controller = self.load_controller()
        return Controller(_controller, self.U_MIN, self.U_MAX)

        # def controller(x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
        #     xe = x - xref
        #     xe = self.project_angle(xe)
        #     #x = self.project_angle(x)
        #     u = _controller(x, xe, uref)  # .squeeze(0)
        #     return u.clip(self.U_MIN, self.U_MAX)

        #return controller

    def load_controller(self):
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
        # if self.small_SS:
        #     controller_path = 'data/trained_controller/controller_CAR_smallSS.pth.tar' #'data/trained_controller/controller_CAR_ext3.pth.tar'
        controller_path = 'data/trained_controller/controller_CAR_3layers.pth.tar'
        _controller = torch.load(controller_path, map_location=torch.device('cpu'))
        _controller.cpu()
        return _controller

    def a_func(self, x: torch.Tensor) -> torch.Tensor:
        # x: bs x n x 1
        # a: bs x n x 1
        bs = x.shape[0]

        #x, y, theta, v, d = [x[:, i, 0] for i in range(Car.DIM_X)]
        a = torch.zeros(bs, Car.DIM_X, 1).type(x.type())
        a[:, 0, 0] = x[:, 3, 0] * torch.cos(x[:, 2, 0])
        a[:, 1, 0] = x[:, 3, 0] * torch.sin(x[:, 2, 0])
        return a.type(torch.FloatTensor)

    def dadx_func(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        #x, y, theta, v, d = [x[:, i, 0] for i in range(Car.DIM_X)]
        dadx = torch.zeros(bs, Car.DIM_X, Car.DIM_X).type(x.type())

        dadx[:, 0, 2] = - x[:, 3, 0] * torch.sin(x[:, 2, 0])
        dadx[:, 0, 3] = torch.cos(x[:, 2, 0])
        dadx[:, 1, 2] = x[:, 3, 0] * torch.cos(x[:, 2, 0])
        dadx[:, 1, 3] = torch.sin(x[:, 2, 0])
        return dadx.type(torch.FloatTensor)

    def b_func(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        b = torch.zeros(bs, Car.DIM_X, Car.DIM_U).type(x.type())

        b[:, 2, 0] = 1
        b[:, 3, 1] = 1
        return b.type(torch.FloatTensor)

    def dbdx_func(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        dbdx = torch.zeros(bs, Car.DIM_X, Car.DIM_U, Car.DIM_X).type(x.type())
        return dbdx.type(torch.FloatTensor)


    def project_angle(self, x_traj) -> torch.Tensor:
        with torch.no_grad():
            x_traj[:, 2, :] = (x_traj[:, 2, :] + np.pi) % (2 * np.pi) - np.pi
        return x_traj
