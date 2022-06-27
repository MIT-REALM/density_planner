import torch
from .ControlAffineSystem import ControlAffineSystem
import numpy as np
import sys

class Controller:
    def __init__(self, _controller, u_min, u_max):
        self.controller = _controller
        self.U_MIN = u_min
        self.U_MAX = u_max

    def __call__(self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor):
        xe = x - xref
        # xe = self.project_angle(xe)
        # x = self.project_angle(x)
        u = self.controller(x[:, :4, :], xe[:, :4, :], uref)  # .squeeze(0)
        return u.clip(self.U_MIN, self.U_MAX)


class Car(ControlAffineSystem):
    DIM_X = 6
    DIM_STATES = 4
    DIM_U = 2

    small_SS = False
    d_max = 0.7
    d_max2 = 1.0
    state_dist = 0   # disturbed state
    state_dist2 = 1   # disturbed state
    if small_SS:
        X_MIN = torch.tensor([-5., -5., -np.pi, 1, -d_max, -d_max2]).reshape(1, -1, 1)
        X_MAX = torch.tensor([5., 5., np.pi, 2, d_max, d_max2]).reshape(1, -1, 1)
        XE_MIN = torch.tensor([-1, -1, -1, -1, -d_max, -d_max2]).reshape(1, -1, 1)
        XE_MAX = torch.tensor([1, 1, 1, 1, d_max, d_max2]).reshape(1, -1, 1)
        UREF_MIN = torch.tensor([-3., -3.]).reshape(1, -1, 1)
        UREF_MAX = torch.tensor([3., 3.]).reshape(1, -1, 1)
        XREF0_MIN = torch.tensor([-2., -2., -1., 1.5, 0]).reshape(1, -1, 1)
        XREF0_MAX = torch.tensor([2., 2., 1., 1.5, 0]).reshape(1, -1, 1)
        XE0_MIN = torch.tensor([-1, -1, -1, -1, -d_max, -d_max2]).reshape(1, -1, 1)
        XE0_MAX = torch.tensor([1, 1, 1, 1, d_max, d_max2]).reshape(1, -1, 1)
    else: #ext SS
        X_MIN = torch.tensor([-50., -50., -np.pi+0.2, 0, -d_max, -d_max2]).reshape(1, -1, 1)
        X_MAX = torch.tensor([50., 50., 3 * np.pi-0.2, 10, d_max, d_max2]).reshape(1, -1, 1)
        XE_MIN = torch.tensor([-2, -2, -1, -1, -d_max, -d_max2]).reshape(1, -1, 1)
        XE_MAX = torch.tensor([2, 2, 1, 1, d_max, d_max2]).reshape(1, -1, 1)
        UREF_MIN = torch.tensor([-3., -3.]).reshape(1, -1, 1)
        UREF_MAX = torch.tensor([3., 3.]).reshape(1, -1, 1)

        #for initialization
        XREF0_MIN = torch.tensor([-30., -10., 0, 1, 0, 0]).reshape(1, -1, 1)
        XREF0_MAX = torch.tensor([30., 10., 2*np.pi, 8, 0, 0]).reshape(1, -1, 1)
        XE0_MIN = torch.tensor([-2, -2, -1, -1, -d_max, -d_max2]).reshape(1, -1, 1)
        XE0_MAX = torch.tensor([2, 2, 1, 1, d_max, d_max2]).reshape(1, -1, 1)

    X_MIN_MP = torch.tensor([-9.9, -29.9, -np.pi + 0.1, 0.1]).reshape(1, -1, 1)
    X_MAX_MP = torch.tensor([9.9, 29.9, 3 * np.pi - 0.1, 9.9]).reshape(1, -1, 1)
    #'for startMiddle'
    # XREF0_MIN = torch.tensor([-10., -10., -np.pi, 1]).reshape(1, -1, 1)
    # XREF0_MAX = torch.tensor([10., 10., np.pi, 6]).reshape(1, -1, 1)

    U_MIN = torch.tensor([-4., -4.]).reshape(1, -1, 1)
    U_MAX = torch.tensor([4., 4.]).reshape(1, -1, 1)

    # parameters for polynomial parametrization of input
    UPOLYN_MIN = -0.5
    UPOLYN_MAX = 0.5

    # parameters for sinus parametrization of input
    USIN_AMPL = UREF_MAX.flatten() * 1.5
    USIN_WIDE = 3 * torch.tensor([1, 1])


    def __init__(self):
        self.controller = self.controller_wrapper()
        self.systemname = "CAR"


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
        if self.small_SS:
            controller_path = 'data/trained_controller/controller_CAR_smallSS.pth.tar' #'data/trained_controller/controller_CAR_ext3.pth.tar'
        else:
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
        if Car.state_dist is not None:
            a[:, Car.state_dist, 0] += x[:, 4, 0]
        if Car.state_dist2 is not None:
            a[:, Car.state_dist2, 0] += x[:, 5, 0]
        return a

    def dadx_func(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        #x, y, theta, v, d = [x[:, i, 0] for i in range(Car.DIM_X)]
        dadx = torch.zeros(bs, Car.DIM_X, Car.DIM_X).type(x.type())

        dadx[:, 0, 2] = - x[:, 3, 0] * torch.sin(x[:, 2, 0])
        dadx[:, 0, 3] = torch.cos(x[:, 2, 0])
        dadx[:, 1, 2] = x[:, 3, 0] * torch.cos(x[:, 2, 0])
        dadx[:, 1, 3] = torch.sin(x[:, 2, 0])
        if Car.state_dist is not None:
            dadx[:, Car.state_dist, 4] = 1
        if Car.state_dist2 is not None:
            dadx[:, Car.state_dist2, 5] = 1
        return dadx

    def b_func(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        b = torch.zeros(bs, Car.DIM_X, Car.DIM_U).type(x.type())

        b[:, 2, 0] = 1
        b[:, 3, 1] = 1
        return b

    def dbdx_func(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        dbdx = torch.zeros(bs, Car.DIM_X, Car.DIM_U, Car.DIM_X).type(x.type())
        return dbdx


    def project_angle(self, x_traj) -> torch.Tensor:
        with torch.no_grad():
            x_traj[:, 2, :] = (x_traj[:, 2, :] + np.pi) % (2 * np.pi) - np.pi
        return x_traj
