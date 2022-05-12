import torch
from .ControlAffineSystem import ControlAffineSystem
import numpy as np
import sys


class Car(ControlAffineSystem):
    DIM_X = 4
    DIM_U = 2

    X_MIN = torch.tensor([-50., -50., -np.inf, 0]).reshape(1, -1, 1) #X_MIN = torch.tensor([-5., -5., -np.pi, 1]).reshape(1, -1, 1)
    X_MAX = torch.tensor([50., 50., np.inf, 10]).reshape(1, -1, 1) #X_MAX = torch.tensor([5., 5., np.pi, 2]).reshape(1, -1, 1)
    XE_MIN = torch.tensor([-2, -2, -1, -1]).reshape(1, -1, 1) #XE_MIN = torch.tensor([-1, -1, -1, -1]).reshape(1, -1, 1)
    XE_MAX = torch.tensor([2, 2, 1, 1]).reshape(1, -1, 1) #XE_MAX = torch.tensor([1, 1, 1, 1]).reshape(1, -1, 1)
    UREF_MIN = torch.tensor([-3., -4.]).reshape(1, -1, 1) #UREF_MIN = torch.tensor([-3., -3.]).reshape(1, -1, 1)
    UREF_MAX = torch.tensor([3., 4.]).reshape(1, -1, 1) #UREF_MAX = torch.tensor([3., 3.]).reshape(1, -1, 1)

    # parameters for polynomial parametrization of input
    UPOLYN_MIN = -0.5
    UPOLYN_MAX = 0.5

    # parameters for sinus parametrization of input
    USIN_AMPL = UREF_MAX.flatten() * 1.5
    USIN_WIDE = 3 * torch.tensor([1, 1])

    # for initialization
    XREF0_MIN = torch.tensor([-40., -40., -np.pi, 1]).reshape(1, -1, 1) #XREF0_MIN = torch.tensor([-2., -2., -1., 1.5]).reshape(1, -1, 1)
    XREF0_MAX = torch.tensor([40., 40., np.pi, 8]).reshape(1, -1, 1) #XREF0_MAX = torch.tensor([2., 2., 1., 1.5]).reshape(1, -1, 1)
    XE0_MIN = torch.tensor([-2, -2, -1, -1]).reshape(1, -1, 1) #XE0_MIN = torch.tensor([-1, -1, -1, -1]).reshape(1, -1, 1)
    XE0_MAX = torch.tensor([2, 2, 1, 1]).reshape(1, -1, 1) #XE0_MAX = torch.tensor([1, 1, 1, 1]).reshape(1, -1, 1)

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

        def controller(x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor) -> torch.Tensor:
            xe = x - xref
            xe = self.project_angle(xe)
            u = _controller(x, xe, uref)  # .squeeze(0)
            return u

        return controller

    @staticmethod
    def load_controller():
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
        controller_path = 'data/trained_controller/controller_CAR_ext3.pth.tar'
        _controller = torch.load(controller_path, map_location=torch.device('cpu'))
        _controller.cpu()
        return _controller

    def a_func(self, x: torch.Tensor) -> torch.Tensor:
        # x: bs x n x 1
        # a: bs x n x 1
        bs = x.shape[0]

        x, y, theta, v = [x[:, i, 0] for i in range(Car.DIM_X)]
        a = torch.zeros(bs, Car.DIM_X, 1).type(x.type())
        a[:, 0, 0] = v * torch.cos(theta)
        a[:, 1, 0] = v * torch.sin(theta)
        a[:, 2, 0] = 0
        a[:, 3, 0] = 0
        return a

    def dadx_func(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x, y, theta, v = [x[:, i, 0] for i in range(Car.DIM_X)]
        dadx = torch.zeros(bs, Car.DIM_X, Car.DIM_X).type(x.type())

        dadx[:, 0, 2] = - v * torch.sin(theta)
        dadx[:, 0, 3] = torch.cos(theta)
        dadx[:, 1, 2] = v * torch.cos(theta)
        dadx[:, 1, 3] = torch.sin(theta)
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
        x_traj[:, 2, :] = (x_traj[:, 2, :] + np.pi) % (2 * np.pi) - np.pi
        return x_traj
