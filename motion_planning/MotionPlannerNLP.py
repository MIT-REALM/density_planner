import numpy as np
import torch
from motion_planning.utils import pos2gridpos, check_collision, idx2time, gridpos2pos, traj2grid, shift_array, \
    pred2grid, get_mesh_sample_points, time2idx, sample_pdf, enlarge_grid, get_closest_free_cell, get_closest_obs_cell, make_path
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
from systems.utils import listDict2dictList
from plots.plot_functions import plot_ref, plot_grid, plot_cost
import pickle
from datetime import datetime
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import shutil
from systems.sytem_CAR import Car
import time
import logging
import sys
import casadi
from MotionPlanner import MotionPlanner


class MotionPlannerNLP(MotionPlanner):
    """
    class using NLP solver for motion planning
    """
    
    def __init__(self, ego, u0=None, xe0=None, plot=True, name="oracle", path_log=None, use_up=True):
        super().__init__(ego, plot=plot, name=name, path_log=path_log)
        self.rho0 = torch.ones(1, 1, 1)
        self.u0 = u0
        self.xe0 = xe0
        self.dt = ego.args.dt_sim
        self.factor_pred = ego.args.factor_pred
        self.use_up = use_up
        self.N = ego.args.N_sim // ego.args.factor_pred

        if ego.args.input_type == "discr10":
            self.num_discr = 10
        elif ego.args.input_type == "discr5":
            self.num_discr = 5

    def plan_motion(self):
        """
        start motion planner: call optimization/planning function and compute cost
        """

        logging.info("##### %s: Starting motion planning %s" % (self.name, self.ego.args.mp_name))
        t0 = time.time()
        u_traj, x_traj = self.solve_nlp()
        if self.plot and x_traj is not None:
            self.ego.visualize_xref(x_traj, name=self.name + " trajectory", save=True, show=False, folder=self.path_log)
        t_plan = time.time() - t0
        logging.info("%s: Planning finished in %.2fs" % (self.name, t_plan))
        if u_traj is not None:  # up is None if MPC didn't find solution
            logging.debug(u_traj)
            cost = self.validate_traj(u_traj)
        else:
            cost = None
        return u_traj, cost, t_plan

    def solve_nlp(self):
        quiet = False
        px_ref = self.ego.xrefN[0, 0, 0].item()
        py_ref = self.ego.xrefN[0, 1, 0].item()
        if self.use_up:
            N_u = (self.N - 1) // 10 + 1
        else:
            N_u = self.N

        opti = casadi.Opti()
        x = opti.variable(4, self.N + 1)  # state (x, y, psi, v)
        u = opti.variable(2, N_u)  # control (accel, omega)
        coll_prob = opti.variable(1, self.N)
        xstart = opti.parameter(4)

        street = True  # TO-DO: try False

        opti.minimize(
            self.weight_goal * ((x[0, self.N] - px_ref) ** 2 + (x[1, self.N] - py_ref) ** 2) +
            self.weight_coll * casadi.sumsqr(coll_prob) +
            self.weight_uref * casadi.sumsqr(u)
        )
        opti.subject_to(x[:4, 0] == xstart[:]) # CHANGED

        opti.subject_to(u[0, :] <= 3)  # accel
        opti.subject_to(u[0, :] >= -3)  # accel
        opti.subject_to(u[1, :] <= 3)  # omega
        opti.subject_to(u[1, :] >= -3)  # omega
        if street:
            x_min = -7
            x_max = 7
            y_min = -30
            y_max = 10
        else:
            x_min = self.ego.args.environment_size[0]  # self.ego.system.X_MIN[0, 0, 0]
            x_max = self.ego.args.environment_size[1]  # self.ego.system.X_MAX[0, 0, 0]
            y_min = self.ego.args.environment_size[2]  # self.ego.system.X_MIN[0, 1, 0]
            y_max = self.ego.args.environment_size[3]  # self.ego.system.X_MAX[0, 1, 0]

        opti.subject_to(x[0, :] <= x_max)  # x
        opti.subject_to(x[0, :] >= x_min)  # x
        opti.subject_to(x[1, :] <= y_max)  # y
        opti.subject_to(x[1, :] >= y_min)
        opti.subject_to(x[2, :] >= self.ego.system.X_MIN[0, 2, 0].item())  # theta # CHANGED
        opti.subject_to(x[2, :] <= self.ego.system.X_MAX[0, 2, 0].item())  # theta # CHANGED
        opti.subject_to(x[3, :] >= self.ego.system.X_MIN[0, 3, 0].item())  # v
        opti.subject_to(x[3, :] <= self.ego.system.X_MAX[0, 3, 0].item())  # v

        if self.u0 is not None:
            logging.info("%s: decision variables are initialized with good parameters" % self.name)
            uref_traj, xref_traj = self.ego.system.up2ref_traj(self.ego.xref0, self.u0, self.ego.args, short=True)
            if self.use_up:
                u_start = self.u0[0, :, :] # CHANGED
            else:
                u_start = uref_traj[0, :, :] # CHANGED
            opti.set_initial(u[:, :N_u], u_start[:, :N_u].numpy()) # CHANGED
            opti.set_initial(x[:, :self.N+1], xref_traj[0, :4, :self.N+1].numpy()) # CHANGED
        else:
            logging.info("%s: decision variables are initialized with zeros" % self.name)

        ix_min, iy_min = pos2gridpos(self.ego.args, pos_x=x_min, pos_y=y_min)
        ix_max, iy_max = pos2gridpos(self.ego.args, pos_x=x_max, pos_y=y_max)
        xgrid = np.arange(x_min, x_max + 0.0001, self.ego.args.grid_wide)
        ygrid = np.arange(y_min, y_max + 0.0001, self.ego.args.grid_wide)

        for k in range(self.N):  # timesteps
            grid_coll_prob = self.ego.env.grid[ix_min:ix_max + 1, iy_min:iy_max + 1, k].numpy().ravel(order='F')
            LUT = casadi.interpolant('name', 'linear', [xgrid, ygrid], grid_coll_prob)
            if self.use_up:
                u_k = k // 10
            else:
                u_k = k
            xk0 = x[0, k]
            xk1 = x[1, k]
            xk2 = x[2, k]
            xk3 = x[3, k]
            for j in range(self.factor_pred):
                xk0 = xk0 + xk3 * casadi.cos(xk2) * self.dt
                xk1 = xk1 + xk3 * casadi.sin(xk2) * self.dt
                xk2 = xk2 + u[0, u_k] * self.dt
                xk3 = xk3 + u[1, u_k] * self.dt
            opti.subject_to(x[0, k + 1] == xk0)  # x+=v*cos(theta)*dt
            opti.subject_to(x[1, k + 1] == xk1)  # y+=v*sin(theta)*dt
            opti.subject_to(x[2, k + 1] == xk2)  # theta+=omega*dt
            opti.subject_to(x[3, k + 1] == xk3)  # v+=a*dt
            opti.subject_to(coll_prob[0, k] == LUT(casadi.hcat([x[0, k + 1], x[1, k + 1]])))

        # optimizer setting
        p_opts = {"expand": False}
        s_opts = {"max_iter": self.ego.args.iter_NLP}
        if quiet:
            p_opts["print_time"] = 0
            s_opts["print_level"] = 0
            s_opts["sb"] = "yes"
        opti.solver("ipopt", p_opts, s_opts)


        if self.xe0 is None:
            self.xe0 = torch.zeros(1, self.ego.system.DIM_X, 1)
        x0 = self.xe0 + self.ego.xref0

        opti.set_value(xstart[:], x0[0, :4, 0].numpy()) # CHANGED
        try:
            sol1 = opti.solve()
        except:
            logging.info("%s: No solution found" % self.name)
            uref_traj = None
            x_traj = None
            return uref_traj, x_traj

        ud = sol1.value(u)
        xd = sol1.value(x)
        up = torch.from_numpy(ud).unsqueeze(0)
        if self.use_up:
            uref_traj, xref_traj = self.ego.system.up2ref_traj(x0, up, self.ego.args)
        else:
            uref_traj = up
            xref_traj = self.ego.system.compute_xref_traj(x0, uref_traj.repeat_interleave(10, dim=2), self.ego.args, short=True)
        x_traj = torch.from_numpy(xd).unsqueeze(0)
        logging.info("%s: Solution found. State trajectory error: %.2f" % (self.name, (x_traj - xref_traj[:, :4, :]).abs().sum()))

        return uref_traj, x_traj

    def get_traj(self, u_traj, xe0=None, name="traj", plot=True, folder=None):
        """
        compute trajectories from up

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory
        name: string
            name of parameter set for plotting
        plot: bool
            True if reference trajectory is plotted
        folder: string
            name of folder to save plot

        Returns
        -------
        uref_traj: torch.Tensor
            1 x 2 x N_sim_short -1
        xref_traj: torch.Tensor
            1 x 5 x N_sim_short
        x_traj: torch.Tensor
            1 x 4 x N_sim_short
        rho_traj: torch.Tensor
            1 x 1 x N_sim_short
        """
        if xe0 is None:
            xe0 = self.xe0

        xref_traj = self.ego.system.compute_xref_traj(self.ego.xref0+xe0, u_traj.repeat_interleave(10, dim=2), self.ego.args, short=True)
        if plot:
            if folder is None:
                folder = self.path_log
            self.ego.visualize_xref(xref_traj, name=name, save=True, show=False, folder=folder)
        x_traj = xref_traj
        rho_traj = torch.ones(1, 1, x_traj.shape[2])

        return u_traj, xref_traj, x_traj, rho_traj

    def validate_traj(self, u_traj):
        """
        evaluate input parameters (plot and compute final cost), assume that reference trajectory starts at ego.xref0+self.xe0

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory
        xe0: torch.Tensor
            batch_size x 4 x 1: tensor of initial deviation of reference trajectory
        rho0: torch.Tensor
            batch_size x 1 x 1: tensor of initial densities
        compute_density: bool
            True if rho_traj is computed

        Returns
        -------
        cost_dict: dictionary
            contains the unweighted cost tensors
        """

        if self.plot:
            path_final = make_path(self.path_log, self.name + "_finalTraj")
        else:
            path_final = None

        u_traj, xref_traj, x_traj, rho_traj = self.get_traj(u_traj, name="finalTraj", plot=self.plot, folder=path_final)
        if self.plot:
            self.ego.animate_traj(path_final, x_traj, x_traj, rho_traj)

        cost, cost_dict = self.get_cost(u_traj, x_traj, rho_traj)
        cost_dict = self.remove_cost_factor(cost_dict)
        logging.info("%s: True cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost_dict["cost_coll"],
                                                                                 cost_dict["cost_goal"],
                                                                                 cost_dict["cost_bounds"],
                                                                                 cost_dict["cost_uref"]))
        return cost_dict


class MotionPlannerMPC(MotionPlannerNLP):
    """
    class using NLP solver for motion planning
    """

    def __init__(self, ego, u0=None, xe0=None, plot=True, name="MPC", path_log=None, N_MPC=20):
        super().__init__(ego, u0=u0, xe0=xe0, plot=plot, name=name, path_log=path_log, use_up=False)
        self.N_MPC = N_MPC

    def plan_motion(self):
        """
        start motion planner: call optimization/planning function and compute cost
        """

        logging.info("##### %s: Starting motion planning %s" % (self.name, self.ego.args.mp_name))
        t0 = time.time()
        u_traj, x_traj = self.solve_nlp()
        if self.plot and x_traj is not None:
            self.ego.visualize_xref(x_traj, name=self.name + " trajectory", save=True, show=False, folder=self.path_log)
        t_plan = time.time() - t0
        logging.info("%s: Planning finished in %.2fs" % (self.name, t_plan))
        if u_traj is not None:  # up is None if MPC didn't find solution
            logging.debug(u_traj)
            cost = self.validate_traj(u_traj)
        else:
            cost = None
        return u_traj, cost, t_plan

    def solve_nlp(self):
        quiet = False
        px_ref = self.ego.xrefN[0, 0, 0].item()
        py_ref = self.ego.xrefN[0, 1, 0].item()
        N_u = self.N_MPC

        opti = casadi.Opti()
        x = opti.variable(4, self.N_MPC + 1)  # state (x, y, psi, v)
        u = opti.variable(2, N_u)  # control (accel, omega)
        coll_prob = opti.variable(1, self.N_MPC)
        xstart = opti.parameter(4)
        u_traj = np.zeros((1, 2, N_u))
        x_traj = np.zeros((1, 4, self.N))

        street = True  # TO-DO: try False

        opti.minimize(
            self.weight_goal * ((x[0, self.N_MPC] - px_ref) ** 2 + (x[1, self.N_MPC] - py_ref) ** 2) +
            self.weight_coll * casadi.sumsqr(coll_prob) +
            self.weight_uref * casadi.sumsqr(u)
        )
        opti.subject_to(x[:4, 0] == xstart[:])

        opti.subject_to(u[0, :] <= 3)  # accel
        opti.subject_to(u[0, :] >= -3)  # accel
        opti.subject_to(u[1, :] <= 3)  # omega
        opti.subject_to(u[1, :] >= -3)  # omega
        if street:
            x_min = -7
            x_max = 7
            y_min = -30
            y_max = 10
        else:
            x_min = self.ego.args.environment_size[0]  # self.ego.system.X_MIN[0, 0, 0]
            x_max = self.ego.args.environment_size[1]  # self.ego.system.X_MAX[0, 0, 0]
            y_min = self.ego.args.environment_size[2]  # self.ego.system.X_MIN[0, 1, 0]
            y_max = self.ego.args.environment_size[3]  # self.ego.system.X_MAX[0, 1, 0]

        opti.subject_to(x[0, :] <= x_max)  # x
        opti.subject_to(x[0, :] >= x_min)  # x
        opti.subject_to(x[1, :] <= y_max)  # y
        opti.subject_to(x[1, :] >= y_min)
        opti.subject_to(x[2, :] <= self.ego.system.X_MAX[0, 2, 0].item())  # v
        opti.subject_to(x[2, :] >= self.ego.system.X_MIN[0, 2, 0].item())  # v
        opti.subject_to(x[3, :] <= self.ego.system.X_MAX[0, 3, 0].item())  # v
        opti.subject_to(x[3, :] >= self.ego.system.X_MIN[0, 3, 0].item())  # v

        ix_min, iy_min = pos2gridpos(self.ego.args, pos_x=x_min, pos_y=y_min)
        ix_max, iy_max = pos2gridpos(self.ego.args, pos_x=x_max, pos_y=y_max)
        xgrid = np.arange(x_min, x_max + 0.0001, self.ego.args.grid_wide)
        ygrid = np.arange(y_min, y_max + 0.0001, self.ego.args.grid_wide)

        # optimizer setting
        p_opts = {"expand": False}
        s_opts = {"max_iter": self.ego.args.iter_MPC}
        if quiet:
            p_opts["print_time"] = 0
            s_opts["print_level"] = 0
            s_opts["sb"] = "yes"
        opti.solver("ipopt", p_opts, s_opts)
        if self.xe0 is None:
            self.xe0 = torch.zeros(1, self.ego.system.DIM_X, 1)
        x0 = self.xe0 + self.ego.xref0
        x_traj[:, :, [0]] = x0[:, :4, :].numpy()

        for k in range(self.N_MPC):  # timesteps
            grid_coll_prob = self.ego.env.grid[ix_min:ix_max + 1, iy_min:iy_max + 1, k].numpy().ravel(order='F')
            LUT = casadi.interpolant('name', 'linear', [xgrid, ygrid], grid_coll_prob)
            if self.use_up:
                u_k = k // 10
            else:
                u_k = k
            xk0 = x[0, k]
            xk1 = x[1, k]
            xk2 = x[2, k]
            xk3 = x[3, k]
            for j in range(self.factor_pred):
                xk0 = xk0 + xk3 * casadi.cos(xk2) * self.dt
                xk1 = xk1 + xk3 * casadi.sin(xk2) * self.dt
                xk2 = xk2 + u[0, u_k] * self.dt
                xk3 = xk3 + u[1, u_k] * self.dt
            opti.subject_to(x[0, k + 1] == xk0)  # x+=v*cos(theta)*dt
            opti.subject_to(x[1, k + 1] == xk1)  # y+=v*sin(theta)*dt
            opti.subject_to(x[2, k + 1] == xk2)  # theta+=omega*dt
            opti.subject_to(x[3, k + 1] == xk3)  # v+=a*dt
            opti.subject_to(coll_prob[0, k] == LUT(casadi.hcat([x[0, k + 1], x[1, k + 1]])))

        for k_start in range(self.N - self.N_MPC):

            # initialization
            if k_start == 0:
                opti.set_value(xstart[:], x0[0, :4, 0].numpy())
                if self.u0 is not None:
                    logging.info("%s: decision variables are initialized with good parameters" % self.name)
                    uref_traj, xref_traj = self.ego.system.up2ref_traj(self.ego.xref0, self.u0, self.ego.args,
                                                                       short=True)
                    opti.set_initial(u[:, :N_u], uref_traj[0, :, :N_u].numpy())
                    opti.set_initial(x[:, :], xref_traj[0, :4, :].numpy())
                else:
                    logging.info("%s: decision variables are initialized with zeros" % self.name)
            else:
                opti.set_initial(u[:, :N_u-1], ud[:, 1:])
                opti.set_initial(x[:, :self.N_MPC], xd[:, 1:])
                opti.set_value(xstart[:], xd[:, 1])

            try:
                sol1 = opti.solve()

            except:
                logging.info("%s: No solution found" % self.name)
                return None, None
            ud = sol1.value(u)
            xd = sol1.value(x)
            u_traj[0, :, k_start] = ud[:, 0]
            x_traj[0, :, k_start+1] = xd[:, 1]

        xref_traj = self.ego.system.compute_xref_traj(x0, u_traj.repeat_interleave(10, dim=2), self.ego.args, dhort=True)
        logging.info("%s: Solution found. State trajectory error: %.2f" % (
            self.name, (x_traj - xref_traj[:, :4, :]).abs().sum()))
        return u_traj, x_traj

    def get_traj(self, u_traj, xe0=None, name="traj", plot=True, folder=None):
        """
        compute trajectories from up

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory
        name: string
            name of parameter set for plotting
        plot: bool
            True if reference trajectory is plotted
        folder: string
            name of folder to save plot

        Returns
        -------
        uref_traj: torch.Tensor
            1 x 2 x N_sim_short -1
        xref_traj: torch.Tensor
            1 x 5 x N_sim_short
        x_traj: torch.Tensor
            1 x 4 x N_sim_short
        rho_traj: torch.Tensor
            1 x 1 x N_sim_short
        """
        if xe0 is None:
            xe0 = self.xe0

        xref_traj = self.ego.system.compute_xref_traj(self.ego.xref0 + xe0, u_traj.repeat_interleave(10, dim=2), self.ego.args, short=True)
        if plot:
            if folder is None:
                folder = self.path_log
            self.ego.visualize_xref(xref_traj, name=name, save=True, show=False, folder=folder)
        x_traj = xref_traj
        rho_traj = torch.ones(1, 1, x_traj.shape[2])

        return u_traj, xref_traj, x_traj, rho_traj

    def validate_traj(self, u_traj):
        """
        evaluate input parameters (plot and compute final cost), assume that reference trajectory starts at ego.xref0+self.xe0

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory
        xe0: torch.Tensor
            batch_size x 4 x 1: tensor of initial deviation of reference trajectory
        rho0: torch.Tensor
            batch_size x 1 x 1: tensor of initial densities
        compute_density: bool
            True if rho_traj is computed

        Returns
        -------
        cost_dict: dictionary
            contains the unweighted cost tensors
        """

        if self.plot:
            path_final = make_path(self.path_log, self.name + "_finalTraj")
        else:
            path_final = None

        u_traj, xref_traj, x_traj, rho_traj = self.get_traj(u_traj, name="finalTraj", plot=self.plot, folder=path_final)
        if self.plot:
            self.ego.animate_traj(path_final, x_traj, x_traj, rho_traj)

        cost, cost_dict = self.get_cost(u_traj, x_traj, rho_traj)
        cost_dict = self.remove_cost_factor(cost_dict)
        logging.info("%s: True cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost_dict["cost_coll"],
                                                                                     cost_dict["cost_goal"],
                                                                                     cost_dict["cost_bounds"],
                                                                                     cost_dict["cost_uref"]))
        return cost_dict
