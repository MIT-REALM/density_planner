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

class MotionPlanner:
    def __init__(self, ego):
        self.ego = ego
        self.initial_traj = []
        self.improved_traj = []
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.initialize_logging()
        self.ego.env.get_gradient()

    def initialize_logging(self):
        self.path_log = make_path(self.ego.args.path_plot_motion, "mp_" + self.ego.args.mp_name)
        shutil.copyfile('hyperparams.py', self.path_log + 'hyperparams.py')
        shutil.copyfile('motion_planning/simulation_objects.py', self.path_log + 'simulation_objects.py')
        shutil.copyfile('motion_planning/optimization_objects.py', self.path_log + 'optimization_objects.py')
        shutil.copyfile('motion_planning/plan_motion.py', self.path_log + 'plan_motion.py')

    def find_traj(self):
        ### 1. generate random trajectory
        num_samples = self.ego.args.mp_numtraj
        if self.ego.args.input_type == "discr5":
            num_discr = 5
        elif self.ego.args.input_type == "discr10":
            num_discr = 10

        u_params = 0.5 * torch.randn((num_samples, 2, num_discr))
        u_params = u_params.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)

        ### 2. make trajectory valid
        check_bounds = False
        check_collision = False

        self.path_log_opt = make_path(self.path_log, "initialTraj")
        u_params, costs_dict = self.optimize(u_params, self.ego.args.mp_epochs, check_bounds=check_bounds, check_collision=check_collision)
        self.initial_traj = [u_params, costs_dict]
        u_params_best, cost_min = self.find_best()
        self.best_traj = [u_params_best, cost_min]

    def find_best(self, criterium="cost_sum", visualize=True):
        costs = self.initial_traj[1][criterium][-1]
        cost_min, idx = costs.min(dim=0)
        u_params_best = self.initial_traj[0][[idx], :, :]
        if visualize:
            uref_traj, xref_traj = self.ego.system.u_params2ref_traj(self.ego.xref0, u_params_best, self.ego.args, short=True)
            self.ego.visualize_xref(xref_traj, name="best_cost%.4f" % cost_min, save=True, show=False, folder=self.path_log_opt)
        return u_params_best, cost_min

    def improve_traj(self, u_params):
        self.path_log_opt = make_path(self.path_log, "improvedTraj")
        u_params, costs_dict = self.optimize(u_params, self.ego.args.mp_epochs_density, with_density=True, plot=True)
        self.improved_traj.append([u_params, costs_dict])
        cost_min = costs_dict["cost_sum"][-1]
        path_final = make_path(self.path_log, "finalTraj_cost%.4f" % cost_min)
        self.ego.animate_traj(u_params, folder=path_final)
        return u_params, cost_min

    def optimize(self, u_params, epochs, check_bounds=True, check_collision=True, plot=True, with_density=False):
        costs_dict = []
        self.rms = torch.zeros_like(u_params)
        self.momentum = torch.zeros_like(u_params)
        self.counts = torch.zeros(u_params.shape[0], 1, 1)
        u_params.requires_grad = True
        if check_bounds:
            self.check_bounds = torch.ones(u_params.shape[0], dtype=torch.bool)
        else:
            self.check_bounds = torch.zeros(u_params.shape[0], dtype=torch.bool)
        if check_collision:
            self.check_collision = torch.ones(u_params.shape[0], dtype=torch.bool)
        else:
            self.check_collision = torch.zeros(u_params.shape[0], dtype=torch.bool)

        for iter in range(epochs):
            if iter > 0:
                cost.sum().backward()
                with torch.no_grad():
                    u_update = self.optimizer_step(u_params.grad)
                    u_params -= torch.clamp(u_update, -self.ego.args.max_gradient, self.ego.args.max_gradient)
                    u_params.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)
                    u_params.grad.zero_()

            uref_traj, x_traj, rho_traj = self.get_traj(u_params, iter, with_density=with_density, plot=plot)
            cost, cost_dict = self.get_cost(uref_traj, x_traj, rho_traj)
            costs_dict.append(cost_dict)

        costs_dict = listDict2dictList(costs_dict)
        plot_cost(costs_dict, self.ego.args, folder=self.path_log_opt)
        return u_params, costs_dict

    def optimizer_step(self, grad):
        if self.ego.args.mp_lr_step != 0:
            lr = self.ego.args.mp_lr * (self.ego.args.mp_lr_step / (self.counts + self.ego.args.mp_lr_step))
        else:
            lr = self.ego.args.mp_lr

        grad = torch.clamp(grad, -1e15, 1e15)
        self.counts += 1
        if self.ego.args.mp_optimizer == "GD":
            step = lr * grad
        elif self.ego.args.mp_optimizer == "Adam":
            self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * grad
            self.rms = self.beta2 * self.rms + (1 - self.beta2) * (grad ** 2)
            momentum_corr = self.momentum / (1 - self.beta1 ** self.counts)
            rms_corr = self.rms / (1 - self.beta2 ** self.counts)
            step = lr * (momentum_corr / (torch.sqrt(rms_corr) + 1e-8))
        return step

    def get_traj(self, u_params, iter, with_density=False, plot=True):
        uref_traj, xref_traj = self.ego.system.u_params2ref_traj(self.ego.xref0.repeat(u_params.shape[0], 1, 1),
                                                                 u_params, self.ego.args)
        if plot:
            self.ego.visualize_xref(xref_traj, name="iter%d" % iter, save=True, show=False, folder=self.path_log_opt)

        if with_density:
            x_all, rho_unnorm = self.ego.predict_density(u_params, xref_traj)
            x_traj = x_all[:, :4, :]
            rho_traj = rho_unnorm / rho_unnorm.sum(dim=0).unsqueeze(0)
        else:
            x_traj = xref_traj[:, :4, :]
            rho_traj = None
        return uref_traj, x_traj, rho_traj

    def get_cost(self, uref_traj, x_traj, rho_traj):
        cost_uref = self.get_cost_uref(uref_traj)
        cost_goal, goal_reached = self.get_cost_goal(x_traj, rho_traj)
        cost_bounds = torch.zeros(uref_traj.shape[0])
        cost_coll = torch.zeros(uref_traj.shape[0])
        if not torch.all(self.check_bounds):
            idx_check = torch.logical_and(goal_reached, torch.logical_not(self.check_bounds))
            if torch.any(idx_check):
                self.check_bounds[idx_check] = True  # TO-DO: get the right indizes
                self.rms[idx_check, :, :] = 0
                self.momentum[idx_check, :, :] = 0
                self.counts[idx_check] = 0

        if torch.any(self.check_bounds):
            in_bounds = torch.zeros(uref_traj.shape[0], dtype=torch.bool)
            if rho_traj is None:
                x_check = x_traj[self.check_bounds, :, :]
            else:
                x_check = x_traj
            cost_bounds[self.check_bounds], in_bounds[self.check_bounds] = self.get_cost_bounds(x_check, rho_traj)
            if not torch.all(self.check_collision):
                idx_check = torch.logical_and(in_bounds, torch.logical_not(self.check_collision))
                if torch.any(idx_check):
                    self.check_collision[idx_check] = True  # TO-DO: get the right indizes
                    self.rms[idx_check, :, :] = 0
                    self.momentum[idx_check, :, :] = 0
                    self.counts[idx_check] = 0

        if torch.any(self.check_collision):
            # t1 = time.time()
            # for j in range(100):
            if rho_traj is None:
                x_check = x_traj[self.check_collision, :, :]
            else:
                x_check = x_traj
            cost_coll[self.check_collision] = self.get_cost_coll(x_check, rho_traj)  # for xref: 0.044s
            # print("Finished in %.4f s" % (time.time() - t1))

        cost = self.ego.args.weight_goal * cost_goal \
               + self.ego.args.weight_uref * cost_uref \
               + self.ego.args.weight_bounds * cost_bounds \
               + self.ego.args.weight_coll * cost_coll
        cost_dict = {
            "cost_sum": cost,
            "cost_goal": self.ego.args.weight_goal * cost_goal,
            "cost_uref": self.ego.args.weight_uref * cost_uref,
            "cost_bounds": self.ego.args.weight_bounds * cost_bounds,
            "cost_coll": self.ego.args.weight_coll * cost_coll
        }
        return cost, cost_dict

    def get_cost_uref(self, uref_traj):
        cost = self.ego.args.weight_uref_effort * (uref_traj ** 2).sum(dim=(1, 2))
        # valid = True
        # if torch.any(uref_traj < self.ego.system.UREF_MIN):
        #     idx = (uref_traj < self.ego.system.UREF_MIN).nonzero(as_tuple=True)
        #     cost += ((uref_traj[idx] - self.ego.system.UREF_MIN[0, idx[1], 0]) ** 2).sum()
        #     valid = False
        # if torch.any(uref_traj > self.ego.system.UREF_MAX):
        #     idx = (uref_traj > self.ego.system.UREF_MAX).nonzero(as_tuple=True)
        #     cost += ((uref_traj[idx] - self.ego.system.UREF_MAX[0, idx[1], 0]) ** 2).sum()
        #     valid = False
        return cost

    def get_cost_goal(self, x_traj, rho_traj=None):
        sq_dist = ((x_traj[:, :2, -1] - self.ego.xrefN[:, :2, 0]) ** 2).sum(dim=1)
        if rho_traj is None:
            cost_goal = sq_dist
        else:
            cost_goal = (rho_traj[:, 0, -1] * sq_dist).sum()
        close = cost_goal < self.ego.args.close2goal_thr
        cost_goal[torch.logical_not(close)] *= self.ego.args.weight_goal_far
        # if rho_traj is not None:
        #     close = torch.all(close)
        return cost_goal, close

    def get_cost_bounds(self, x_traj, rho_traj=None):
        # penalize leaving the admissible state space
        if rho_traj is None:
            cost = torch.zeros(x_traj.shape[0])
        else:
            cost = torch.zeros(1)

        in_bounds = torch.ones(x_traj.shape[0], dtype=torch.bool)
        if torch.any(x_traj < self.ego.system.X_MIN_MP):
            idx = (x_traj < self.ego.system.X_MIN_MP).nonzero(as_tuple=True)
            sq_error = ((x_traj[idx] - self.ego.system.X_MIN_MP[0, idx[1], 0]) ** 2)
            if rho_traj is None:
                if sq_error.dim() == 3:
                    cost[idx[0]] += sq_error.sum(dim=(1, 2))
                else:
                    cost[idx[0]] += sq_error
            else:
                cost += (rho_traj[idx[0], 0, idx[2]] * sq_error).sum()
            in_bounds[idx[0]] = False
        if torch.any(x_traj > self.ego.system.X_MAX_MP):
            idx = (x_traj > self.ego.system.X_MAX_MP).nonzero(as_tuple=True)
            sq_error = ((x_traj[idx] - self.ego.system.X_MAX_MP[0, idx[1], 0]) ** 2)
            if rho_traj is None:
                if sq_error.dim() == 3:
                    cost[idx[0]] += sq_error.sum(dim=(1, 2))
                else:
                    cost[idx[0]] += sq_error
            else:
                cost += (rho_traj[idx[0], 0, idx[2]] * sq_error).sum()
            in_bounds[idx[0]] = False
        if rho_traj is not None:
            in_bounds = torch.all(in_bounds)
        return cost, in_bounds

    def get_cost_coll(self, x_traj, rho_traj=None):
        if rho_traj is None:
            cost = torch.zeros(x_traj.shape[0])
        else:
            cost = torch.zeros(1)

        with torch.no_grad():
            gridpos_x, gridpos_y = pos2gridpos(self.ego.args, pos_x=x_traj[:, 0, :], pos_y=x_traj[:, 1, :])
            gridpos_x = torch.clamp(gridpos_x, 0, self.ego.args.grid_size[0]-1)
            gridpos_y = torch.clamp(gridpos_y, 0, self.ego.args.grid_size[1]-1)
        for i in range(x_traj.shape[2]):
            gradX = self.ego.env.grid_gradientX[gridpos_x[:, i], gridpos_y[:, i], i]
            gradY = self.ego.env.grid_gradientY[gridpos_x[:, i], gridpos_y[:, i], i]
            if torch.any(gradX != 0) or torch.any(gradY != 0):
                idx = torch.logical_or(gradX != 0, gradY != 0).nonzero(as_tuple=True)[0]
                des_gridpos_x = gridpos_x[idx, i] + 100 * gradX[idx]
                des_gridpos_y = gridpos_y[idx, i] + 100 * gradY[idx]
                des_pos_x, des_pos_y = gridpos2pos(self.ego.args, pos_x=des_gridpos_x, pos_y=des_gridpos_y)
                sq_dist = (des_pos_x - x_traj[idx, 0, i]) ** 2 + (des_pos_y - x_traj[idx, 1, i]) ** 2
                coll_prob = self.ego.env.grid[gridpos_x[idx, i], gridpos_y[idx, i], i]
                if rho_traj is None:
                    cost[idx] += (coll_prob * sq_dist)
                else:
                    cost += (rho_traj[idx, 0, i] * coll_prob * sq_dist).sum()
        return cost




