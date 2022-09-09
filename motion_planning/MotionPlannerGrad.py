import numpy as np
import torch
from motion_planning.utils import pos2gridpos, gridpos2pos, make_path
from systems.utils import listDict2dictList
from plots.plot_functions import plot_cost
import time
import logging
from motion_planning.MotionPlanner import MotionPlanner


class MotionPlannerGrad(MotionPlanner):
    """
    class to use proposed density planner for motion planning
    """
    def __init__(self, ego, path_log=None, name="grad"):
        super().__init__(ego, name=name, path_log=path_log)
        self.initial_traj = []
        self.improved_traj = []
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.plot_cost = ego.args.mp_plot_cost

    def plan_motion(self):
        """
        start motion planner: call optimization/planning function and compute cost
        """

        logging.info("")
        logging.info("##### %s: Starting motion planning %s" % (self.name, self.ego.args.mp_name))
        tall = time.time()

        logging.debug("%s: Optimizing %d Random Trajectories without density" % (self.name, self.ego.args.mp_numtraj))
        t0 = time.time()
        up_best, cost_min = self.find_initial_traj()
        t_init = time.time() - t0
        logging.debug("%s: Initilialization finished in %.2fs" % (self.name, t_init))
        logging.debug("%s: Best Trajectory with cost %.4f:" % (self.name, cost_min))
        logging.debug(up_best)

        logging.debug("%s: Improving input parameters with density prediction" % self.name)
        t0 = time.time()
        up, cost = self.optimize_traj(up_best)
        t_opt = time.time() - t0
        logging.debug("%s: Optimization finished in %.2fs" % (self.name, t_opt))

        t_plan = time.time() - tall
        logging.info("%s: Planning finished in %.2fs" % (self.name, t_plan))
        logging.info("%s: Final cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost["cost_coll"],
                                                                                  cost["cost_goal"],
                                                                                  cost["cost_bounds"],
                                                                                  cost["cost_uref"]))
        logging.debug(up)
        cost = self.validate_ref(up)
        return up, cost, t_plan

    def find_initial_traj(self):
        if self.plot:
            self.path_log_opt = make_path(self.path_log, self.name + "_initialTraj")

        ### 1. generate random trajectory
        num_samples = self.ego.args.mp_numtraj
        if self.ego.args.input_type == "discr5":
            num_discr = 5
        elif self.ego.args.input_type == "discr10":
            num_discr = 10

        up = 0.5 * torch.randn((num_samples, 2, num_discr))
        up = up.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)

        ### 2. make trajectory valid
        up, costs_dict = self.optimize(up, self.ego.args.mp_epochs, initializing=True)
        self.initial_traj = [up, costs_dict]
        up_best, cost_min = self.find_best()
        up_best = up_best.detach()
        self.best_traj = [up_best, cost_min]
        return up_best, cost_min

    def find_best(self, criterium="cost_sum"):
        costs = self.initial_traj[1][criterium][-1]
        cost_min, idx = costs.min(dim=0)
        up_best = self.initial_traj[0][[idx], :, :]
        if self.plot:
            uref_traj, xref_traj = self.ego.system.up2ref_traj(self.ego.xref0, up_best, self.ego.args, short=True)
            self.ego.visualize_xref(xref_traj, name="best_cost%.4f" % cost_min, save=True, show=False,
                                    folder=self.path_log_opt)
        return up_best, cost_min

    def optimize_traj(self, up):
        if self.plot:
            self.path_log_opt = make_path(self.path_log, self.name + "_improvedTraj")
        up, costs_dict = self.optimize(up, self.ego.args.mp_epochs_density, initializing=False)
        self.improved_traj.append([up, costs_dict])
        cost_min = {}
        for key in costs_dict.keys():
            cost_min[key] = costs_dict[key][-1]
        cost_min = self.remove_cost_factor(cost_min)
        return up, cost_min

    def optimize(self, up, epochs, initializing=False):
        if self.plot:
            folder = self.path_log_opt
        else:
            folder = None
        costs_dict = []
        self.rms = torch.zeros_like(up)
        self.momentum = torch.zeros_like(up)
        self.counts = torch.zeros(up.shape[0], 1, 1)
        up.requires_grad = True
        if initializing:
            self.check_bounds = torch.zeros(up.shape[0], dtype=torch.bool)
            self.check_collision = torch.zeros(up.shape[0], dtype=torch.bool)

        for iter in range(epochs):
            if iter > 0:
                cost.sum().backward()
                with torch.no_grad():
                    u_update = self.optimizer_step(up.grad)
                    up -= torch.clamp(u_update, -self.ego.args.max_gradient, self.ego.args.max_gradient)
                    up.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)
                    up.grad.zero_()

            if initializing:
                uref_traj, xref_traj = self.get_traj_initialize(up, name="iter%d" % iter, plot=self.plot, folder=folder)
                #x_traj = xref_traj[:, :4, :]
                cost, cost_dict = self.get_cost_initialize(uref_traj, xref_traj)
            else:
                uref_traj, _, x_traj, rho_traj = self.get_traj(up, name="iter%d" % iter, folder=folder,
                                                               compute_density=True, plot=self.plot)
                cost, cost_dict = self.get_cost(uref_traj, x_traj, rho_traj)
            costs_dict.append(cost_dict)

        costs_dict = listDict2dictList(costs_dict)
        if self.plot_cost:
            path_log_cost = make_path(self.path_log, self.name + "_initialTrajCost")
            plot_cost(costs_dict, self.ego.args, folder=path_log_cost)
        return up, costs_dict

    def get_traj_initialize(self, up, name="traj", plot=True, folder=None):
        """
        compute reference trajectory from input parameters for initialization process

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
            1 x 4 x N_sim_short
        """
        uref_traj, xref_traj = self.ego.system.up2ref_traj(self.ego.xref0.repeat(up.shape[0], 1, 1),
                                                           up, self.ego.args, short=True)
        if plot:
            if folder is None:
                folder = self.path_log
            self.ego.visualize_xref(xref_traj, name=name, save=True, show=False, folder=folder)
        return uref_traj, xref_traj

    def get_cost_initialize(self, uref_traj, x_traj):
        """
        compute cost of a given trajectory

        Parameters
        ----------
        uref_traj: torch.Tensor
            1 x 2 x N_sim -1
        xref_traj: torch.Tensor
            1 x 5 x N_sim
        x_traj: torch.Tensor
            1 x 4 x N_sim
        rho_traj: torch.Tensor
            1 x 1 x N_sim

        Returns
        -------
        cost: torch.Tensor
            overall cost for given trajectory
        cost_dict: dictionary
            contains the weighted costs of all types
        """
        cost_uref = self.get_cost_uref(uref_traj)
        cost_goal, goal_reached = self.get_cost_goal_initialize(x_traj)
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
            x_check = x_traj[self.check_bounds, :, :]
            cost_bounds[self.check_bounds], in_bounds[self.check_bounds] = self.get_cost_bounds_initialize(x_check)
            if not torch.all(self.check_collision):
                idx_check = torch.logical_and(in_bounds, torch.logical_not(self.check_collision))
                if torch.any(idx_check):
                    self.check_collision[idx_check] = True  # TO-DO: get the right indizes
                    self.rms[idx_check, :, :] = 0
                    self.momentum[idx_check, :, :] = 0
                    self.counts[idx_check] = 0

        if torch.any(self.check_collision):
            x_check = x_traj[self.check_collision, :, :]
            cost_coll[self.check_collision] = self.get_cost_coll_initialize(x_check)  # for xref: 0.044s

        cost = self.weight_goal * cost_goal \
               + self.weight_uref * cost_uref \
               + self.weight_bounds * cost_bounds \
               + self.weight_coll * cost_coll
        cost_dict = {
            "cost_sum": cost,
            "cost_coll": self.weight_coll * cost_coll,
            "cost_goal": self.weight_goal * cost_goal,
            "cost_uref": self.weight_uref * cost_uref,
            "cost_bounds": self.weight_bounds * cost_bounds
        }
        return cost, cost_dict


    def get_cost_goal_initialize(self, x_traj, rho_traj=None):
        """
        compute cost for reaching the goal in inilialization process

        Parameters
        ----------
        x_traj: torch.Tensor
            1 x 4 x N_sim
        rho_traj: torch.Tensor
            1 x 1 x N_sim

        Returns
        -------
        cost: torch.Tensor
            cost for distance to the goal in the last iteration
        close: bool
            True if distance smaller than args.close2goal_thr
        """

        sq_dist = ((x_traj[:, :2, -1] - self.ego.xrefN[:, :2, 0]) ** 2).sum(dim=1)
        if rho_traj is None:  # is none for initialize method of grad motion planner
            cost_goal = sq_dist
        else:
            cost_goal = (rho_traj[:, 0, -1] * sq_dist).sum()
        close = cost_goal < self.ego.args.close2goal_thr
        cost_goal[torch.logical_not(close)] *= self.ego.args.weight_goal_far
        # if rho_traj is not None:
        #     close = torch.all(close)
        return cost_goal, close

    def get_cost_bounds_initialize(self, x_traj):
        """
        compute the cost for traying in the valid state space in inilialization process

        Parameters
        ----------
        x_traj: torch.Tensor
            1 x 4 x N_sim
        rho_traj: torch.Tensor
            1 x 1 x N_sim

        Returns
        -------
        cost: torch.Tensor
            cost for staying in the admissible state space
        in_bounds: bool
            True if inside of valid state space for all time steps
        """

        cost = torch.zeros(x_traj.shape[0])

        in_bounds = torch.ones(x_traj.shape[0], dtype=torch.bool)
        if torch.any(x_traj < self.ego.system.X_MIN_MP):
            idx = (x_traj < self.ego.system.X_MIN_MP).nonzero(as_tuple=True)
            sq_error = ((x_traj[idx] - self.ego.system.X_MIN_MP[0, idx[1], 0]) ** 2)
            if sq_error.dim() == 3:
                cost[idx[0]] += sq_error.sum(dim=(1, 2))
            else:
                cost[idx[0]] += sq_error
            in_bounds[idx[0]] = False
        if torch.any(x_traj > self.ego.system.X_MAX_MP):
            idx = (x_traj > self.ego.system.X_MAX_MP).nonzero(as_tuple=True)
            sq_error = ((x_traj[idx] - self.ego.system.X_MAX_MP[0, idx[1], 0]) ** 2)
            if sq_error.dim() == 3:
                cost[idx[0]] += sq_error.sum(dim=(1, 2))
            else:
                cost[idx[0]] += sq_error
            in_bounds[idx[0]] = False
        return cost, in_bounds

    def get_cost_coll_initialize(self, x_traj):
        """
        compute cost for high collision probabilities in inilialization process

        Parameters
        ----------
        x_traj: torch.Tensor
            1 x 4 x N_sim

        Returns
        -------
        cost: torch.Tensor
            cost for collisions
        """

        cost = torch.zeros(x_traj.shape[0])
        with torch.no_grad():
            gridpos_x, gridpos_y = pos2gridpos(self.ego.args, pos_x=x_traj[:, 0, :], pos_y=x_traj[:, 1, :])
            gridpos_x = torch.clamp(gridpos_x, 0, self.ego.args.grid_size[0] - 1)
            gridpos_y = torch.clamp(gridpos_y, 0, self.ego.args.grid_size[1] - 1)
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
                cost[idx] += (coll_prob * sq_dist)
        return cost

    def optimizer_step(self, grad):
        """
        compute step of optimizer

        Parameters
        ----------
        grad: torch.Tensor
            gradient of cost

        Returns
        -------
        step: torch.Tensor
            step for optimizing
        """

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

    def validate_ref(self, up):
        """
        evaluate input parameters (plot and compute final cost), assume that reference trajectory starts at ego.xref0

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory

        Returns
        -------
        cost_dict: dictionary
            contains the unweighted cost tensors
        """

        if self.plot_final:
            path_final = make_path(self.path_log, self.name + "_finalMotionPlan")
        else:
            path_final = None

        uref_traj, xref_traj, x_traj, rho_traj = self.get_traj(up, name=self.name+"_finalRef", compute_density=True,
                                                               plot=self.plot_final, use_nn=False, folder=path_final)

        if self.plot_final:
            self.ego.animate_traj(path_final, xref_traj, x_traj, rho_traj)
        self.xref_traj = xref_traj
        cost, cost_dict = self.get_cost(uref_traj, x_traj, rho_traj, evaluate=True)
        cost_dict = self.remove_cost_factor(cost_dict)
        logging.info("%s: True cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost_dict["cost_coll"],
                                                                                 cost_dict["cost_goal"],
                                                                                 cost_dict["cost_bounds"],
                                                                                 cost_dict["cost_uref"]))
        self.plot_final = False
        return cost_dict

    def validate_traj(self, up, xe0=None, return_time=False, biased=False):
        """
        evaluate input parameters (plot and compute final cost), assume that reference trajectory starts at ego.xref0

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory
        xe0: torch.Tensor
            batch_size x 4 x 1: tensor of initial deviation of reference trajectory

        Returns
        -------
        cost_dict: dictionary
            contains the unweighted cost tensors
        """

        logging.info("")
        logging.info("##### %s: Validate trajectory" % self.name)
        if xe0 is None:
            xe0 = torch.zeros(1, self.ego.system.DIM_X, 1)
        elif not biased:
            xe0[:, 4, :] = 0
        rho0 = torch.ones(1, 1, 1)

        if self.plot_final:
            path_final = make_path(self.path_log, self.name + "_finalTraj")
        else:
            path_final = None

        # TO-DO: get u_traj and compute "true" u_cost
        t0 = time.time()
        uref_traj, xref_traj, x_traj, _ = self.get_traj(up, name=self.name+"_validTraj", xe0=xe0, rho0=rho0,
                                                               compute_density=False, plot=self.plot_final, use_nn=False,
                                                               folder=path_final)
        rho_traj = torch.ones(1, 1, x_traj.shape[2])
        self.x_traj = x_traj

        if self.plot_final:
            self.ego.animate_traj(path_final, xref_traj, x_traj, rho_traj)

        cost, cost_dict = self.get_cost(uref_traj, x_traj, rho_traj, evaluate=True)
        cost_dict = self.remove_cost_factor(cost_dict)
        t_plan = time.time() - t0
        logging.debug("%s: Evaluation finished in %.2fs" % (self.name, t_plan))
        logging.info("%s: True cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost_dict["cost_coll"],
                                                                                 cost_dict["cost_goal"],
                                                                                 cost_dict["cost_bounds"],
                                                                                 cost_dict["cost_uref"]))
        if return_time:
            return cost_dict, t_plan
        return cost_dict



class MotionPlannerSearch(MotionPlanner):
    def __init__(self, ego, name="search", path_log=None):
        super().__init__(ego, name=name, path_log=path_log)
        self.incl_cost_goal = True
        self.incl_cost_uref = True
        self.up_saved = []
        self.cost_dict = []
        self.cost_path = []
        self.weight_goal = 0.002
        self.weight_uref = 0.1
        self.weight_bounds = 1
        self.weight_coll = 10
        self.coll_thr = ego.args.coll_thr
        self.goal_thr = ego.args.goal_thr
        if ego.args.input_type == "discr10":
            self.num_discr = 10
        elif ego.args.input_type == "discr5":
            self.num_discr = 5
        self.repeats = 1

    def plan_motion(self):
        """
        start motion planner: call optimization/planning function and compute cost
        """

        logging.info("")
        logging.info("##### %s: Starting motion planning %s" % (self.name, self.ego.args.mp_name))
        t0 = time.time()
        up, cost = self.plan_traj()
        t_plan = time.time() - t0
        logging.info("%s: Planning finished in %.2fs" % (self.name, t_plan))
        if up is not None:
            logging.info("%s: Final cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost["cost_coll"],
                                                                                      cost["cost_goal"],
                                                                                      cost["cost_bounds"],
                                                                                      cost["cost_uref"]))
            logging.debug(up)
            cost = self.validate_ref(up)
        else:
            self.xref_traj = None
            logging.info("%s: No valid solution found" % self.name)
        return up, cost, t_plan

    def validate_ref(self, up):
        """
        evaluate input parameters (plot and compute final cost), assume that reference trajectory starts at ego.xref0

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory

        Returns
        -------
        cost_dict: dictionary
            contains the unweighted cost tensors
        """

        if self.plot_final:
            path_final = make_path(self.path_log, self.name + "_finalMotionPlan")
        else:
            path_final = None

        uref_traj, xref_traj, x_traj, rho_traj = self.get_traj(up, name=self.name+"_finalRef", compute_density=True,
                                                               plot=self.plot_final, use_nn=False, folder=path_final)

        if self.plot_final:
            self.ego.animate_traj(path_final, xref_traj, x_traj, rho_traj)
        self.xref_traj = xref_traj
        cost, cost_dict = self.get_cost(uref_traj, x_traj, rho_traj, evaluate=True)
        cost_dict = self.remove_cost_factor(cost_dict)
        logging.info("%s: True cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost_dict["cost_coll"],
                                                                                 cost_dict["cost_goal"],
                                                                                 cost_dict["cost_bounds"],
                                                                                 cost_dict["cost_uref"]))
        return cost_dict

    def plan_traj(self):
        t0 = time.time()
        self.u0 = torch.arange(self.ego.system.UREF_MIN[0, 0, 0] + 1, self.ego.system.UREF_MAX[0, 0, 0] - 1 + 1e-5,
                               self.ego.args.du_search[0])
        self.u1 = torch.arange(self.ego.system.UREF_MIN[0, 1, 0] + 1, self.ego.system.UREF_MAX[0, 1, 0] - 1 + 1e-5,
                               self.ego.args.du_search[1])
        if self.plot:
            self.path_log_opt = make_path(self.path_log, self.name + "_foundTraj")
        else:
            self.path_log_opt = None
        success = False
        for u0 in self.u0:
            for u1 in self.u1:
                up = torch.tensor([u0, u1]).reshape((1, -1, 1)).repeat(1, 1, self.repeats)
                self.check_up(up)

        while not success and len(self.up_saved) > 0:
            # get best parameterset
            cost_path_min = min(self.cost_path)
            idx_min = self.cost_path.index(cost_path_min)

            # remove and extend best parameterset
            _ = self.cost_path.pop(idx_min)
            cost_dict_min = self.cost_dict.pop(idx_min)
            cost_goal_min = cost_dict_min["cost_goal"]
            up_min = self.up_saved.pop(idx_min)
            success = self.extend_traj(up_min, cost_goal_min)
            if time.time() - t0 > self.ego.args.opt_time_limit:
                break
        if success:
            cost_min = self.remove_cost_factor(cost_dict_min)
        else:
            up_min = None
            cost_min = None
        return up_min, cost_min

    def check_up(self, up, cost_goal_old=np.inf):
        with torch.no_grad():
            uref_traj, _, x_traj, rho_traj = self.get_traj(up, name="length%d" % up.shape[2], folder=self.path_log_opt, compute_density=True, plot=self.plot)
            cost, cost_dict = self.get_cost(uref_traj, x_traj, rho_traj)

        time_left = (self.num_discr - up.shape[2]) * 10 / self.num_discr
        distance_goal = (cost_dict["cost_goal"] / (self.ego.args.weight_goal_far * self.weight_goal)).sqrt()
        if up.shape[2] < self.num_discr:
            heading_to_goal = np.arctan2(self.ego.xrefN[0, 1, 0] - x_traj[0, 1, -1],
                                     self.ego.xrefN[0, 0, 0] - x_traj[0, 0, -1])
            heading_diff = ((x_traj[0, 2, -1] - heading_to_goal + np.pi) % (2 * np.pi) - np.pi).abs()
        else:
            if distance_goal > self.goal_thr: # too hard: cost_dict["cost_goal"] < self.ego.args.close2goal_thr:
                return False
            heading_diff = 0
            cost_goal_old = np.inf
            time_left = np.inf

        if cost_dict["cost_bounds"] == 0 and cost_dict["cost_goal"] < cost_goal_old and heading_diff < np.pi/2\
                and distance_goal / time_left < 9 and cost_dict["cost_coll"] / self.weight_coll < self.coll_thr:
            self.up_saved.append(up)
            self.cost_dict.append(cost_dict)
            cost_path = cost_dict["cost_coll"].item()
            if self.incl_cost_goal:
                cost_path += cost_dict["cost_goal"].item()
            if self.incl_cost_uref:
                cost_path += cost_dict["cost_uref"].item()

            if isinstance(self, MotionPlannerSampling):
                self.cost_path = np.concatenate((self.cost_path, np.array([cost_path])))
            else:
                self.cost_path.append(cost_path)
            return True
        return False

    def extend_traj(self, up_old, cost_goal_old):
        if up_old.shape[2] == self.num_discr:
            return True
        for u0 in self.u0:
            for u1 in self.u1:
                # extend up
                up = torch.cat((up_old, torch.tensor([u0, u1]).reshape((1, -1, 1)).repeat(1, 1, self.repeats)), dim=2)
                self.check_up(up, cost_goal_old)
        return False


class MotionPlannerSampling(MotionPlannerSearch):
    def __init__(self, ego, name="sampling", path_log=None):
        super().__init__(ego, name=name, path_log=path_log)
        self.cost_path = np.array([])
        self.incl_cost_goal = True
        self.incl_cost_uref = False
        self.start_samples = 10
        self.cost_chosen = 0.1
        self.weight_goal = 0.1
        self.weight_coll = 14

    def plan_traj(self):
        if self.plot:
            self.path_log_opt = make_path(self.path_log, self.name + "_foundTraj")
        else:
            self.path_log_opt = None
        success = False

        t0 = time.time()
        while not success:
            decision_new = torch.randint(0, len(self.up_saved) + self.start_samples, (1,))
            if decision_new >= len(self.up_saved): # up_add gets added as new parameter set
                up_add = 0.5 * torch.randn((1, 2, 1))
                up_add = up_add.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)
                self.check_up(up_add)
            else: # up_add extends an already saved parameter set
                thr = np.mean(self.cost_path)
                weights = thr - self.cost_path + 1e-8
                weights[self.cost_path > thr] = 0
                decision_cost = np.random.choice(self.cost_path, p=weights / weights.sum())
                idx_chosen = np.where(self.cost_path == decision_cost)[0][0]
                self.cost_path[idx_chosen] += self.cost_chosen

                cost_dict_chosen = self.cost_dict[idx_chosen]
                cost_goal_chosen = cost_dict_chosen["cost_goal"]
                up_chosen = self.up_saved[idx_chosen]
                success, up_ext = self.extend_traj(up_chosen, cost_goal_chosen)
            if time.time() - t0 > self.ego.args.opt_time_limit:
                break

        if success:
            uref_traj, xref_traj, x_traj, rho_traj = self.get_traj(up_ext, name=self.name+"_finalRef", compute_density=True,
                                                                       plot=False, use_nn=False, folder=None)
            cost, cost_dict = self.get_cost(uref_traj, x_traj, rho_traj)
            cost_min = self.remove_cost_factor(cost_dict)
            self.xref_traj = xref_traj
        else:
            self.xref_traj = None
            cost_min = None
            up_ext = None
        return up_ext, cost_min

    def extend_traj(self, up_old, cost_goal_old):
        up_add = 0.5 * torch.randn((1, 2, 1))
        up_add = up_add.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)
        up = torch.cat((up_old, up_add), dim=2)
        added = self.check_up(up, cost_goal_old)
        if up.shape[2] == self.num_discr and added:
            return True, up
        return False, up

