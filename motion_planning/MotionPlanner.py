import torch
from motion_planning.utils import pos2gridpos, initialize_logging, gridpos2pos
from abc import ABC, abstractmethod


class MotionPlanner(ABC):
    def __init__(self, ego, name="mp", path_log=None):
        """
        initialize motion planner object (function is called by all child classes)
        """
        self.name = name
        self.ego = ego
        self.xe0 = None
        self.rho0 = None
        self.weight_goal = ego.args.weight_goal
        self.weight_uref = ego.args.weight_uref
        self.weight_bounds = ego.args.weight_bounds
        self.weight_coll = ego.args.weight_coll
        self.plot = ego.args.mp_plot
        if ego.args.mp_plot_final is None:
            self.plot_final = ego.args.mp_plot
        else:
            self.plot_final = ego.args.mp_plot_final
        if path_log is None:
            self.path_log = initialize_logging(self.ego.args, self.ego.args.mp_name + "_" + self.name)
        else:
            self.path_log = path_log


    @abstractmethod
    def plan_motion(self):
        """
        method for planning the motion
        """
        pass

    def get_traj(self, up, xe0=None, rho0=None, name="traj", compute_density=False, use_nn=True, plot=True, folder=None):
        """
        compute trajectories from up

        Parameters
        ----------
        up: torch.Tensor
            parameters specifying the reference input trajectory
        name: string
            name of parameter set for plotting
        compute_density: bool
            True if rho_traj is computed
        use_nn: bool
            True if nn is used to predict density and state trajectory
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
        uref_traj, xref_traj = self.ego.system.up2ref_traj(self.ego.xref0, up, self.ego.args, short=True)
        if plot:
            if folder is None:
                folder = self.path_log
            self.ego.visualize_xref(xref_traj, name=name, save=True, show=False, folder=folder)

        x_traj, rho_traj = self.ego.predict_density(up, xref_traj, xe0=xe0, rho0=rho0, use_nn=use_nn, compute_density=compute_density)
        #x_traj = x_all[:, :4, :]

        return uref_traj, xref_traj, x_traj, rho_traj

    def get_cost(self, uref_traj, x_traj, rho_traj, evaluate=False, get_max=False):
        """
        compute cost of a given trajectory

        Parameters
        ----------
        uref_traj: torch.Tensor
            1 x 2 x N_sim -1
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
        cost_goal, goal_reached = self.get_cost_goal(x_traj, rho_traj, evaluate=evaluate, get_max=get_max)
        cost_bounds, in_bounds = self.get_cost_bounds(x_traj, rho_traj, evaluate=evaluate, get_max=get_max)
        cost_coll = self.get_cost_coll(x_traj, rho_traj, get_max=get_max)  # for xref: 0.044s

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

    def get_cost_uref(self, uref_traj):
        """
        compute cost for reference input trajectory

        Parameters
        ----------
        uref_traj: torch.Tensor
            1 x 2 x N_sim -1

        Returns
        -------
        cost: torch.Tensor
            control effort cost for given trajectory
        """
        cost = self.ego.args.weight_uref_effort * (uref_traj ** 2).sum(dim=(1, 2))
        return cost

    def get_cost_goal(self, x_traj, rho_traj, evaluate=False, get_max=False):
        """
        compute cost for reaching the goal

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
        elif get_max:
            cost_goal = (rho_traj[:, 0, -1] * sq_dist).max()
        else:
            cost_goal = (rho_traj[:, 0, -1] * sq_dist).sum()
        close = cost_goal < self.ego.args.close2goal_thr
        if not evaluate:
            cost_goal[torch.logical_not(close)] *= self.ego.args.weight_goal_far
        # if rho_traj is not None:
        #     close = torch.all(close)
        return cost_goal, close

    def get_cost_bounds(self, x_traj, rho_traj, evaluate=False, get_max=False):
        """
        compute the cost for traying in the valid state space

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

        cost = torch.zeros(1)
        if evaluate:
            x_min = self.ego.system.X_MIN
            x_max = self.ego.system.X_MAX
        else:
            x_min = self.ego.system.X_MIN_MP
            x_max = self.ego.system.X_MAX_MP

        in_bounds = torch.ones(x_traj.shape[0], dtype=torch.bool)
        if torch.any(x_traj[:, :4, :] < x_min[:, :4, :]):
            idx = (x_traj < x_min).nonzero(as_tuple=True)
            sq_error = ((x_traj[idx] - x_min[0, idx[1], 0]) ** 2)
            if get_max:
                cost += (rho_traj[idx[0], 0, idx[2]] * sq_error).max()
            else:
                cost += (rho_traj[idx[0], 0, idx[2]] * sq_error).sum()
            in_bounds[idx[0]] = False
        if torch.any(x_traj[:, :4, :] > x_max[:, :4, :]):
            idx = (x_traj > x_max).nonzero(as_tuple=True)
            sq_error = ((x_traj[idx] - x_max[0, idx[1], 0]) ** 2)
            if get_max:
                cost += (rho_traj[idx[0], 0, idx[2]] * sq_error).max()
            else:
                cost += (rho_traj[idx[0], 0, idx[2]] * sq_error).sum()
            in_bounds[idx[0]] = False
        in_bounds = torch.all(in_bounds)
        return cost, in_bounds

    def get_cost_coll(self, x_traj, rho_traj, get_max=False):
        """
        compute cost for high collision probabilities

        Parameters
        ----------
        x_traj: torch.Tensor
            1 x 4 x N_sim
        rho_traj: torch.Tensor
            1 x 1 x N_sim

        Returns
        -------
        cost: torch.Tensor
            cost for collisions
        """
        cost = torch.zeros(1)

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
                if get_max:
                    cost += (rho_traj[idx, 0, i] * coll_prob * sq_dist).max()
                else:
                    cost += (rho_traj[idx, 0, i] * coll_prob * sq_dist).sum()
        return cost

    def remove_cost_factor(self, cost_dict):
        """
        remove the weighting factors from the entries of the cost dictionary

        Parameters
        ----------
        cost_dict: dictionary
            contains the weighted cost tensors

        Returns
        -------
        cost_dict: dictionary
            contains the unweighted cost tensors
        """

        cost_dict["cost_coll"] = cost_dict["cost_coll"] / self.weight_coll
        cost_dict["cost_goal"] = cost_dict["cost_goal"] / self.weight_goal
        cost_dict["cost_bounds"] = cost_dict["cost_bounds"] / self.weight_bounds
        cost_dict["cost_uref"] = cost_dict["cost_uref"] / self.weight_uref
        cost_dict["cost_sum"] = 0
        for key in cost_dict.keys():
            cost_dict["cost_sum"] += cost_dict[key]
        return cost_dict

    # @abstractmethod
    # def validate_traj(self, up, xref0=None, xe0=None, rho0=None, compute_density=True):
    #     """
    #     method for evaluating the optimized input (plot resulting trajectory and compute its cost)
    #     """
    #     pass
