import numpy as np
import torch
from motion_planning.utils import pos2gridpos, make_path
import time
import logging
import casadi
from motion_planning.MotionPlanner import MotionPlanner


class MotionPlannerNLP(MotionPlanner):
    """
    class using NLP solver for motion planning
    """
    
    def __init__(self, ego, u0=None, xe0=None,  name="oracle", path_log=None, use_up=True, biased=False):
        super().__init__(ego, name=name, path_log=path_log)
        self.rho0 = torch.ones(1, 1, 1)
        self.u0 = u0
        self.xe0 = xe0
        self.biased = biased
        if not biased:
            self.xe0[:, 4, :] = 0
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

        logging.info("")
        logging.info("##### %s: Starting motion planning %s" % (self.name, self.ego.args.mp_name))
        t0 = time.time()
        u_min = None
        x_min = None
        cost_dict_min = None
        cost_min = np.inf
        costs = []
        for j in range(10):
            if j > 0:
                up = 0.5 * torch.randn((1, 2, 10))
                self.u0 = up.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)
            u_traj, x_traj = self.solve_nlp()
            if u_traj is not None:  # up is None if MPC didn't find solution
                logging.debug(u_traj)
                cost = self.validate_traj(u_traj, output=False)
                costs.append(cost["cost_coll"].item())
                if cost["cost_coll"].item() < cost_min:
                    cost_min = cost["cost_coll"].item()
                    u_min = u_traj
                    x_min = x_traj
                    cost_dict_min = cost
        t_plan = time.time() - t0
        logging.info("%s: Costs:" % (self.name))
        logging.info(costs)
        logging.info("%s: Planning finished in %.2fs" % (self.name, t_plan))
        if u_min is not None:
            cost_dict_min = self.validate_traj(u_min)
            if self.plot and x_min is not None:
                self.ego.visualize_xref(x_min, name=self.name + " trajectory", save=True, show=False, folder=self.path_log)
        self.x_traj = x_min
        return u_min, cost_dict_min, t_plan

    def solve_nlp(self):
        """
        solve the nonlinear programming problem with casadi for the whole planning horizon
        """
        quiet = True
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
        goal_factor = 1

        opti.minimize(
            goal_factor * self.weight_goal * ((x[0, self.N] - px_ref) ** 2 + (x[1, self.N] - py_ref) ** 2) +
            self.weight_coll * casadi.sumsqr(coll_prob) +
            self.weight_uref * casadi.sumsqr(u)
        )
        opti.subject_to(x[:4, 0] == xstart[:]) # CHANGED

        opti.subject_to(u[0, :] <= 3)  # accel
        opti.subject_to(u[0, :] >= -3)  # accel
        opti.subject_to(u[1, :] <= 3)  # omega
        opti.subject_to(u[1, :] >= -3)  # omega

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
            logging.debug("%s: decision variables are initialized with random parameters" % self.name)
            uref_traj, xref_traj = self.ego.system.up2ref_traj(self.ego.xref0, self.u0, self.ego.args, short=True)
            if self.use_up:
                u_start = self.u0[0, :, :] # CHANGED
            else:
                u_start = uref_traj[0, :, :] # CHANGED
            opti.set_initial(u[:, :N_u], u_start[:, :N_u].numpy()) # CHANGED
            opti.set_initial(x[:, :self.N+1], xref_traj[0, :4, :self.N+1].numpy()) # CHANGED
        else:
            logging.debug("%s: decision variables are initialized with zeros" % self.name)

        ix_min, iy_min = pos2gridpos(self.ego.args, pos_x=x_min, pos_y=y_min)
        ix_max, iy_max = pos2gridpos(self.ego.args, pos_x=x_max, pos_y=y_max)
        xgrid = np.linspace(x_min, x_max, self.ego.env.grid.shape[0]) # np.arange(x_min, x_max + 0.0001, self.ego.args.grid_wide)
        ygrid = np.linspace(y_min, y_max, self.ego.env.grid.shape[1]) # np.arange(y_min, y_max + 0.0001, self.ego.args.grid_wide)

        for k in range(self.N):  # timesteps
            grid_coll_prob = self.ego.env.grid[ix_min:ix_max + 1, iy_min:iy_max + 1, k + 1].numpy().ravel(order='F')
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
            logging.debug("%s: No solution found" % self.name)
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
        logging.debug("%s: Solution found. State trajectory error: %.2f" % (self.name, (x_traj - xref_traj[:, :4, :]).abs().sum()))

        return uref_traj, x_traj

    def get_traj(self, u_traj, xe0=None, name="traj", plot=True, folder=None):
        """
        compute trajectories from up

        :param up: torch.Tensor
            parameters specifying the reference input trajectory
        :param name: string
            name of parameter set for plotting
        :param plot: bool
            True if reference trajectory is plotted
        :param folder: string
            name of folder to save plot

        :return: uref_traj: torch.Tensor
            1 x 2 x N_sim_short -1
        :return: xref_traj: torch.Tensor
            1 x 5 x N_sim_short
        :return: x_traj: torch.Tensor
            1 x 4 x N_sim_short
        :return: rho_traj: torch.Tensor
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

    def validate_traj(self, u_traj, output=True, x_traj=None, rho_traj=None):
        """
        evaluate input parameters (plot and compute final cost), assume that reference trajectory starts at ego.xref0+self.xe0

        :param up: torch.Tensor
            parameters specifying the reference input trajectory
        :param xe0: torch.Tensor
            batch_size x 4 x 1: tensor of initial deviation of reference trajectory
        :param rho0: torch.Tensor
            batch_size x 1 x 1: tensor of initial densities
        :param compute_density: bool
            True if rho_traj is computed

        :return: cost_dict: dictionary
            contains the unweighted cost tensors
        """

        if output and self.plot_final:
            path_final = make_path(self.path_log, self.name + "_finalTraj")
        else:
            path_final = None

        if x_traj is None or rho_traj is None:
            u_traj, xref_traj, x_traj, rho_traj = self.get_traj(u_traj, name="finalTraj", plot=self.plot_final, folder=path_final)
        if x_traj.shape[0] > 1:  # for tube-based MPC
            get_max = True
        else:
            get_max = False
        cost, cost_dict = self.get_cost(u_traj, x_traj, rho_traj, evaluate=True, get_max=get_max)
        cost_dict = self.remove_cost_factor(cost_dict)
        if output:
            logging.info("%s: True cost coll %.4f, goal %.4f, bounds %.4f, uref %.4f" % (self.name, cost_dict["cost_coll"],
                                                                                 cost_dict["cost_goal"],
                                                                                 cost_dict["cost_bounds"],
                                                                                 cost_dict["cost_uref"]))
        if output and self.plot_final:
            self.ego.animate_traj(path_final, x_traj, x_traj, rho_traj)

        return cost_dict


class MotionPlannerMPC(MotionPlannerNLP):
    """
    class using NLP solver for motion planning
    """

    def __init__(self, ego, u0=None, xe0=None, name="MPC", path_log=None, N_MPC=10, biased=False, safe=False, tube=0):
        super().__init__(ego, u0=u0, xe0=xe0, name=name, path_log=path_log, use_up=False, biased=biased)
        self.N_MPC = N_MPC
        self.safe = safe
        self.tube = tube

    def plan_motion(self):
        """
        start motion planner: call optimization/planning function and compute cost
        """

        logging.info("")
        logging.info("##### %s: Starting motion planning %s" % (self.name, self.ego.args.mp_name))
        t0 = time.time()
        u_traj, x_traj, mean_time = self.solve_nlp()
        t_plan = time.time() - t0
        logging.info("%s: Planning finished in %.2fs" % (self.name, t_plan))
        if u_traj is not None:  # up is None if MPC didn't find solution
            logging.info("%s: Solution found" % (self.name))
            logging.debug(u_traj)
            rho_traj = torch.ones(x_traj.shape[0], 1, x_traj.shape[2]) / x_traj.shape[0]
            cost = self.validate_traj(u_traj, x_traj=x_traj, rho_traj=rho_traj)
        else:
            logging.info("%s: No valid solution found" % (self.name))
            cost = None
        if self.plot and x_traj is not None:
            self.ego.visualize_xref(x_traj, name=self.name + " trajectory", save=True, show=False, folder=self.path_log)
        self.x_traj = x_traj
        return u_traj, cost, mean_time

    def solve_nlp(self):
        """
        solve the nonlinear programming problem with casadi in a receding horizon fashion
        """
        quiet = True
        px_ref = self.ego.xrefN[0, 0, 0].item()
        py_ref = self.ego.xrefN[0, 1, 0].item()
        N_u = self.N_MPC
        goal_factor = 1  # 100

        u_traj = np.zeros((1, 2, self.N))
        x_traj = np.zeros((1, 5, self.N + 1))
        if self.xe0 is None:
            self.xe0 = torch.zeros(1, self.ego.system.DIM_X, 1)
        x0 = self.xe0 + self.ego.xref0
        x_traj[:, :, [0]] = x0.numpy()

        x_min = self.ego.args.environment_size[0]  # self.ego.system.X_MIN[0, 0, 0]
        x_max = self.ego.args.environment_size[1]  # self.ego.system.X_MAX[0, 0, 0]
        y_min = self.ego.args.environment_size[2]  # self.ego.system.X_MIN[0, 1, 0]
        y_max = self.ego.args.environment_size[3]  # self.ego.system.X_MAX[0, 1, 0]
        ix_min, iy_min = pos2gridpos(self.ego.args, pos_x=x_min, pos_y=y_min)
        ix_max, iy_max = pos2gridpos(self.ego.args, pos_x=x_max, pos_y=y_max)
        xgrid = np.linspace(x_min, x_max, self.ego.env.grid.shape[0]) # np.arange(x_min, x_max + 0.0001, self.ego.args.grid_wide)
        ygrid = np.linspace(y_min, y_max, self.ego.env.grid.shape[1]) # np.arange(y_min, y_max + 0.0001, self.ego.args.grid_wide)
        times = []

        ### optimizer settings
        p_opts = {"expand": False}
        if self.tube == 0:
            s_opts = {"max_iter": self.ego.args.iter_MPC, "max_cpu_time": 0.1}
        else:
            s_opts = {"max_iter": self.ego.args.iter_tubeMPC, "max_cpu_time": 0.1}
        if quiet:
            p_opts["print_time"] = 0
            s_opts["print_level"] = 0
            s_opts["sb"] = "yes"
        num_failures = 0

        # get vectors from nominal state to all states of the tube
        if self.tube != 0:
            vectors = set()
            for dx in np.arange(0, self.tube + 1e-7, self.ego.args.grid_wide):
                for dy in np.arange(0, self.tube + 1e-7, self.ego.args.grid_wide):
                    if np.sqrt(dx ** 2 + dy ** 2) <= self.tube:
                        vectors.add((dx, dy))
                        vectors.add((dx, -dy))
                        vectors.add((-dx, dy))
                        vectors.add((-dx, -dy))

        xstart_true = x0
        for k_start in range(self.N):
            xstart_measured = xstart_true[:, :4, :].detach().clone()
            xstart_measured[:, 2, :] += xstart_true[:, 4, :]

            ### prepare optimizer
            opti = casadi.Opti()
            x = opti.variable(4, self.N_MPC + 1)  # state (x, y, psi, v)
            u = opti.variable(2, self.N_MPC)  # control (accel, omega)
            coll_prob = opti.variable(self.N_MPC)
            xstart = opti.parameter(4)
            opti.solver("ipopt", p_opts, s_opts)

            ### add constraints
            opti.subject_to(x[:4, 0] == xstart[:])
            opti.subject_to(u[0, :] <= 3)  # accel
            opti.subject_to(u[0, :] >= -3)  # accel
            opti.subject_to(u[1, :] <= 3)  # omega
            opti.subject_to(u[1, :] >= -3)  # omega
            opti.subject_to(x[0, :] <= x_max)  # x
            opti.subject_to(x[0, :] >= x_min)  # x
            opti.subject_to(x[1, :] <= y_max)  # y
            opti.subject_to(x[1, :] >= y_min)  # y
            opti.subject_to(x[2, :] <= self.ego.system.X_MAX[0, 2, 0].item())  # theta
            opti.subject_to(x[2, :] >= self.ego.system.X_MIN[0, 2, 0].item())  # theta
            opti.subject_to(x[3, :] <= self.ego.system.X_MAX[0, 3, 0].item())  # v
            opti.subject_to(x[3, :] >= self.ego.system.X_MIN[0, 3, 0].item())  # v

            for k in range(min(self.N_MPC, self.N - k_start)):  # timesteps
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

                grid_coll_prob = self.ego.env.grid[ix_min:ix_max + 1, iy_min:iy_max + 1, k_start + k + 1].numpy().ravel(
                    order='F')
                LUT = casadi.interpolant('name', 'linear', [xgrid, ygrid], grid_coll_prob)
                if self.tube == 0:
                    opti.subject_to(coll_prob[k] == LUT(casadi.hcat([x[0, k + 1], x[1, k + 1]])))
                else:
                    coll_prob_sum = 0
                    for v in vectors:
                        coll_prob_sum += LUT(casadi.hcat([x[0, k + 1] + v[0], x[1, k + 1] + v[1]]))
                    opti.subject_to(coll_prob[k] == coll_prob_sum)
                if self.safe:
                    opti.subject_to(coll_prob[k] == 0)

            opti.minimize(
                goal_factor * k_start ** 2 / self.N ** 2 * self.weight_goal * (
                        (x[0, self.N_MPC] - px_ref) ** 2 + (x[1, self.N_MPC] - py_ref) ** 2) +
                self.weight_coll * casadi.sumsqr(coll_prob) +
                self.weight_uref * casadi.sumsqr(u)
            )

            ### online computations
            t0 = time.time()

            ### initialization
            opti.set_value(xstart[:], xstart_measured[0, :, 0].numpy())
            if k_start == 0:
                #opti.set_value(xstart[:], x0[0, :4, 0].numpy())
                if self.u0 is not None:
                    logging.info("%s: decision variables are initialized with good parameters" % self.name)
                    uref_traj, xref_traj = self.ego.system.up2ref_traj(self.ego.xref0, self.u0, self.ego.args,
                                                                       short=True)
                    opti.set_initial(u[:, :N_u], uref_traj[0, :, :N_u].numpy())
                    opti.set_initial(x[:, :], xref_traj[0, :4, :].numpy())
                else:
                    logging.info("%s: decision variables are initialized with zeros" % self.name)
            else:
                opti.set_initial(u[:, :N_u - 1], ud[:, 1:])
                opti.set_initial(x[:, :self.N_MPC], xd[:, 1:])
                #opti.set_value(xstart[:], xd[:, 1])

            ### solve
            try:
                sol1 = opti.solve()
                ud = sol1.value(u)
                xd = sol1.value(x)
                logging.debug("%s: Solution found." % (self.name))

            except:  # no solution found
                num_failures += 1
                times.append(time.time() - t0)
                logging.debug(
                    "%s: No solution found at iteration %d. Continueing with debug values." % (self.name, k_start))
                ud = opti.debug.value(u)
                xd = opti.debug.value(x)

            # solution found
            times.append(time.time() - t0)
            u_traj[0, :, k_start] = ud[:, 0]
            x_traj_true = self.ego.system.compute_xref_traj(xstart_true,
                                        torch.from_numpy(np.repeat(u_traj[:, :, [k_start]], 10, axis=2)), self.ego.args,
                                                          short=True)
            xstart_true = x_traj_true[:, :, [-1]]
            x_traj[:, :, [k_start + 1]] = xstart_true
            if ((xstart_true[0, 0, 0] - px_ref) ** 2 + (xstart_true[0, 1, 0] - py_ref) ** 2) < self.ego.args.goal_reached_MPC:
                u_traj = u_traj[:, :, :k_start + 1]
                x_traj = x_traj[:, :, :k_start + 2]
                break

        ### save and print results
        u_traj = torch.from_numpy(u_traj)
        x_traj = torch.from_numpy(x_traj)
        xref_traj = self.ego.system.compute_xref_traj(x0, u_traj.repeat_interleave(10, dim=2), self.ego.args, short=True)
        error = (x_traj - xref_traj).abs().sum()
        mean_time = np.array(times).mean()
        logging.info("%s: Average computation time: %.4f, maximum computation time: %.4f" % (self.name, mean_time, np.array(times).max()))
        logging.info("%s: State trajectory error: %.2f, number failures: %d" % (self.name, error, num_failures))

        if error > 10:
            logging.info("%s: Solution not valid (trajectory error too big)" % (self.name))
            u_traj = None
            x_traj = None

        return u_traj, x_traj, mean_time

