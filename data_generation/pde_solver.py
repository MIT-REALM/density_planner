import torch

from plots.plot_functions import plot_density_heatmap_fpe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf

class Solver_PDE:

    def __init__(self, system, args):
        # preparations
        dim_x = 4

        Lx = (system.XE_MAX - system.XE_MIN).squeeze()
        nx = args.bin_number.long()
        dx = Lx / nx
        points = []

        rho0 = torch.zeros(nx[0], nx[1], nx[2], nx[3])

        start = torch.floor(nx / 2 * (1 - system.XE0_MIN.squeeze() / system.XE_MIN.squeeze())).long()
        end = torch.floor(nx / 2 * (1 - system.XE0_MAX.squeeze() / system.XE_MAX.squeeze())).long()
        rho0[start[0]:-end[0], start[1]:-end[1], start[2]:-end[2], start[3]:-end[3]] = 1  # / (num_nonz * vol)
        rho0 = rho0 / (rho0.sum() * Lx.prod())

        if args.equation == "FPE_fourier":
            for i in range(dim_x):
                points.append(torch.arange(-Lx[i] / 2 + dx[i] / 2, Lx[i] / 2, dx[i]))
            x0, x1, x2, x3 = torch.meshgrid(points[0], points[1], points[2], points[3])
            xpoints = [x0, x1, x2, x3]
            xe = torch.zeros(nx.prod(), dim_x, 1)
            for i in range(dim_x):
                xe[:, i, 0] = xpoints[i].flatten()

            # wavenumbers for derivative
            kkx = []
            for i in range(dim_x):
                kkx.append(torch.cat((torch.arange(0, nx[i] / 2 + 1), torch.arange(- nx[i] / 2 + 1, 0))))
            kx0, kx1, kx2, kx3 = torch.meshgrid(kkx[0], kkx[1], kkx[2], kkx[3])

            # dealiasing
            dealias01 = torch.logical_and(torch.abs(kx0) < 2 / 3 * nx[0] / 2, torch.abs(kx1) < 2 / 3 * nx[1] / 2)
            dealias23 = torch.logical_and(torch.abs(kx2) < 2 / 3 * nx[2] / 2, torch.abs(kx3) < 2 / 3 * nx[3] / 2)
            dealias = torch.logical_and(dealias01, dealias23)
            rho0_hat = torch.fft.fftn(rho0) * dealias

            rk = torch.zeros(3, 3)
            rk[0, 0] = 1 / 3
            rk[1, 0] = -1
            rk[1, 1] = 2
            rk[2, 0] = 0
            rk[2, 1] = 3 / 4
            rk[2, 2] = 1 / 4
        else:
            nxm = nx - 1
            nxg = nx + 1
            nxp = nxm - 1
            #Ind = torch.eye(nxm[0], nxm[1], nxm[2], nxm[3])
            alpha = 1.5

            points_g = []
            for i in range(dim_x):
                points.append(torch.arange(-Lx[i] / 2 + dx[i] / 2, Lx[i] / 2, dx[i]))
                points_g.append(torch.arange(-Lx[i] / 2, Lx[i] / 2 + dx[i] / 2, dx[i]))
            x0, x1, x2, x3 = torch.meshgrid(points[0], points[1], points[2], points[3])
            xpoints = [x0, x1, x2, x3]
            xe = torch.zeros(nx.prod(), dim_x, 1)
            for i in range(dim_x):
                xe[:, i, 0] = xpoints[i].flatten()

            # interpolation weights(faces to centers)
            # wu_1 = torch.zeros(nx, nyg - 1)
            # wu_2 = torch.zeros(nx, nyg - 1)
            # for i in range(nx):
            #     wu_2[i,:] = (y - yg[0:-1] ) / (yg[1:] - yg[0: - 1])
            #     wu_1[i,:] = 1 - wu_2[i,:]

        G = torch.matmul(system.DIST[0,:, :], system.DIST[0,:, :].t())

        self.kx = [kx0, kx1, kx2, kx3]
        self.Lx = Lx
        self.nx = nx
        self.xpoints = xpoints
        self.xe = xe
        self.dealias = dealias
        self.rho0 = rho0
        self.rho0_hat = rho0_hat
        self.rk = rk
        self.dt = args.dt_sim
        #self.nt = args.N_sim
        self.system = system
        self.G = G


    def solve_fpe(self, uref_traj, xref_traj, time_steps, args, folder=None):
        # starting solver
        t = 0
        rho_hat = self.rho0_hat
        rho_save = []

        for istep in range(uref_traj.shape[2]):
            f = self.system.f_func(self.xe+xref_traj[:, :, [istep]], xref_traj[:, :, [istep]], uref_traj[:, :, [istep]],
                                noise=False) - self.system.fref_func(xref_traj[:, :, [istep]], uref_traj[:, :, [istep]])

            f_shaped = []
            for i in range(4):
                f_shaped.append(f[:, i, 0].reshape(self.nx[0], self.nx[1], self.nx[2], self.nx[3]))

            # numerical solution
            rho_hat_old = rho_hat

            F1rho = self.rhs_fpe(rho_hat, f_shaped)
            rho_hat = rho_hat_old + self.dt * self.rk[0, 0] * F1rho

            F2rho = self.rhs_fpe(rho_hat, f_shaped)
            rho_hat = rho_hat_old + self.dt * (self.rk[1, 0] * F1rho + self.rk[1, 1] * F2rho)

            F3rho = self.rhs_fpe(rho_hat, f_shaped)
            rho_hat = rho_hat_old + self.dt * (self.rk[2, 0] * F1rho + self.rk[2, 1] * F2rho + self.rk[2, 2] * F3rho)

            t = t + self.dt

            if istep % 1 == 0:
                rho = torch.clamp(torch.real(torch.fft.ifftn(rho_hat)), min=0)
                mask = rho < 1e-7
                rho[mask] = 0
                rho = rho / (rho.sum() * self.Lx.prod())
                rho_hat = torch.fft.fftn(rho) * self.dealias

            if istep in time_steps: #% 10 == 0:
                rho = torch.real(torch.fft.ifftn(rho_hat))
                # plot_density_heatmap("FPE_Fourier_timestep%d" % istep, args, {"FPE": self.xe},
                #                     {"FPE": torch.clamp(rho, 0, 100).flatten().unsqueeze(-1).unsqueeze(-1)}, folder=folder)
                plot_density_heatmap_fpe("timestep%d" % istep, torch.clamp(rho, min=0).sum(dim=(2, 3)).T, args,
                                         save=True, show=True, filename=None, include_date=True, folder=folder)
                #if istep in time_steps:
                rho_save.append(torch.clamp(rho, 0, 100).flatten().unsqueeze(-1).unsqueeze(-1))
        return rho_save

    def rhs_fpe(self, rho_hat, f):
        # to physical space
        rho_hat = rho_hat * self.dealias
        rho = torch.real(torch.fft.ifftn(rho_hat))

        # compute f_i*rho TO-DO: check dimensions!!!
        dfrho_hat_sum = 0
        for i in range(4):
            frho_i_hat = torch.fft.fftn(f[i] * rho)
            dfrho_hat_sum += 1j * self.kx[i] * frho_i_hat

        # simplification for the given noise matrix (otherwise use loop)
        ddrho_sum = - self.G[0, 0] * self.kx[0] ** 2 * rho_hat - self.G[1, 1] * self.kx[1] ** 2 * rho_hat
        #ddrho_sum = 0
        #for i in range(4):
            #for ii in range(4):
            #    ddrho_sum -= self.G[i, ii] * self.kx[i] * self.kx[ii] * rho_hat

        # compute RHS
        rhs_hat = -dfrho_hat_sum + ddrho_sum

        # avoid contamination by high frequencies
        rhs_hat = rhs_hat * self.dealias
        return rhs_hat

    def solve_fpe_fplanc(self, uref_traj, xref_traj, time_steps, args, folder=None):
        nm = 1e-9
        viscosity = 8e-4
        radius = 50 * nm
        drag = 6 * np.pi * viscosity * radius

        L = 20 * nm
        #def F(x):
        F = lambda x: 5e-21 * (np.sin(x / L) + 4) / L

        sim = fokker_planck(temperature=300, drag=drag, extent=600 * nm,
                            resolution=10 * nm, boundary=boundary.periodic, force=F)

        ### steady-state solution
        steady = sim.steady_state()

        ### time-evolved solution
        pdf = gaussian_pdf(-150 * nm, 30 * nm)
        p0 = pdf(sim.grid[0])
        Nsteps = 200
        time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

        ### animation
        fig, ax = plt.subplots()

        ax.plot(sim.grid[0] / nm, steady, color='k', ls='--', alpha=.5)
        ax.plot(sim.grid[0] / nm, p0, color='red', ls='--', alpha=.3)
        line, = ax.plot(sim.grid[0] / nm, p0, lw=2, color='C3')

        def update(i):
            line.set_ydata(Pt[i])
            return [line]

        anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
        ax.set(xlabel='x (nm)', ylabel='normalized PDF')
        ax.margins(x=0)

        plt.show()
        return None

    # def solve_fpe_fe(self, uref_traj, xref_traj, time_steps, args, folder=None):
    #     # starting solver
    #     t = 0
    #     rho_hat = self.rho0_hat
    #     rho_save = []
    #
    #     for istep in range(uref_traj.shape[2]):
    #         f = self.system.f_func(self.xe+xref_traj[:, :, [istep]], xref_traj[:, :, [istep]], uref_traj[:, :, [istep]],
    #                             noise=False) - self.system.fref_func(xref_traj[:, :, [istep]], uref_traj[:, :, [istep]])
    #
    #         f_shaped = []
    #         for i in range(4):
    #             f_shaped.append(f[:, i, 0].reshape(self.nx[0], self.nx[1], self.nx[2], self.nx[3]))
    #
    #         # numerical solution
    #         rho_hat_old = rho_hat
    #
    #         F1rho = self.rhs_fpe(rho_hat, f_shaped)
    #         rho_hat = rho_hat_old + self.dt * self.rk[0, 0] * F1rho
    #
    #         F2rho = self.rhs_fpe(rho_hat, f_shaped)
    #         rho_hat = rho_hat_old + self.dt * (self.rk[1, 0] * F1rho + self.rk[1, 1] * F2rho)
    #
    #         F3rho = self.rhs_fpe(rho_hat, f_shaped)
    #         rho_hat = rho_hat_old + self.dt * (self.rk[2, 0] * F1rho + self.rk[2, 1] * F2rho + self.rk[2, 2] * F3rho)
    #
    #         t = t + self.dt
    #
    #         if istep % 10 == 0:
    #             rho = torch.clamp(torch.real(torch.fft.ifftn(rho_hat)), min=0)
    #             rho = rho / (rho.sum() * self.Lx.prod())
    #             rho_hat = torch.fft.fftn(rho) * self.dealias
    #
    #         if istep in time_steps: #% 10 == 0:
    #             rho = torch.real(torch.fft.ifftn(rho_hat))
    #             # plot_density_heatmap("FPE_Fourier_timestep%d" % istep, args, {"FPE": self.xe},
    #             #                     {"FPE": torch.clamp(rho, 0, 100).flatten().unsqueeze(-1).unsqueeze(-1)}, folder=folder)
    #             plot_density_heatmap_fpe("timestep%d" % istep, torch.clamp(rho, min=0).sum(dim=(2, 3)).T, args,
    #                                      save=True, show=True, filename=None, include_date=True, folder=folder)
    #             #if istep in time_steps:
    #             rho_save.append(torch.clamp(rho, 0, 100).flatten().unsqueeze(-1).unsqueeze(-1))
    #     return rho_save