import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    ### SIMULATION
    # simulation parameter
    parser.add_argument('--N_sim', type=int, default=1001)
    parser.add_argument('--input_type', type=str, default="discr10")  # discr10, polyn3, sin, cust*, sins5
    #parser.add_argument('--input_params_zero', type=list, default=[0])
    parser.add_argument('--dt_sim', type=int, default=0.01)
    parser.add_argument('--N_sim_max', type=int, default=1001)
    parser.add_argument('--factor_pred', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=0)

    ### DATA GENERATION
    # data generation
    parser.add_argument('--samplesX_rawdata', type=int, default=20)
    parser.add_argument('--samplesT_rawdata', type=int, default=20)
    parser.add_argument('--size_rawdata', type=int, default=100)
    parser.add_argument('--iterations_rawdata', type=int, default=500)
    parser.add_argument('--bin_number', type=int, default=torch.Tensor([20, 20, 10, 10]).long())
    #parser.add_argument('--bin_width', type=int, default=0.1) #TO-DO: remove???

    # processing
    parser.add_argument('--gpus', type=str, default="3")
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_jobs', type=int, default=1)

    # data paths
    parser.add_argument('--path_rawdata', type=str, default="data/rawdata/2022-07-12_discr10_train/")  # directory for the density data
    parser.add_argument('--path_dataset', type=str, default="data/dataset/")  # directory for the density data
    parser.add_argument('--path_nn', type=str, default="data/trained_nn/")  # directory for saving and loading the trained NN
    parser.add_argument('--path_traj0', type=str, default="data/initial_traj/")
    parser.add_argument('--path_matlab', type=str, default="data/matlab_env/")

    parser.add_argument('--nameend_rawdata', type=str,
                        default=".pickle")  # ending of the file used for creating the data set / dataloader
    parser.add_argument('--nameend_TrainDataset', type=str,
                        default="files810_Train.pickle")
    parser.add_argument('--nameend_ValDataset', type=str,
                        default="files104_Val.pickle")
    parser.add_argument('--nameend_nn', type=str,
                        default="CAR_dt10ms_Nsim100_Nu10_iter1000.pickle")
    parser.add_argument('--name_pretrained_nn', type=str,
                        default="data/trained_nn/2022-07-26-06-54-08_NN_rlw0.0005_lr0.001_numHidd7_sizeHidd150_rhoLoss0.0005.pt")
    parser.add_argument('--name_pretrained_nn2', type=str,
                        default="data/trained_nn/2022-07-26-06-52-02_NN_rlw0.0001_lr0.001_numHidd7_sizeHidd150_rhoLoss0.0001.pt")

    ### PLOTTING
    # plot
    parser.add_argument('--path_plot_loss', type=str, default="plots/losscurves/")
    parser.add_argument('--path_plot_cost', type=str, default="plots/costcurves/")
    parser.add_argument('--path_plot_densityheat', type=str, default="plots/density_heatmaps/")
    parser.add_argument('--path_plot_scatter', type=str, default="plots/scatter/")
    parser.add_argument('--path_plot_references', type=str, default="plots/references/")
    parser.add_argument('--path_plot_grid', type=str, default="plots/grid/")
    parser.add_argument('--path_plot_motion', type=str, default="plots/motion/")
    parser.add_argument('--plot_loss', type=bool, default=False)
    parser.add_argument('--plot_densityheat', type=bool, default=True)

    ### DENSITY NEURAL NETWORK
    # NN parameter
    parser.add_argument('--run_name', type=str, default="all")
    parser.add_argument('--load_pretrained_nn', type=bool, default=False)
    parser.add_argument('--equation', type=str, default="LE") #LE, FPE_MC, FPE_fourier, FPE_FE
    parser.add_argument('--batch_size', type=int, default=512) # 256
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--nn_type', type=str, default="MLP")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--number_units', type=int, default=150)
    parser.add_argument('--number_layers', type=int, default=7)
    parser.add_argument('--optimizer', type=str, default="Adam") # Adam or LFBGS
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 2 e-5
    parser.add_argument('--lr_step', type=int, default=1)
    parser.add_argument('--lr_step_epoch', type=int, default=160)
    parser.add_argument('--weight_decay', type=float, default=0)  #1e-6 L2 regularization

    # FPE NN
    parser.add_argument('--fpe_iterations', type=int, default=10000) # 256

    # LE NN
    parser.add_argument('--rho_loss_weight', type=float, default=0.001)


    ### MOTION PLANNING
    # general motion planning options
    parser.add_argument('--mp_name', type=str, default="test")
    parser.add_argument('--weight_goal_far', type=float, default=10) #1 if no influence
    parser.add_argument('--weight_uref_effort', type=float, default=1e-2) #0 if no influence
    parser.add_argument('--close2goal_thr', type=float, default=3)
    # ego vehicle
    parser.add_argument('--mp_gaussians', type=int, default=5)
    parser.add_argument('--mp_sampling', type=str, default='random')
    parser.add_argument('--mp_sample_size', type=int, default=500) #5000 #if sampling == 'random'
    # environment
    parser.add_argument('--environment_size', type=list, default=[-12, 12, -30, 10])
    parser.add_argument('--grid_wide', type=float, default=0.1)

    # optimization with gradient descent
    parser.add_argument('--mp_optimizer', type=str, default="Adam")
    parser.add_argument('--mp_epochs', type=int, default=100) #100
    parser.add_argument('--mp_epochs_density', type=int, default=100) #100
    parser.add_argument('--mp_numtraj', type=float, default=30) #30
    parser.add_argument('--mp_lr', type=float, default=1e-2)
    parser.add_argument('--mp_lr_step', type=int, default=0)
    parser.add_argument('--max_gradient', type=float, default=0.02)
    # cost parameters
    parser.add_argument('--weight_goal', type=float, default=1e-2)
    parser.add_argument('--weight_coll', type=float, default=1e-1)
    parser.add_argument('--weight_uref', type=float, default=1e-4)
    parser.add_argument('--weight_bounds', type=float, default=1e1)

    # optimization with search
    parser.add_argument('--du_search', type=list, default=[1, 1])
    # cost parameters
    parser.add_argument('--search_weight_goal', type=float, default=1e-2)
    parser.add_argument('--search_weight_coll', type=float, default=1e-1)
    parser.add_argument('--search_weight_uref', type=float, default=1e-4)
    parser.add_argument('--search_weight_bounds', type=float, default=1e1)
    parser.add_argument('--goal_thr', type=float, default=4)
    parser.add_argument('--coll_thr', type=float, default=4)
    parser.add_argument('--opt_time_limit', type=float, default=3600)

    # optimization with NLP
    parser.add_argument('--iter_NLP', type=int, default=5000) #1000
    parser.add_argument('--iter_MPC', type=int, default=2000) #1000
    parser.add_argument('--goal_reached_MPC', type=float, default=1)




    args = parser.parse_args()
    args.grid_size = [int((args.environment_size[1]-args.environment_size[0]) / args.grid_wide)+1,
                      int((args.environment_size[3]-args.environment_size[2]) / args.grid_wide)+1]
    args.size_hidden = [args.number_units] * args.number_layers
    return args
