import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # simulation parameter
    parser.add_argument('--N_sim', type=int, default=1001)
    parser.add_argument('--input_type', type=str, default="discr5")  # discr10, polyn3, sin, cust*, sins5
    #parser.add_argument('--input_params_zero', type=list, default=[0])
    parser.add_argument('--dt_sim', type=int, default=0.01)
    parser.add_argument('--N_sim_max', type=int, default=1001)
    parser.add_argument('--factor_pred', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=0)

    # data generation
    parser.add_argument('--samplesX_rawdata', type=int, default=20)
    parser.add_argument('--samplesT_rawdata', type=int, default=20)
    parser.add_argument('--size_rawdata', type=int, default=500)
    parser.add_argument('--iterations_rawdata', type=int, default=6)


    # processing
    parser.add_argument('--gpus', type=str, default="3")
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_jobs', type=int, default=1)

    # data paths
    parser.add_argument('--path_rawdata', type=str, default="data/rawdata/2022-06-13_filesVal_dist")  # directory for the density data
    parser.add_argument('--path_dataset', type=str, default="data/dataset/")  # directory for the density data
    parser.add_argument('--path_nn', type=str, default="data/trained_nn/")  # directory for saving and loading the trained NN
    parser.add_argument('--path_traj0', type=str, default="data/initial_traj/")

    parser.add_argument('--nameend_rawdata', type=str,
                        default=".pickle")  # ending of the file used for creating the data set / dataloader
    parser.add_argument('--nameend_TrainDataset', type=str,
                        default="files18_Train_dist.pickle")
    parser.add_argument('--nameend_ValDataset', type=str,
                        default="files6_Val_dist.pickle")
    parser.add_argument('--nameend_nn', type=str,
                        default="CAR_dt10ms_Nsim100_Nu10_iter1000.pickle")
    parser.add_argument('--name_pretrained_nn', type=str,
                        default="data/trained_nn/2022-06-11-21-26-28_NN_pretr5l_lr0.0001_numHidd5_sizeHidd100_rhoLoss0.01.pt") #2022-05-26-16-19-09_NN_normal_lr0.0001_numHidd4_sizeHidd100_rhoLoss0.01.pt") #
    parser.add_argument('--load_pretrained_nn', type=bool, default=False)
    #parser.add_argument('--load_dataset', type=bool, default=True)

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

    # NN parameter
    parser.add_argument('--run_name', type=str, default="discr5")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--nn_type', type=str, default="MLP")
    parser.add_argument('--batch_size', type=int, default=256) # 256
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--number_units', type=int, default=100)
    parser.add_argument('--number_layers', type=int, default=5)
    parser.add_argument('--rho_loss_weight', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default="Adam") # Adam or LFBGS
    parser.add_argument('--learning_rate', type=float, default=0.0001)  # 2 e-5
    parser.add_argument('--lr_step', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)  #1e-6 L2 regularization
    #parser.add_argument('--patience', type=int, default=0)

    # motion planning options
    parser.add_argument('--mp_name', type=str, default="notEnlarged")
    parser.add_argument('--mp_init', type=str, default="random") # how to find the initial trajectory
    parser.add_argument('--mp_search_numtraj', type=int, default=100) # number of trajectories which have to be found
    parser.add_argument('--mp_anim_initial', type=bool, default=False)
    parser.add_argument('--mp_anim_final', type=bool, default=False)
    parser.add_argument('--mp_opt_normal', type=bool, default=True)
    parser.add_argument('--mp_opt_density', type=bool, default=True)
    parser.add_argument('--mp_enlarged', type=bool, default=False) # optimize reference trajectory with enlarged grid?
    parser.add_argument('--mp_optimizer', type=str, default="Adam")

    # motion planning parameter
    parser.add_argument('--environment_size', type=list, default=[-12, 12, -30, 10])
    parser.add_argument('--grid_wide', type=float, default=0.1)
    #parser.add_argument('--grid_size', type=list, default=[240, 600])
    parser.add_argument('--sampling', type=str, default='grid')
    parser.add_argument('--sample_size', type=int, default=100000) #if sampling == 'random'
    parser.add_argument('--coll_sum', type=float, default=0.01)
    parser.add_argument('--coll_max', type=float, default=0.001)

    # optimization
    parser.add_argument('--du_search', type=list, default=[0.5, 0.5])
    parser.add_argument('--mp_lr', type=float, default=0.5)
    parser.add_argument('--mp_lrfactor_density', type=float, default=0.5)
    parser.add_argument('--mp_lr_step', type=int, default=20)
    parser.add_argument('--mp_beta1', type=int, default=0.9)
    parser.add_argument('--mp_beta2', type=int, default=0.999)
    parser.add_argument('--cost_goal_weight', type=float, default=1e-2)
    parser.add_argument('--cost_coll_weight', type=float, default=1e-2)
    parser.add_argument('--cost_uref_weight', type=float, default=1e-4)
    parser.add_argument('--cost_bounds_weight', type=float, default=1e-7)
    parser.add_argument('--cost_obsDist_weight', type=float, default=1e-5)
    parser.add_argument('--max_obsDist', type=int, default=15)
    parser.add_argument('--max_gradient', type=float, default=0.02)
    parser.add_argument('--cost_threshold', type=float, default=70)
    parser.add_argument('--addit_enlargement', type=int, default=5)
    parser.add_argument('--epochs_mp', type=float, default=300)

    args = parser.parse_args()
    args.grid_size = [int((args.environment_size[1]-args.environment_size[0]) / args.grid_wide)+1,
                      int((args.environment_size[3]-args.environment_size[2]) / args.grid_wide)+1]
    args.size_hidden = [args.number_units] * args.number_layers
    return args
