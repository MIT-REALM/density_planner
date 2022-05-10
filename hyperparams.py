import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # simulation parameter
    parser.add_argument('--N_sim', type=int, default=101)
    parser.add_argument('--N_u', type=int, default=10)
    parser.add_argument('--input_type', type=str, default="polyn3")  # discr10, polyn3
    #parser.add_argument('--input_params_zero', type=list, default=[0])
    parser.add_argument('--dt_sim', type=int, default=0.01)
    parser.add_argument('--random_seed', type=int, default=4)

    # data paths
    parser.add_argument('--path_rawdata', type=str, default="data/rawdata/2022-04-27_filesTrain_polyn3/")  # directory for the density data
    parser.add_argument('--path_dataset', type=str, default="data/dataset/")  # directory for the density data
    parser.add_argument('--path_nn', type=str, default="data/trained_nn/")  # directory for saving and loading the trained NN

    parser.add_argument('--nameend_rawdata', type=str,
                        default=".pickle")  # ending of the file used for creating the data set / dataloader
    parser.add_argument('--nameend_TrainDataset', type=str,
                        default="Train_polyn3_xpred.pickle")
    parser.add_argument('--nameend_ValDataset', type=str,
                        default="Val_polyn3_xpred.pickle")
    parser.add_argument('--nameend_nn', type=str,
                        default="CAR_dt10ms_Nsim100_Nu10_iter1000.pickle")
    parser.add_argument('--name_pretrained_nn', type=str,
                        default="data/trained_nn/2022-04-26_polyn3_best/2022-04-28-09-01-30_NN__lr0.0005_numHidd4_sizeHidd100_rhoLoss0.1.pt")  # location and name for the pretrained NN which is supposed to be loaded
    parser.add_argument('--load_pretrained_nn', type=bool, default=False)
    #parser.add_argument('--load_dataset', type=bool, default=True)

    # plot
    parser.add_argument('--path_plot_loss', type=str, default="plots/losscurves/")
    parser.add_argument('--path_plot_densityheat', type=str, default="plots/density_heatmaps/")
    parser.add_argument('--path_plot_scatter', type=str, default="plots/scatter/")
    parser.add_argument('--plot_loss', type=bool, default=False)
    parser.add_argument('--plot_densityheat', type=bool, default=True)

    # NN parameter
    parser.add_argument('--run_name', type=str, default="xpred")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--nn_type', type=str, default="MLP")
    parser.add_argument('--batch_size', type=int, default=256) # 256
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--train_len', type=float, default=0.7)  # val_len = 1-train_len
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--size_hidden', type=list, nargs="+", default=[100, 100, 100, 100]) #4 x 64
    parser.add_argument('--rho_loss_weight', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default="Adam") # Adam or LFBGS
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 2 e-5
    parser.add_argument('--weight_decay', type=float, default=0)  #1e-6 L2 regularization
    parser.add_argument('--patience', type=int, default=0)

    return parser.parse_args()
