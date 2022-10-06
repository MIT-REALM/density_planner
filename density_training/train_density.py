import torch
import os
import hyperparams
from datetime import datetime
from plots.plot_functions import plot_losscurves
from systems.utils import listDict2dictList
from density_training.utils import load_nn, load_args, load_dataloader, get_output_variables, create_configs


def evaluate_le(dataloader, model, args, optimizer=None, mode="val"):
    """
    function to evaluate the loss function and optimize the NN

    :param dataloader:  NN dataloader
    :param model:       NN model
    :param args:        settings
    :param optimizer:   NN optimizer
    :param mode:        mode (validation or training)
    :return: dictionary with loss
    """

    if mode == "train" and optimizer is not None:
        model.train()
    elif mode == "val":
        model.eval()
    else:
        raise NotImplemented('Mode not defined')

    total_loss, total_loss_xe, total_loss_rho_w = 0, 0, 0
    max_loss_rho_w = torch.zeros(len(dataloader))

    for batch, (input, target) in enumerate(dataloader):
        input, target = input.to(args.device), target.to(args.device)

        # Compute prediction error
        output = model(input)
        xe_nn, rholog_nn = get_output_variables(output, dataloader.dataset.output_map)
        xe_true, rholog_true = get_output_variables(target, dataloader.dataset.output_map)
        if batch == 0:
            max_loss_xe = torch.zeros(len(dataloader), xe_nn.shape[1])

        loss_xe, loss_rho_w = loss_function_le(xe_nn, xe_true, rholog_nn, rholog_true, args)
        loss = loss_rho_w + loss_xe
        total_loss_xe += loss_xe.item()
        total_loss_rho_w += loss_rho_w.item()
        max_loss_xe[batch, :], _ = torch.max(torch.abs(xe_nn - xe_true), dim=0)
        max_loss_rho_w[batch] = args.rho_loss_weight * torch.max(torch.abs(rholog_nn - torch.log(rholog_true)))
        total_loss += loss.item()

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    maxMax_loss_xe, _ = torch.max(max_loss_xe, dim=0)
    loss_all = {
        "loss": total_loss / len(dataloader),
        "loss_xe": total_loss_xe / len(dataloader),
        "loss_rho_w": total_loss_rho_w / len(dataloader),
        "max_error_xe": maxMax_loss_xe.detach().numpy(),
        "max_error_rho_w": (torch.max(max_loss_rho_w)).detach().numpy()
    }
    return loss_all


def loss_function_le(xe_nn, xe_true, rholog_nn, rholog_true, args):
    """
    loss function

    :param xe_nn:       NN predictions for deviation of the reference trajectory
    :param xe_true:     true deviation of the reference trajectory (by integrating the dynamics)
    :param rholog_nn:   NN prediction for logarithmic density
    :param rholog_true: true logarithmic density (by LE)
    :param args:        settings
    :return: loss
    """
    loss_xe = ((xe_nn - xe_true) ** 2).mean()
    mask = torch.logical_or(rholog_true.abs() > 1e30, torch.isnan(rholog_true))
    if mask.any():
        print("invalid rho set to 1e30")
        rholog_true[mask] = 1e30
    loss_rho = ((rholog_nn - rholog_true) ** 2).mean()

    return loss_xe, args.rho_loss_weight * loss_rho


### script to train the neural density predictor
if __name__ == "__main__":
    args = hyperparams.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name

    configs = create_configs(args=args)  # create different hyperparameter configurations
    results = []

    for i, config in enumerate(configs):

        print(f"_____Configuration {i}:______")
        # define the names
        if args.equation == "LE":
            name = run_name + f", learning_rate={config['learning_rate']}, num_hidden={config['num_hidden']}, size_hidden={config['size_hidden']}, weight_decay={config['weight_decay']}, optimizer={config['optimizer']}, rho_loss_weight={config['rho_loss_weight']}"
            short_name = run_name + f"_lr{config['learning_rate']}_numHidd{config['num_hidden']}_sizeHidd{config['size_hidden']}_rhoLoss{config['rho_loss_weight']}"
        else:
            name = run_name + f", fpe, learning_rate={config['learning_rate']}, lrstep{args.lr_step}, lrstepe{args.lr_step_epoch}, num_hidden={config['num_hidden']}, size_hidden={config['size_hidden']}, weight_decay={config['weight_decay']}"
            short_name = run_name + f"_fpe_lr{config['learning_rate']}_lrstep{args.lr_step}_lrstepe{args.lr_step_epoch}_numHidd{config['num_hidden']}_sizeHidd{config['size_hidden']}"

        nn_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_NN_" + short_name + '.pt'
        lossPlot_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_lossCurve_" + short_name + ".jpg"
        maxErrorPlot_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_maxErrorCurve_" + short_name + ".jpg"
        print(name)

        # loading
        train_dataloader, validation_dataloader = load_dataloader(args)  # load the data
        args = load_args(config, args)
        model, optimizer = load_nn(train_dataloader.dataset.num_inputs, train_dataloader.dataset.num_outputs,
                                   args)  # load new or pretrained NN

        test_loss_best = float('Inf')
        test_loss = []
        train_loss = []

        # train NN
        for epoch in range(args.epochs):
            loss_train = evaluate_le(train_dataloader, model, args, optimizer=optimizer, mode="train")
            loss_test = evaluate_le(validation_dataloader, model, args, mode="val")
            print(
                f"Epoch {epoch},    Train loss: {(loss_train['loss']):.3f}  (x: {(loss_train['loss_xe']):.5f}),    Test loss: {(loss_test['loss']):.5f}  (x: {(loss_test['loss_xe']):.5f})")

            train_loss.append(loss_train)
            test_loss.append(loss_test)

            # plots losscurve
            if epoch % 20 == 2:
                train_loss_dict = listDict2dictList(train_loss)
                test_loss_dict = listDict2dictList(test_loss)
                result = {
                    'train_loss': train_loss_dict,
                    'test_loss': test_loss_dict,
                    'num_epochs': epoch,
                    'name': nn_name,
                    'hyperparameters': config
                }
                plot_losscurves(result, "t%d_" % (epoch) + short_name, args, filename=lossPlot_name)

            # save model
            if epoch % 50 == 49:
                torch.save([model.state_dict(), args, result], nn_name)

        torch.save([model.state_dict(), args, result], nn_name)
        plot_losscurves(result, "final_t%d_" % (epoch) + short_name, args, filename=lossPlot_name)
    print("Done!")
