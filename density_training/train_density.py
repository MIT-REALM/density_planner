import torch
import os
import hyperparams
from datetime import datetime
from plots.plot_functions import plot_losscurves
from systems.utils import listDict2dictList
from density_training.utils import NeuralNetwork, load_nn, load_args, load_dataloader, loss_function, evaluate, create_configs


if __name__ == "__main__":
    args = hyperparams.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name

    # configs = create_configs(args=args)
    configs = create_configs(args=args)
    results = []

    for i, config in enumerate(configs):

        print(f"_____Configuration {i}:______")
        name = run_name + f", learning_rate={config['learning_rate']}, num_hidden={config['num_hidden']}, size_hidden={config['size_hidden']}, weight_decay={config['weight_decay']}, optimizer={config['optimizer']}, rho_loss_weight={config['rho_loss_weight']}"
        short_name = run_name + f"_lr{config['learning_rate']}_numHidd{config['num_hidden']}_sizeHidd{config['size_hidden']}_rhoLoss{config['rho_loss_weight']}"
        nn_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_NN_" + short_name + '.pt'
        lossPlot_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_lossCurve_" + short_name + ".jpg"
        maxErrorPlot_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_maxErrorCurve_" + short_name + ".jpg"
        print(name)

        train_dataloader, validation_dataloader = load_dataloader(args)
        args = load_args(config, args)
        model, optimizer = load_nn(len(train_dataloader.dataset.data[0][0]), len(train_dataloader.dataset.data[0][1]),
                                   args)

        patience = 0
        test_loss_best = float('Inf')
        test_loss = []
        train_loss = []
        for epoch in range(args.epochs):
            train_loss.append(evaluate(train_dataloader, model, args, optimizer=optimizer, mode="train"))
            test_loss.append(evaluate(validation_dataloader, model, args, mode="val"))
            print(f"Epoch {epoch},    Train loss: {(train_loss[-1]['loss']):.3f}  (x: {(train_loss[-1]['loss_xe']):.5f}),    Test loss: {(test_loss[-1]['loss']):.5f}  (x: {(test_loss[-1]['loss_xe']):.5f})")

            # if test_loss[-1]["loss"] > test_loss_best:
            #     patience += 1
            # else:
            #     patience = 0
            #     test_loss_best = test_loss[-1]["loss"]
            # if args.patience and patience >= args.patience:
            #     break
            if epoch > 2000 and epoch % 1000 == 19:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / args.lr_step

            if epoch % 20 == 2:
                train_loss_dict = listDict2dictList(train_loss)
                test_loss_dict = listDict2dictList(test_loss)
                result = {
                    'train_loss': train_loss_dict,
                    'test_loss': test_loss_dict,
                    'num_epochs': epoch,
                    'name': nn_name,
                    'cost_function': "rho_nn=rho0*exp(out)",
                    'hyperparameters': config
                }
                #plot_losscurves(result, "t%d_" % (epoch) + short_name, args, filename=maxErrorPlot_name, type="maxError")
                plot_losscurves(result, "t%d_" % (epoch) + short_name, args, filename=lossPlot_name)

            if epoch % 100 == 99:
                torch.save([model.state_dict(), args, result], nn_name)
        #print(f"Best test loss after epoch {epoch}: {test_loss_best} \n")
        torch.save([model.state_dict(), args, result], nn_name)
        plot_losscurves(result, "final_t%d_" % (epoch) + short_name, args, filename=lossPlot_name)
    print("Done!")

