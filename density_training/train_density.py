import torch
import hyperparams
from torch import nn
from datetime import datetime
from plots.plot_functions import plot_losscurves
from systems.utils import listDict2dictList
from density_training.utils import *

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(NeuralNetwork, self).__init__()

        if args.activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplemented('NotImplemented')
        self.type = args.nn_type

        if args.nn_type == 'MLP':
            self.linear_list = nn.ModuleList()
            self.linear_list.append(nn.Linear(num_inputs, args.size_hidden[0]))
            for i in range(len(args.size_hidden) - 1):
                self.linear_list.append(nn.Linear(args.size_hidden[i], args.size_hidden[i + 1]))
            self.linear_list.append(nn.Linear(args.size_hidden[-1], num_outputs))
        elif args.nn_type == 'LSTM':
            self.hidden_size = args.size_hidden[0]
            self.num_layers = len(args.size_hidden)
            self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True)
            self.linear = nn.Linear(args.size_hidden[0], num_outputs)

    def forward(self, x):
        if self.type == 'MLP':
            for i in range(len(self.linear_list) - 1):
                x = self.activation(self.linear_list[i](x))
            x = self.linear_list[len(self.linear_list) - 1](x)
        elif self.type == 'LSTM':
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        return x


if __name__ == "__main__":
    args = hyperparams.parse_args()
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
            #
            # if test_loss[-1]["loss"] > test_loss_best:
            #     patience += 1
            # else:
            #     patience = 0
            #     test_loss_best = test_loss[-1]["loss"]
            # if args.patience and patience >= args.patience:
            #     break
            # if epoch == 18:
            #     for g in optimizer.param_groups:
            #         g['lr'] = g['lr'] / 2


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
                plot_losscurves(result, "t%d_" % (epoch) + short_name, args, filename=maxErrorPlot_name, type="maxError")
                plot_losscurves(result, "t%d_" % (epoch) + short_name, args, filename=lossPlot_name)

            if epoch % 100 == 99:
                torch.save([model.state_dict(), args, result], nn_name)
        print(f"Best test loss after epoch {epoch}: {test_loss_best} \n")
        torch.save([model.state_dict(), args, result], nn_name)
        plot_losscurves(result, "final_t%d_" % (epoch) + short_name, args, filename=lossPlot_name)
    print("Done!")

