import matplotlib.pyplot as plt
import torch
import hyperparams
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from density_estimation.create_dataset import densityDataset
from datetime import datetime
from plot_functions import plot_losscurves
from utils import listDict2dictList
import os
import pickle


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(NeuralNetwork, self).__init__()

        if args.activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplemented('NotImplemented')

        self.linear_list = nn.ModuleList()
        self.linear_list.append(nn.Linear(num_inputs, args.size_hidden[0]))
        for i in range(len(args.size_hidden) - 1):
            self.linear_list.append(nn.Linear(args.size_hidden[i], args.size_hidden[i + 1]))
        self.linear_list.append(nn.Linear(args.size_hidden[-1], num_outputs))

    def forward(self, x):
        for i in range(len(self.linear_list) - 1):
            x = self.activation(self.linear_list[i](x))
        x = self.linear_list[len(self.linear_list) - 1](x)
        return x


def loss_function(x_nn, x_true, rho_nn, rho_true, args):
    loss_x = ((x_nn - x_true) ** 2).mean()
    mask = torch.logical_or(rho_true < 1e-5, rho_nn < 1e-5)
    loss_rho = 0
    if mask.any():
        loss_rho = ((rho_nn[mask] - rho_true[mask]) ** 2).mean()
        rho_nn = rho_nn[torch.logical_not(mask)]
        rho_true = rho_true[torch.logical_not(mask)]
    if not mask.all():
        loss_rho += ((torch.log(rho_nn) - torch.log(rho_true)) ** 2).mean()

    return loss_x, args.rho_loss_weight * loss_rho


def load_dataloader(args):
    train_data = densityDataset(args, mode="Train")
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data = densityDataset(args, mode="Val")
    validation_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    # train_set_size = int(len(density_data) * args.train_len)
    # val_set_size = len(density_data) - train_set_size
    # train_data, validation_data = random_split(density_data, [train_set_size, val_set_size])
    # train_dataloader = DataLoader(train_data.dataset, batch_size=args.batch_size, shuffle=True)
    # validation_dataloader = DataLoader(validation_data.dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, validation_dataloader


def create_configs(learning_rate=None, num_hidden=None, size_hidden=None, weight_decay=None, optimizer=None, rho_loss_weight=None, args=None):
    if learning_rate is None:
        learning_rate = [args.learning_rate]
    if num_hidden is None:
        num_hidden = [len(args.size_hidden)]
    if size_hidden is None:
        size_hidden = [args.size_hidden[0]]
    if weight_decay is None:
        weight_decay = [args.weight_decay]
    if optimizer is None:
        optimizer = [args.optimizer]
    if rho_loss_weight is None:
        rho_loss_weight = [args.rho_loss_weight]

    configs = []
    for lr in learning_rate:
        for nh in num_hidden:
            for sh in size_hidden:
                for wd in weight_decay:
                    for opt in optimizer:
                        for rw in rho_loss_weight:
                            configs.append({"learning_rate": lr,
                                            "num_hidden": nh,
                                            "size_hidden": sh,
                                            "weight_decay": wd,
                                            "optimizer": opt,
                                            "rho_loss_weight": rw})
    return configs


def load_args(config, args):
    args.learning_rate = config["learning_rate"]
    args.size_hidden = [config["size_hidden"]] * config["num_hidden"]
    args.weight_decay = config["weight_decay"]
    args.optimizer = config["optimizer"]
    args.rho_loss_weight = config["rho_loss_weight"]
    return args


def load_nn(num_inputs, num_outputs, args):
    model = NeuralNetwork(num_inputs-1, num_outputs, args).to(args.device)
    if args.load_pretrained_nn:
        model_params, _, _ = torch.load(args.name_pretrained_nn, map_location=args.device)
        model.load_state_dict(model_params)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplemented('NotImplemented')

    return model, optimizer


def evaluate(dataloader, model, args, optimizer=None, mode="val"):
    size = len(dataloader.dataset)
    if mode == "train" and optimizer is not None:
        model.train()
    elif mode == "val":
        model.eval()
    else:
        print("mode not defined")

    total_loss, total_loss_x, total_loss_rho_w = 0, 0, 0
    max_loss_x = torch.zeros(len(dataloader), 4)
    max_loss_rho_w = torch.zeros(len(dataloader))

    for batch, (input, target) in enumerate(dataloader):
        input, target = input.to(args.device), target.to(args.device)

        # Compute prediction error
        input_nn = torch.cat((input[:, :dataloader.dataset.input_map['rho0']], input[:, dataloader.dataset.input_map['rho0']+1:]), 1)
        output = model(input_nn)
        x_nn = output[:, dataloader.dataset.output_map['x']]
        rho_nn = input[:, dataloader.dataset.input_map['rho0']] * torch.exp(output[:, dataloader.dataset.output_map['rho']])
        x_true = target[:, dataloader.dataset.output_map['x']]
        rho_true = target[:, dataloader.dataset.output_map['rho']]
        loss_x, loss_rho_w = loss_function(x_nn, x_true, rho_nn, rho_true, args)
        loss = loss_x + loss_rho_w
        total_loss_x += loss_x.item()
        total_loss_rho_w += loss_rho_w.item()
        max_loss_x[batch, :], _ = torch.max(torch.abs(x_nn-x_true), dim=0)
        max_loss_rho_w[batch] = args.rho_loss_weight * torch.log(torch.max(torch.abs(rho_nn-rho_true)))
        total_loss += loss.item()

        if mode == "train":
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    maxMax_loss_x, _ = torch.max(max_loss_x, dim=0)
    loss_all = {
        "loss": total_loss / len(dataloader),
        "loss_x": total_loss_x / len(dataloader),
        "loss_rho_w": total_loss_rho_w / len(dataloader),
        "max_loss_x": maxMax_loss_x.detach().numpy(),
        "max_loss_rho_w": (torch.max(max_loss_rho_w)).detach().numpy()
        }
    return loss_all




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
            #      break
            if epoch == 100:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 10


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

