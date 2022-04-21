import matplotlib.pyplot as plt
import torch
import hyperparams
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from density_estimation.create_dataset import densityDataset
from datetime import datetime
from plot_functions import plot_losscurves
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

    return loss_x + args.rho_loss_weight * loss_rho


def load_dataloader(args):
    density_data = densityDataset(args)
    train_set_size = int(len(density_data) * args.train_len)
    val_set_size = len(density_data) - train_set_size
    train_data, validation_data = random_split(density_data, [train_set_size, val_set_size])
    train_dataloader = DataLoader(train_data.dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data.dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, validation_dataloader


def create_configs(learning_rate=None, num_hidden=None, size_hidden=None, weight_decay=None, optimizer=None, args=None):
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

    configs = []
    for lr in learning_rate:
        for nh in num_hidden:
            for sh in size_hidden:
                for wd in weight_decay:
                    for opt in optimizer:
                        configs.append({"learning_rate": lr,
                                        "num_hidden": nh,
                                        "size_hidden": sh,
                                        "weight_decay": wd,
                                        "optimizer": opt})
    return configs


def load_args(config, args):
    args.learning_rate = config["learning_rate"]
    args.size_hidden = [config["size_hidden"]] * config["num_hidden"]
    args.weight_decay = config["weight_decay"]
    args.optimizer = config["optimizer"]
    return args


def load_NN(num_inputs, num_outputs, args):
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


def train(dataloader, model, optimizer, args):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (input, target) in enumerate(dataloader):
        input, target = input.to(args.device), target.to(args.device)

        # Compute prediction error
        input_nn = torch.cat((input[:, :dataloader.dataset.input_map['rho0']], input[:, dataloader.dataset.input_map['rho0']+1:]), 1)
        output = model(input_nn)
        x_nn = output[:, dataloader.dataset.output_map['x']]
        rho_nn = input[:, dataloader.dataset.input_map['rho0']] * torch.exp(output[:, dataloader.dataset.output_map['rho']])
        x_true = target[:, dataloader.dataset.output_map['x']]
        rho_true = target[:, dataloader.dataset.output_map['rho']]
        loss = loss_function(x_nn, x_true, rho_nn, rho_true, args)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)


def test(dataloader, model, args):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (input, target) in enumerate(dataloader):
            input, target = input.to(args.device), target.to(args.device)

            # Compute prediction error
            input_nn = torch.cat((input[:, :dataloader.dataset.input_map['rho0']], input[:, dataloader.dataset.input_map['rho0']+1:]), 1)
            output = model(input_nn)
            x_nn = output[:, dataloader.dataset.output_map['x']]
            rho_nn = input[:, dataloader.dataset.input_map['rho0']] * torch.exp(
                output[:, dataloader.dataset.output_map['rho']])
            x_true = target[:, dataloader.dataset.output_map['x']]
            rho_true = target[:, dataloader.dataset.output_map['rho']]
            test_loss += loss_function(x_nn, x_true, rho_nn, rho_true, args).item()

    test_loss /= num_batches
    return test_loss


if __name__ == "__main__":

    args = hyperparams.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = 'newCost'

    # configs = create_configs(args=args)
    configs = create_configs(learning_rate=[0.001, 0.0001], num_hidden=[3, 4], size_hidden=[32, 64],
                             weight_decay=[0, 1e-5], args=args)
    results = []

    for i, config in enumerate(configs):

        print(f"_____Configuration {i}:______")
        name = run_name + f", learning_rate={config['learning_rate']}, num_hidden={config['num_hidden']}, size_hidden={config['size_hidden']}, weight_decay={config['weight_decay']}, optimizer={config['optimizer']}"
        short_name = run_name + f"_lr{config['learning_rate']}_numHidd{config['num_hidden']}_sizeHidd{config['size_hidden']}_reg{config['weight_decay']}_opt{config['optimizer']}"
        print(name)

        train_dataloader, validation_dataloader = load_dataloader(args)
        args = load_args(config, args)
        model, optimizer = load_NN(len(train_dataloader.dataset.data[0][0]), len(train_dataloader.dataset.data[0][1]),
                                   args)

        patience = 0
        test_loss_best = float('Inf')
        test_loss = float('Inf') * torch.ones(args.epochs)
        train_loss = float('Inf') * torch.ones(args.epochs)
        for t in range(args.epochs):
            train_loss[t] = train(train_dataloader, model, optimizer, args)
            test_loss[t] = test(validation_dataloader, model, args)

            if test_loss[t] > test_loss_best:
                patience += 1
            else:
                patience = 0
                test_loss_best = test_loss[t]

            if args.patience and patience >= args.patience:
                break

        print(f"Best test loss after epoch {t}: {test_loss_best} \n")
        nn_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_NN_" + short_name + '.pt'

        result = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_loss_best': test_loss_best,
            'num_epochs': t,
            'name': nn_name,
            'cost_function': "rho_nn=rho0*exp(out)",
            'hyperparameters': config
        }
        results.append(result)
        torch.save([model.state_dict(), args, result], nn_name)
        plot_losscurves(train_loss, test_loss, short_name, args)

    results_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_results_" + run_name + '.pt'
    torch.save(results, results_name)
    print("Done!")

