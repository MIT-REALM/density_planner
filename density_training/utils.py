import torch
from torch import nn
from data_generation.create_dataset import densityDataset
from data_generation.utils import get_output_variables, get_input_tensors, load_outputmap
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    """
    class for neural density predictor
    """
    def __init__(self, num_inputs, num_outputs, args):
        super(NeuralNetwork, self).__init__()

        if args.activation == "relu":
            self.activation = nn.ReLU()
        elif args.activation == "tanh":
            self.activation = nn.Tanh()
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


def load_dataloader(args):
    """
    load the data loader
    :param args:    settings
    :return: train and validation dataloader
    """
    train_data = densityDataset(args, mode="Train")
    # if args.equation == "FPE":
    #     args.batch_size = 1
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    #train_data.data = train_data.data[0:500]
    val_data = densityDataset(args, mode="Val")
    # val_data = train_data
    # val_data.data = train_data.data[0:100]
    validation_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    # train_set_size = int(len(density_data) * args.train_len)
    # val_set_size = len(density_data) - train_set_size
    # train_data, validation_data = random_split(density_data, [train_set_size, val_set_size])
    # train_dataloader = DataLoader(train_data.dataset, batch_size=args.batch_size, shuffle=True)
    # validation_dataloader = DataLoader(validation_data.dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, validation_dataloader


def create_configs(learning_rate=None, num_hidden=None, size_hidden=None, weight_decay=None, optimizer=None, rho_loss_weight=None, args=None):
    """
    create different hyperparameter configurations

    :param learning_rate:   list with learning rates which should be tested
    :param num_hidden:      list with number of layers which should be tested
    :param size_hidden:     list with number of units per layer which should be tested
    :param weight_decay:    list with weight decays which should be tested
    :param optimizer:       list with optimizers which should be tested
    :param rho_loss_weight: list with weights for density which should be tested
    :param args:            settings
    :return: configurations
    """
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
    """
    adapt settings according to the given config
    :param config:
    :param args:
    :return: new settings
    """
    args.learning_rate = config["learning_rate"]
    args.size_hidden = [config["size_hidden"]] * config["num_hidden"]
    args.weight_decay = config["weight_decay"]
    args.optimizer = config["optimizer"]
    args.rho_loss_weight = config["rho_loss_weight"]
    return args


def load_nn(num_inputs, num_outputs, args, load_pretrained=None, nn2=False):
    """
    create a new NN or load a pretrained NN

    :param num_inputs:      number of NN inputs
    :param num_outputs:     number of NN outputs
    :param args:            settings
    :param load_pretrained: True, if a pretrained NN should be loaded
    :param nn2:             True, if the pretrained NN should be loaded from args.name_pretrained_nn2
    :return: model and optimizer of the NN
    """
    if load_pretrained is None:
        load_pretrained = args.load_pretrained_nn

    model = NeuralNetwork(num_inputs, num_outputs, args).to(args.device)
    if load_pretrained:
        if nn2:
            model_params, _, _ = torch.load(args.name_pretrained_nn2, map_location=args.device)
        else:
            model_params, _, _ = torch.load(args.name_pretrained_nn, map_location=args.device)
        model.load_state_dict(model_params)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplemented('NotImplemented')

    return model, optimizer


def get_nn_prediction(model, xe0, xref0, t, u_params, args):
    """
    function to get the density predictions with the NN
    :param model:   NN model
    :param xe0:     initial deviation of reference trajectory
    :param xref0:   initial state of reference trajectory
    :param t:       prediction time points
    :param u_params:input parameters
    :param args:    settings
    :return:    xe:     predicted deviation of reference trajectory
                rholog: predicted logarithmic density
    """

    input_tensor, _ = get_input_tensors(u_params.flatten(), xref0, xe0, t, args)
    output_map, num_outputs = load_outputmap(dim_x=xe0.shape[1], args=args)

    input = input_tensor.to(args.device)
    output = model(input)
    xe, rholog = get_output_variables(output, output_map)
    return xe.unsqueeze(-1), rholog.unsqueeze(-1).unsqueeze(-1)