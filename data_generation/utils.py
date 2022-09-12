import torch


def load_inputmap(dim_x, args):
    """
    function to create dictionary "inputmap" which contains the mappings from the input tensor to the input variables
    :param dim_x:   dimension of the state
    :param args:    settings
    :return:    input_map:  dictionary
                num_inputs: number of inputs
    """

    if args.input_type == "discr10":
        dim_u = 20
    elif args.input_type == "discr5":
        dim_u = 10
    elif args.input_type == "polyn3":
        dim_u = 8
    elif args.input_type == "sincos3":
        dim_u = 12
    elif args.input_type == "sincos4":
        dim_u = 16
    elif args.input_type == "sin":
        dim_u = 8
    elif args.input_type == "cust2":
        dim_u = 12
    elif args.input_type == "cust3":
        dim_u = 18
    else:
        raise NotImplementedError
    num_inputs = 2 * dim_x + 1 + dim_u
    input_map = {'xe0': torch.arange(0, dim_x),
                 'xref0': torch.arange(dim_x, 2 * dim_x),
                 't': 2 * dim_x,
                 'u_params': torch.arange(2 * dim_x + 1, num_inputs)}
    return input_map, num_inputs


def load_outputmap(dim_x=5, args=None):
    """
    function to create dictionary "output_map" with mapping from output tensor to variables
    :param dim_x:   dimesnions of the state
    :param args:    settings
    :return:    output_map:  dictionary
                num_outputs: number of outputs
    """

    if args is None or args.equation == "LE":
        num_outputs = dim_x + 1
        output_map = {'xe': torch.arange(0, dim_x),
                      'rholog': dim_x}
    else:
        num_outputs = 1
        output_map = {'rho': 0}
    return output_map, num_outputs


def raw2nnData(results_all, args):
    """
    function to transform rawdata to the input and output tensor which are used for the NN training

    :param results_all: rawdata from "compute_rawdata.py"
    :param args:        settings
    :return:
    """

    data = []
    input_map = None

    for results in results_all:
        u_params = results['u_params']
        xref0 = results['xref0']
        t = results['t']
        if input_map is None:
            input_map, num_inputs = load_inputmap(xref0.flatten().shape[0], args)
            input_tensor = torch.zeros(num_inputs)
            output_map, num_outputs = load_outputmap(xref0.flatten().shape[0], args)
            output_tensor = torch.zeros(num_outputs)

        if args.equation == "LE":
            xe0 = results['xe0']
            xe_traj = results['xe_traj']
            rholog_traj = results['rholog_traj']
            input_tensor[input_map['u_params']] = u_params.flatten()
            input_tensor[input_map['xref0']] = xref0
            for i_x in range(min(xe_traj.shape[0], 10)):
                input_tensor[input_map['xe0']] = xe0[i_x, :]
                for i_t in range(0, t.shape[0]):
                    input_tensor[input_map['t']] = t[i_t]
                    output_tensor[output_map['xe']] = xe_traj[i_x, :, i_t]
                    output_tensor[output_map['rholog']] = rholog_traj[i_x, 0, i_t]
                    data.append([input_tensor.numpy().copy(), output_tensor.numpy().copy()])
        else:
            density_map = results['density']
            xref_traj = results['xref_traj']
            uref_traj = results['uref_traj']
            data.append([u_params.flatten(), xref0, t, density_map, xref_traj, uref_traj])
    return data, input_map, output_map, num_inputs, num_outputs


def get_input_tensors(u_params, xref0, xe0, t, args):
    """
    create input tesnor for NN

    :param u_params:    input parameters
    :param xref0:       start of the reference trajectory
    :param xe0:         initial deviation of the reference trajectory
    :param t:           time point
    :param args:        settings
    :return:    input_tensor:   tensor for NN
                input_map:      mapping from tensor to input values
    """

    bs = xe0.shape[0]
    if xe0.dim() > 1:
        input_map, num_inputs = load_inputmap(xe0.shape[1], args)
        input_tensor = torch.zeros(bs, num_inputs)
        if xref0.shape[0] == 1:
            xref0 = xref0.flatten()
        else:
            xref0 = xref0.squeeze()
        xe0 = xe0.squeeze()
    else:
        input_map, num_inputs = load_inputmap(xe0.shape[0], args)
        input_tensor = torch.zeros(1, num_inputs)
    if u_params.shape[0] == bs:
        input_tensor[:, input_map['u_params']] = u_params
    else:
        input_tensor[:, input_map['u_params']] = u_params.flatten()
    input_tensor[:, input_map['xref0']] = xref0
    input_tensor[:, input_map['xe0']] = xe0
    input_tensor[:, input_map['t']] = t
    return input_tensor, input_map


def get_output_tensors(xe, rholog):
    """
    create output tesnor for NN

    :param xe:          deviation of the reference trajectory
    :param rholog:      logarithmic density value
    :return:    output_tensor:   tensor for NN
                output_map:      mapping from tensor to output values
    """

    output_map, num_outputs = load_outputmap(xe.shape[0])
    output_tensor = torch.zeros(num_outputs)
    output_tensor[output_map['xe']] = xe
    output_tensor[output_map['rholog']] = rholog
    return output_tensor, output_map


def get_input_variables(input, input_map):
    """
    transform input tensor to input values
    :param input:       input tensor
    :param input_map:   mapping from input tensor to values
    :return: input values
    """
    if input.dim() == 2:
        xref0 = input[:, input_map['xref0']]
        xe0 = input[:, input_map['xe0']]
        t = input[:, input_map['t']]
        u_params = input[:, input_map['u_params']]
    elif input.dim() == 1:
        xref0 = input[input_map['xref0']]
        xe0 = input[input_map['xe0']]
        t = input[input_map['t']]
        u_params = input[input_map['u_params']]
    return xref0, xe0, t, u_params


def get_output_variables(output, output_map, type='normal'):
    """
    transform output tensor to output values
    :param output:      output tensor
    :param output_map:  mapping from tensor to output values
    :param type:        type of how to process density
    :return: output values
    """
    if output.dim() == 2:
        xe = output[:, output_map['xe']]
        if type == 'exp':
            rho = torch.exp(output[:, output_map['rholog']])
        elif type == 'normal':
            rho = (output[:, output_map['rholog']])
    elif output.dim() == 1:
        xe = output[output_map['xe']]
        if type == 'exp':
            rho = torch.exp(output[output_map['rholog']])
        elif type == 'normal':
            rho = (output[output_map['rholog']])
    return xe, rho


def nn2rawData(data, input_map, output_map, args):
    """
    transform NN tensors to rawdata dictionary
    :param data:
    :param input_map:
    :param output_map:
    :param args:
    :return: results:   dictionary with rawdata
    """
    input, output = data
    xref0, xe0, t, u_params = get_input_variables(input, input_map)
    xe, rho = get_output_variables(output, output_map)
    results = {
        'xe0': xe0,
        'xref0': xref0,
        't': t,
        'u_params': u_params,
        'xe': xe,
        'rho': rho,
    }
    return results
