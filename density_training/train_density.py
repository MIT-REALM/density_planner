import torch
from systems.sytem_CAR import Car
import os
import hyperparams
from datetime import datetime
from plots.plot_functions import plot_losscurves
from systems.utils import listDict2dictList, sample_binpos
from data_generation.utils import get_input_tensors
from density_training.utils import NeuralNetwork, load_nn, load_args, load_dataloader, get_output_variables, create_configs
from motion_planning.utils import time2idx
import pickle


def evaluate_le(dataloader, model, args, optimizer=None, mode="val"):

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
        max_loss_xe[batch, :], _ = torch.max(torch.abs(xe_nn-xe_true), dim=0)
        max_loss_rho_w[batch] = args.rho_loss_weight * torch.max(torch.abs(rholog_nn-torch.log(rholog_true)))
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
    loss_xe = ((xe_nn - xe_true) ** 2).mean()
    # if mask.any():
    #     loss_rho = ((rho_log_nn[mask] - rho_true[mask]) ** 2).mean()
    #     rho_log_nn = rho_log_nn[torch.logical_not(mask)]
    #     rho_true = rho_true[torch.logical_not(mask)]
    #if not mask.all():
    mask = torch.logical_or(rholog_true.abs() > 1e30, torch.isnan(rholog_true))
    if mask.any():
        print("invalid rho set to 1e30")
        rholog_true[mask] = 1e30
    loss_rho = ((rholog_nn - rholog_true) ** 2).mean()

    # mask = torch.logical_and(rholog_true > 0, rholog_nn > 0)
    # if mask.any():
    #     rholog_nn[mask] = torch.log(rholog_nn[mask])
    #     rholog_true[mask] = torch.log(rholog_true[mask])
    # loss_rho_train = ((rholog_nn - rholog_true) ** 2).mean()

    return loss_xe, args.rho_loss_weight * loss_rho #, args.rho_loss_weight * loss_rho_train

def evaluate_fpe(dataloader, model, args, optimizer=None, mode="val"):

    if mode == "train" and optimizer is not None:
        model.train()
    elif mode == "val":
        model.eval()
    else:
        raise NotImplemented('Mode not defined')

    total_loss_fpe = 0
    total_loss_mc = 0
    system = Car(args)
    #dt = args.dt_sim * args.factor_pred

    for batch, (u_params, xref0, t, density_map, xref_traj, uref_traj) in enumerate(dataloader):
        # u_params = u_params.to(args.device)
        # xref0 = xref0.to(args.device)
        # xref_traj = xref_traj.to(args.device)
        # uref_traj = uref_traj.to(args.device)
        #t_vec = t_vec[0]#.to(args.device)
        bs = t.shape[0]
        k = torch.arange(0, bs)
        for j in range(10):
            xe_t = system.sample_xe(bs)
            binpos_mc, xe_mc = sample_binpos(bs, args.bin_number, args.bin_wide)
            for jj in range(10):
                i = torch.randint(0, t.shape[1], (1,)).item()
                for s in ["MC", "FPE"]:
                    iter_loss = torch.tensor([0.]).to(args.device)
                    #for i, t in enumerate(t_vec):
                    if s == "FPE":
                        # compute loss of FPE
                        # xe_t = xe_t.to(args.device)
                        x_t = (xe_t + xref_traj[:, :, [i]])

                        f = system.f_func(x_t, xref_traj[:, :, [i]], uref_traj[:, :, [i]], noise=False)
                        divf = system.divf_func(x_t, xref_traj[:, :, [i]], uref_traj[:, :, [i]])

                        input, input_map = get_input_tensors(u_params, xref0, xe_t, t[:, i], args)
                        input = input.to(args.device)
                        input.requires_grad = True
                        rho_nn = model(input)
                        loss_fpe = loss_function_fpe(rho_nn, input, input_map, f.to(args.device), divf.to(args.device))
                        iter_loss += loss_fpe
                    else:
                        # compute loss of mc comparison
                        rho_mc = density_map[k, binpos_mc[k, 0], binpos_mc[k, 1], binpos_mc[k, 2], binpos_mc[k, 3], i].to(args.device)
                        # xe_mc = xe_mc.to(args.device)
                        input, _ = get_input_tensors(u_params, xref0, xe_mc, t[:, i], args)
                        input = input.to(args.device)
                        rho_nn = model(input)
                        loss_mc = loss_function_fpemc(rho_nn, rho_mc)
                        iter_loss += loss_mc

                    #iter_loss = 1e4 * iter_loss / len(t_vec)
                    if mode == "train":
                        optimizer.zero_grad()
                        iter_loss.backward()
                        optimizer.step()
                total_loss_mc += loss_fpe.item()
                total_loss_fpe += loss_mc.item()
    total_loss = {
        "loss_fpe": total_loss_fpe / len(dataloader),
        "loss_mc": total_loss_mc / len(dataloader)
    }
    return total_loss

def loss_function_fpe(rho_nn, input, input_map, f, divf):
    drhodinput = torch.autograd.grad(rho_nn.sum(), input, create_graph=True)[0]
    drhodt = drhodinput[:, input_map["t"]]
    drhodx = drhodinput[:, input_map["xe0"]]
    ddrhodxdinput = torch.autograd.grad(drhodx.sum(), input, create_graph=True)[0]
    ddrhodxdx = ddrhodxdinput[:, input_map["xe0"]].sum(dim=1)
    dfrhodx = (f.squeeze(-1) * drhodx).sum(dim=1) + divf * rho_nn.squeeze(-1)

    loss_fpe = ((drhodt + dfrhodx - ddrhodxdx) ** 2).mean()
    return loss_fpe

def loss_function_fpemc(rho_nn, rho_mc):
    loss_mc = ((rho_nn.squeeze(-1) - rho_mc) ** 2).mean()
    return loss_mc

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

        train_dataloader, validation_dataloader = load_dataloader(args)
        args = load_args(config, args)
        model, optimizer = load_nn(train_dataloader.dataset.num_inputs, train_dataloader.dataset.num_outputs, args)
            #len(train_dataloader.dataset.data[0][0]), len(train_dataloader.dataset.data[0][1]), args)

        patience = 0
        test_loss_best = float('Inf')
        test_loss = []
        train_loss = []
        for epoch in range(args.epochs):
            if args.equation == "LE":
                loss_train = evaluate_le(train_dataloader, model, args, optimizer=optimizer, mode="train")
                loss_test = evaluate_le(validation_dataloader, model, args, mode="val")
                #loss_test = loss_train
                print(f"Epoch {epoch},    Train loss: {(loss_train['loss']):.3f}  (x: {(loss_train['loss_xe']):.5f}),    Test loss: {(loss_test['loss']):.5f}  (x: {(loss_test['loss_xe']):.5f})")
            else:
                loss_train = evaluate_fpe(train_dataloader, model, args, optimizer=optimizer, mode="train")
                loss_test = evaluate_fpe(validation_dataloader, model, args, mode="val")
                print(f"Epoch {epoch},    Train loss fpe: {(loss_train['loss_fpe']):.5f}  (mc: {(loss_train['loss_mc']):.6f}),    Test loss fpe: {(loss_test['loss_fpe']):.5f}  (mc: {(loss_test['loss_mc']):.6f})")


            train_loss.append(loss_train)
            test_loss.append(loss_test)
            # if test_loss[-1]["loss"] > test_loss_best:
            #     patience += 1
            # else:
            #     patience = 0
            #     test_loss_best = test_loss[-1]["loss"]
            # if args.patience and patience >= args.patience:
            #     break
            # if epoch % args.lr_step_epoch == args.lr_step_epoch - 1: #epoch > 2000 and
            #     for g in optimizer.param_groups:
            #         g['lr'] = g['lr'] / args.lr_step

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
                #plot_losscurves(result, "t%d_" % (epoch) + short_name, args, filename=maxErrorPlot_name, type="maxError")
                plot_losscurves(result, "t%d_" % (epoch) + short_name, args, filename=lossPlot_name)

            if epoch % 50 == 49:
                torch.save([model.state_dict(), args, result], nn_name)
        #print(f"Best test loss after epoch {epoch}: {test_loss_best} \n")
        torch.save([model.state_dict(), args, result], nn_name)
        plot_losscurves(result, "final_t%d_" % (epoch) + short_name, args, filename=lossPlot_name)
        # with open(args.path_plot_loss + lossPlot_name + "_data", "wb") as f:
        #     pickle.dump(result, f)

    print("Done!")

