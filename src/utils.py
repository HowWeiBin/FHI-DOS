import torch
import numpy as np
import ase
import ase.io
import copy
import os


def generate_train_val_split(n_samples):
    """Generate train and test split .

    Args:
        n_samples ([int]): [Total number of samples]

    Returns:
        indices[np.array]: [Indices for train and test split]
    """
    n_structures = n_samples
    n_train = int(0.8 * n_structures)
    train_index = np.arange(n_structures)
    np.random.shuffle(train_index)
    val_index = train_index[n_train:]
    train_index = train_index[:n_train]
    
    return train_index, val_index

def t_get_mse(a, b, xdos, perc = False):
    """Compute mean squared error between two Density of States .

    Args:
        a ([tensor]): [Predicted DOS]
        b ([tensor]): [True DOS]
        xdos ([tensor], optional): [Energy axis of DOS]. 
        perc (bool, optional): [Determines if output is absolute error or percentage error]. Defaults to False.

    Returns:
        [float]: [MSE or %MSE]
    """
    if len(a.size()) > 1:
        mse = (torch.trapezoid((a - b)**2, xdos, axis=1)).mean()
    else:
        mse = (torch.trapezoid((a - b)**2, xdos, axis=0)).mean()
    if not perc:
        return mse
    else:
        mean = b.mean(axis = 0)
        std = torch.trapezoid((b - mean)**2, xdos, axis=1).mean()
        return (100 * mse / std)
        
def t_get_rmse(a, b, xdos, perc=False):
    """Compute root mean squared error between two Density of States .

    Args:
        a ([tensor]): [Predicted DOS]
        b ([tensor]): [True DOS]
        xdos ([tensor], optional): [Energy axis of DOS].
        perc (bool, optional): [Determines if output is absolute error or percentage error]. Defaults to False.


    Returns:
        [float]: [RMSE or %RMSE]
    """
    if len(a.size()) > 1:
        rmse = torch.sqrt((torch.trapezoid((a - b)**2, xdos, axis=1)).mean())
    else:
        rmse = torch.sqrt((torch.trapezoid((a - b)**2, xdos, axis=0)).mean())
    if not perc:
        return rmse
    else:
        mean = b.mean(axis = 0)
        std = torch.sqrt((torch.trapezoid((b - mean)**2, xdos, axis=1)).mean())
        return (100 * rmse / std)

def evaluate_spline(spline_coefs, spline_positions, x):
    """ Evaluate splines on selected points .

    Args:
        spline_coefs ([tensor]): [Cubic Hermite Spline Coefficients] 
        spline_positions ([tensor]): [Spline Positions] 
        x ([tensor]): [Points to evaluate splines on]

    Returns:
        [tensor]: [Evaluated spline values]
    """

    interval = torch.round(spline_positions[1] - spline_positions[0], decimals = 4)
    x = torch.clamp(x, min = spline_positions[0], max = spline_positions[-1]- 0.0005)
    indexes = torch.floor((x - spline_positions[0])/interval).long()
    expanded_index = indexes.unsqueeze(dim=1).expand(-1,4,-1)
    x_1 = x - spline_positions[indexes]
    x_2 = x_1 * x_1
    x_3 = x_2 * x_1
    x_0 = torch.ones_like(x_1)
    x_powers = torch.stack([x_3, x_2, x_1, x_0]).permute(1,0,2)
    value = torch.sum(torch.mul(x_powers, torch.gather(spline_coefs, 2, expanded_index)), axis = 1) 
    return value

def Opt_RMSE_spline(y_pred, critical_xdos, target_splines, spline_positions, n_epochs):
    """RMSE on optimal shift of energy axis

    Args:
        y_pred ([tensor]): [Prediction/s of DOS]
        critical_xdos ([tensor]): [Relevant range of energy axis, should correspond to the y_pred]
        target_splines ([list]): [Contains spline coefficients]
        spline_positions ([tensor]): [Contains spline positions]
        n_epochs ([int]): [Number of epochs to run for Gradient Descent (GD)]

    Returns:
        [rmse([float]), optimal_shift[tensor]]: [%RMSE on optimal shift and the optimal shift itself]
    """
    all_shifts = []
    all_mse = []
    optim_search_mse = []
    offsets = torch.arange(-2,2,0.1) #Grid-search is first done to reduce number of epochs needed for gradient descent, typically 50 epochs will be sufficient if searching within 0.1
    with torch.no_grad():
        for offset in offsets:
            shifts = torch.zeros(y_pred.shape[0]) + offset
            shifted_target = evaluate_spline(target_splines, spline_positions, critical_xdos + shifts.view(-1,1))
            loss_i = ((y_pred - shifted_target)**2).mean(dim = 1)
            optim_search_mse.append(loss_i)
        optim_search_mse = torch.vstack(optim_search_mse)
        min_index = torch.argmin(optim_search_mse, dim = 0)
        optimal_offset = offsets[min_index]
    
    offset = optimal_offset
    shifts = torch.nn.parameter.Parameter(offset.float())
    opt_adam = torch.optim.Adam([shifts], lr = 1e-2)
    best_error = torch.zeros(len(shifts)) + 100
    best_shifts = shifts.clone()
    for i in (range(n_epochs)):
        shifted_target = evaluate_spline(target_splines, spline_positions, critical_xdos + shifts.view(-1,1)).detach()
        def closure():
            opt_adam.zero_grad()            
            shifted_target = evaluate_spline(target_splines, spline_positions, critical_xdos + shifts.view(-1,1))
            loss_i = ((y_pred - shifted_target)**2).mean()
            loss_i.backward(gradient = torch.tensor(1), inputs = shifts)
            return loss_i
        mse = opt_adam.step(closure)
            
        with torch.no_grad():
            each_loss = ((y_pred - shifted_target)**2).mean(dim = 1).float() 
            index = each_loss < best_error
            best_error[index] = each_loss[index].clone()
            best_shifts[index] = shifts[index].clone() 
    #Evaluate
    optimal_shift = best_shifts
    shifted_target = evaluate_spline(target_splines, spline_positions, critical_xdos + optimal_shift.view(-1,1))
    rmse = t_get_rmse(y_pred, shifted_target, critical_xdos, perc = False)
    return rmse, optimal_shift

def generate_atomstructure_index(n_atoms_per_structure):
    """Generate a sequence of indices for each atom in the structure .

    Args:
        n_atoms_per_structure ([array]): [Array containing the number of atoms each structure contains]

    Return s:
        [tensor]: [Total index, matching atoms to structure]
    """
    total_index = []
    for i, atoms in enumerate(n_atoms_per_structure):
        indiv_index = torch.zeros(atoms) + i
        total_index.append(indiv_index)
    total_index = torch.hstack(total_index)
    return total_index.long()