import os
import argparse
import ase
import ase.io
from scipy.interpolate import CubicHermiteSpline
import rascaline
import yaml
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import time
import _codecs

from utils import *
from AtomicDataset import *
from model import *

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "hypers_path",
            help="Path to hyperparameters .yaml file",
            type=str,
        )

    args = parser.parse_args()
    with open(args.hypers_path, "r") as f:
        TOTAL_HYPERS = yaml.safe_load(f)

    dataset_directory = TOTAL_HYPERS['PATH_HYPERS']['FHI_OUTPUTS_PATH']
    output_directory = TOTAL_HYPERS['PATH_HYPERS']['MODEL_OUTPUT_PATH']
    try:
        os.mkdir(f"{output_directory}")
    except:
        None
    

    n_structures = len(os.listdir(dataset_directory))

    eigenenergies = []
    structures = []
    Ef = []

    SOAP_saved = os.path.exists(f"{output_directory}/data/structural_soap.pt")
    splines_saved = os.path.exists(f"{output_directory}/data/total_splines.pt")
    structures_saved = os.path.exists(f"{output_directory}/data/structures.xyz")

    if SOAP_saved and splines_saved and structures_saved:
        print ('Using precalculated Splines and SOAP')
        try:
            structure_soap = torch.load(f"{output_directory}/data/structural_soap.pt", weights_only = True)
        except:
            structure_soap = np.load(f"{output_directory}/data/structural_soap.pt", allow_pickle = True)
        structure_splines = torch.load(f"{output_directory}/data/total_splines.pt", weights_only = True)
        structures = ase.io.read(f"{output_directory}/data/structures.xyz", ":")
        n_structures = len(structures)
        n_atoms = torch.tensor([len(i) for i in structures])

        lower_bound = TOTAL_HYPERS['DOS_HYPERS']['MIN_ENERGY']
        upper_bound = TOTAL_HYPERS['DOS_HYPERS']['MAX_ENERGY']
        dx = torch.tensor(TOTAL_HYPERS['DOS_HYPERS']['GRID_INTERVAL']).double()
        n_points = torch.ceil((upper_bound - lower_bound)/dx)
        x_dos = lower_bound + torch.arange(n_points)*dx
        spline_upper_bound = TOTAL_HYPERS['DOS_HYPERS']['SPLINE_MAX_ENERGY']
        n_points = torch.ceil((spline_upper_bound - lower_bound)/dx)
        spline_positions = lower_bound + torch.arange(n_points)*dx

    else:
        for i in (os.listdir(dataset_directory)):
            ref_path = dataset_directory + i + "/aims.out"
            with open(ref_path, "r") as ref_f:
                ref_lines = ref_f.readlines()
            if "Have a nice day." in ref_lines[-2]:
                structure_i = ase.io.read(ref_path)
                structures.append(structure_i)
                for j in range(len(ref_lines)-1, 0, -1):
                    if '| Chemical potential (Fermi level):' in ref_lines[j]:
                        Ef.append(float(ref_lines[j].split()[-2]))
                        break
                evalue_path = dataset_directory + str(i) + "/Final_KS_eigenvalues.dat"
                echunk = False # Determines if we are in the output chunk with the eigenenergies
                first = True
                energies = []
                k_energy = []
                with open(evalue_path, 'r') as f:
                    while True:
                        line = f.readline()
                        if "k-point number:" in line:
                            echunk = False #We have reached the start of the next k-point
                            if first: # Save the stored eigenenergies for each k-point, unless its the first one
                                first = False
                            else:
                                energies.append(k_energy)
                        if echunk:
                            try:
                                energy = float(line.split()[-1])
                                k_energy.append(energy)
                            except:
                                pass


                        if "k-point in cartesian units" in line:
                            echunk = True
                            k_energy = []
                        if line == '':
                            energies.append(k_energy)
                            break
                eigenenergies.append(energies)

            else:
                print ('Data Extraction failed for index {}'.format(i))
                continue

        print (f"Extracted Data for {len(eigenenergies)} structures")
        try:
            os.mkdir(f"{output_directory}/data")
        except:
            None

        ase.io.write(f"{output_directory}/data/structures.xyz", structures)
        print ('Now Processing Data for ML')

        n_structures = len(structures)
        n_atoms = torch.tensor([len(i) for i in structures])

        if TOTAL_HYPERS['DOS_HYPERS']['REFERENCE'] == 'FERMI':
            adjusted_eigenenergies = []
            for j, i in enumerate(eigenenergies):
                eigenenergy_i = torch.tensor(i) - Ef[j]
                adjusted_eigenenergies.append(eigenenergy_i)
            eigenenergies = adjusted_eigenenergies
        elif TOTAL_HYPERS['DOS_HYPERS']['REFERENCE'] == 'HARTREE':
            pass
        else:
            print ("Energy Reference not recognized, only 'HARTREE' AND 'FERMI' is supported")
            exit()

        sigma = torch.tensor(TOTAL_HYPERS['DOS_HYPERS']['SMEARING']) #Gaussian Smearing for the eDOS
        lower_bound = TOTAL_HYPERS['DOS_HYPERS']['MIN_ENERGY']
        upper_bound = TOTAL_HYPERS['DOS_HYPERS']['MAX_ENERGY']

        dx = torch.tensor(TOTAL_HYPERS['DOS_HYPERS']['GRID_INTERVAL']).double()
        n_points = torch.ceil((upper_bound - lower_bound)/dx)
        x_dos = lower_bound + torch.arange(n_points)*dx

        k_normalization = torch.tensor([len(i) for i in eigenenergies])
        normalization = (1/torch.sqrt(2*torch.tensor(np.pi)*sigma**2)/n_atoms/k_normalization).double()

        spline_upper_bound = TOTAL_HYPERS['DOS_HYPERS']['SPLINE_MAX_ENERGY']
        n_points = torch.ceil((spline_upper_bound - lower_bound)/dx)
        spline_positions = lower_bound + torch.arange(n_points)*dx
        structure_splines = []

        for j, i in enumerate(eigenenergies):
            def value_fn(x):
                l_dos_E = torch.sum(torch.exp(-0.5*((x - i.view(-1,1))/sigma)**2), dim = 0) * 2 * normalization[j]
                return l_dos_E
            def derivative_fn(x):
                dfn_E = torch.sum(torch.exp(-0.5*((x - i.view(-1,1))/sigma)**2) *
                                (-1 * ((x - i.view(-1,1))/sigma)**2), dim =0) * 2 * normalization[j]
                return dfn_E

            spliner = CubicHermiteSpline(spline_positions, value_fn(spline_positions), derivative_fn(spline_positions))
            structure_splines.append(torch.tensor(spliner.c))

        structure_splines = torch.stack(structure_splines)

        torch.save(structure_splines, f"{output_directory}/data/total_splines.pt")

        SOAP_HYPERS = TOTAL_HYPERS['SOAP_HYPERS']
        SOAP_HYPERS =  {key.lower(): value for key, value in SOAP_HYPERS.items()}

        calculator = rascaline.SoapPowerSpectrum(**SOAP_HYPERS)
        R_total_soap = calculator.compute(structures)
        R_total_soap = R_total_soap.keys_to_samples("center_type")
        R_total_soap = R_total_soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

        atomic_soap = []
        for structure_i in range(n_structures):
            a_i = R_total_soap.block(0).samples["system"] == structure_i
            atomic_soap.append(torch.tensor(R_total_soap.block(0).values[a_i, :]))

        if (torch.sum(n_atoms - n_atoms[0])):
            structure_soap = np.array(atomic_soap, dtype = 'object')
            with open(f"{output_directory}/data/structural_soap.pt", 'wb') as f:
                np.save(f, structure_soap)

        else:
            structure_soap = torch.stack(atomic_soap)
            torch.save(structure_soap, f"{output_directory}/data/structural_soap.pt")

    try:
        os.mkdir(f"{output_directory}/model")
    except:
        None

    state_path = f"{output_directory}/model/model_checkpoint.pt"
    optimizer_state_path = f"{output_directory}/model/optimizer.pt"
    scheduler_state_path = f"{output_directory}/model/scheduler.pt"
    final_state_path = f"{output_directory}/model/best_model.pt"
    parameter_state_path = f"{output_directory}/model/parameters.pt"

    train_index, val_index = generate_train_val_split(n_structures)
    n_train = len(train_index)

    if (torch.sum(n_atoms - n_atoms[0])):
        atomic_soap = torch.vstack(structure_soap.tolist())
        soap_train = torch.vstack(structure_soap[train_index].tolist()).float()
    else:
        atomic_soap = structure_soap.reshape(-1, structure_soap.shape[-1])
        soap_train = structure_soap[train_index].reshape(-1, structure_soap.shape[-1]).float()


    train_features = AtomicDataset(soap_train, n_atoms[train_index])
    full_atomstructure_index = generate_atomstructure_index(n_atoms)

    Sampler = RandomSampler(train_features)
    BSampler = BatchSampler(Sampler, batch_size = TOTAL_HYPERS['TRAINING_HYPERS']['BATCH_SIZE'], drop_last = False)
    traindata_loader = DataLoader(train_features, sampler = BSampler, collate_fn = collate)

    n_outputs = len(x_dos)

    model = SOAP_DOS(soap_train.shape[1], TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['INTERMEDIATE_LAYERS'], n_train,
                    n_outputs, TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['ADAPTIVE'])

    if not TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['ADAPTIVE']: #target only needs to be defined once
        train_target = evaluate_spline(structure_splines[train_index], spline_positions, x_dos + torch.zeros(n_train).view(-1,1)).detach()
        val_target = evaluate_spline(structure_splines[val_index], spline_positions, x_dos + torch.zeros(len(val_index)).view(-1,1)).detach()

    #Check if checkpoint exists:
    scheduler_saved = os.path.exists(scheduler_state_path)
    optimizer_saved = os.path.exists(optimizer_state_path)
    model_saved = os.path.exists(state_path)
    parameters_saved = os.path.exists(parameter_state_path)
    best_state_saved = os.path.exists(final_state_path)

    lr = TOTAL_HYPERS['TRAINING_HYPERS']['LEARNING_RATE']
    n_epochs = TOTAL_HYPERS['TRAINING_HYPERS']['MAX_EPOCHS']
    patience = TOTAL_HYPERS['TRAINING_HYPERS']['PATIENCE']

    opt = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = TOTAL_HYPERS['TRAINING_HYPERS']['LR_DECAY'], patience = patience, threshold = 1e-5, min_lr = TOTAL_HYPERS['TRAINING_HYPERS']['MIN_LR'])
    best_state = copy.deepcopy(model.state_dict())

    train_loss = torch.tensor(100.0)
    val_loss = torch.tensor(100.0)
    best_train_loss = torch.tensor(100.0)
    best_val_loss = torch.tensor(100.0)

    if scheduler_saved & optimizer_saved & model_saved & parameters_saved & best_state_saved:
        opt.load_state_dict(torch.load(optimizer_state_path, weights_only = True))
        model.load_state_dict(torch.load(state_path, weights_only = True))
        scheduler.load_state_dict(torch.load(scheduler_state_path, weights_only = True))
        best_state = torch.load(final_state_path, weights_only = True)
        best_train_loss, best_val_loss = torch.load(parameter_state_path, weights_only = True)

    for epoch in range(n_epochs):
        for x_data, idx, index in traindata_loader:
            def closure():
                opt.zero_grad()
                predictions = model.forward(x_data)
                structure_results = torch.zeros([len(idx), n_outputs])
                structure_results = structure_results.index_add_(0, index, predictions)/n_atoms[train_index[idx]].view(-1,1)
                if TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['ADAPTIVE']:
                    reference = model.reference - torch.mean(model.reference)
                    target = evaluate_spline(structure_splines[train_index[idx]], spline_positions, x_dos + reference[idx].view(-1,1))
                    pred_loss = t_get_mse(structure_results, target, x_dos)
                    pred_loss.backward()
                else:
                    pred_loss = t_get_mse(structure_results, train_target[idx], x_dos)
                    pred_loss.backward()
                return pred_loss
            opt.step(closure)
        with torch.no_grad():
            model.eval()
            all_pred = model.forward(atomic_soap.float())
            structure_results = torch.zeros([n_structures, n_outputs])
            structure_results = structure_results.index_add_(0, full_atomstructure_index, all_pred)/(n_atoms).view(-1,1)
            if TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['ADAPTIVE']:
                reference = model.reference - torch.mean(model.reference)
                target = evaluate_spline(structure_splines[train_index], spline_positions, x_dos + reference.view(-1,1))
                train_loss = t_get_rmse(structure_results[train_index], target, x_dos, perc = False)
                val_loss, final_val_shifts = Opt_RMSE_spline(structure_results[val_index], x_dos, structure_splines[val_index], spline_positions, 50)
            else:
                train_loss = t_get_rmse(structure_results[train_index], train_target, x_dos, perc = False)
                val_loss = t_get_rmse(structure_results[val_index], val_target, x_dos, perc = False)

            if train_loss < best_train_loss:
                best_train_loss = train_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

            scheduler.step(val_loss)

            if (epoch % 100 == 0) or (epoch == n_epochs-1) :
                print (f"Epoch {epoch}: Current loss (Train, Val): {train_loss.item():.4}, {val_loss.item():.4}"
                    f", Best loss (Train, Val): {best_train_loss.item():.4}, {best_val_loss.item():.4}, Learning rate: {scheduler.get_last_lr()}")

                torch.save(model.state_dict(), state_path)
                torch.save(opt.state_dict(), optimizer_state_path)
                torch.save(scheduler.state_dict(), scheduler_state_path)
                torch.save([best_train_loss, best_val_loss], parameter_state_path)
                torch.save(best_state, final_state_path)

    print("--- Time Elapsed: %s seconds---" % (time.time() - start_time))
    torch.save(best_state, final_state_path)

if __name__ == "__main__":
    main()
