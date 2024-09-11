import torch
import yaml
import numpy as np
import ase
import ase.io
import argparse
import rascaline

from model import *
from utils import generate_atomstructure_index
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "hypers_path",
            help="Path to hyperparameters .yaml file",
            type=str,
        )
    parser.add_argument(
            "inference_structure_path",
            help="Path to inference structures .xyz file"
        )
    args = parser.parse_args()
    with open(args.hypers_path, "r") as f:
        TOTAL_HYPERS = yaml.safe_load(f)
    structures = ase.io.read(f'{args.inference_structure_path}', ":")
    n_structures = len(structures)
    n_atoms = torch.tensor([len(i) for i in structures])
    best_model_state_dict = torch.load(f'{TOTAL_HYPERS['PATH_HYPERS']['MODEL_OUTPUT_PATH']}/model/best_model.pt', weights_only = True)
    

    lower_bound = TOTAL_HYPERS['DOS_HYPERS']['MIN_ENERGY']
    upper_bound = TOTAL_HYPERS['DOS_HYPERS']['MAX_ENERGY']
    dx = torch.tensor(TOTAL_HYPERS['DOS_HYPERS']['GRID_INTERVAL']).double()
    n_points = torch.ceil((upper_bound - lower_bound)/dx)
    x_dos = lower_bound + torch.arange(n_points)*dx
    n_outputs = len(x_dos)

    SOAP_HYPERS = TOTAL_HYPERS['SOAP_HYPERS']
    SOAP_HYPERS =  {key.lower(): value for key, value in SOAP_HYPERS.items()}

    calculator = rascaline.SoapPowerSpectrum(**SOAP_HYPERS)
    R_total_soap = calculator.compute(structures)
    R_total_soap = R_total_soap.keys_to_samples("center_type")
    R_total_soap = R_total_soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

    atomic_soap = torch.tensor(R_total_soap.block(0).values)
    full_atomstructure_index = generate_atomstructure_index(n_atoms)

    if TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['ADAPTIVE']:
        n_train = len(best_model_state_dict['reference'])
        model = SOAP_DOS(atomic_soap.shape[1], TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['INTERMEDIATE_LAYERS'], n_train,
                            n_outputs, TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['ADAPTIVE'])
    else:
        model = SOAP_DOS(atomic_soap.shape[1], TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['INTERMEDIATE_LAYERS'], 1,
                            n_outputs, TOTAL_HYPERS['ARCHITECTURAL_HYPERS']['ADAPTIVE'])

    model.load_state_dict(best_model_state_dict)

    model.eval() 
    all_pred = model.forward(atomic_soap.float())
    structure_results = torch.zeros([n_structures, n_outputs])
    structure_results = structure_results.index_add_(0, full_atomstructure_index, all_pred)

    torch.save(structure_results, f'./inference.pt')

if __name__ == "__main__":
    main()