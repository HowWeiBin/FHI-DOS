# FHI-DOS

Contains a python workflow capable of training a machine learning model from the output files of FHI-AIMS. The directory FHI-outputs comprise of folders which contain the FHI output of each structure in the training set based on hyperparameters defined in a file 'hypers.yaml'. The training script would read the process the data from the FHI-AIMS output, generating SOAP features and DOS targets, to train a fully connected neural network capable of predicting the electronic density of states of a structure, gridwise, in a user-defined energy window. To integrate the workflow for both the adaptive energy reference and fixed energy reference, the DOS targets will be computed and stored as Cubic Hermite Splines. 

## Installing Dependencies

1. ase
```
pip install --upgrade ase
```
2. rascaline
```
pip install --extra-index-url https://luthaf.fr/nightly-wheels/ rascaline
```
3. yaml
```
pip install pyyaml
```
3. scipy
```
python -m pip install scipy
```
4. torch
You can get further details on how to install torch on the pytorch official website: https://pytorch.org/get-started/locally/. As a minimal installation, one can simply run:
```
pip install torch
```

## Example Usage
As an example, navigate to each of the folder, 1) adaptive and 2) fixed, run these code.

1. To train the model, run this command, the arguments to add is the path to the hypers.yaml file.
```
python ../src/train_model.py ./hypers.yaml 
```
2. To use the model for inference, run this command. The arguments are 1) path to hypers.yaml file, 2) path to the inference structures .xyz file. The inference outputs will be saved as inference.pt in the directory where the command is run. In this example, we will use the training set structures for inference as well but in practice any .xyz will do.
```
python ../src/inference_model.py ./hypers.yaml ./data/structures.xyz
```

The folders adaptive and fixed difference in terms of the energy reference used.

## Hyperparameter Description
This is a brief description of the parameters used in the hypers.yaml file 

```
PATH_HYPERS: Hyperparameters containing the relevant paths
    FHI_OUTPUTS_PATH: Path to the directory containing FHI outputs
    MODEL_OUTPUT_PATH: Path to the directory where the code will store model and relevant checkpoint files in MODEL_OUTPUT_PATH/model and processed data in MODEL_OUTPUT_PATH/data

DOS_HYPERS: Hyperparameters for the construction of the electronic density of states
    SMEARING: Gaussian Smearing Value (eV)
    GRID_INTERVAL: Prediction Energy Grid Interval (eV)
    MIN_ENERGY: Minimum Prediction Energy Grid Value (eV)
    MAX_ENERGY: Maximum Prediction Energy Grid Value (eV)
    SPLINE_MAX_ENERGY: Maximum Spline Energy Grid Value (eV), most relevant when using an Adaptive energy reference during training
    REFERENCE: Energy Reference [FERMI or HARTREE]

SOAP_HYPERS: Rascaline SoapPowerSpectrum hyperparameters, check https://luthaf.fr/rascaline/latest/references/calculators/soap-power-spectrum.html for the documentation
    CUTOFF: Spherical Cutoff (A)
    MAX_RADIAL: Number of radial basis function to use
    MAX_ANGULAR: Number of spherical harmonics to use
    ATOMIC_GAUSSIAN_WIDTH: Width of the atom-centered gaussian creating the atomic density
    CENTER_ATOM_WEIGHT: Weight of the central atom contribution to the features
    RADIAL_BASIS: Type of Radial Basis Functions used
    CUTOFF_FUNCTION: Type of Smoothing cutoff function
    RADIAL_SCALING: Radial scaling of the atomic density around an atom

ARCHITECTURAL_HYPERS: Hyperparameters for a Sequential fully connected Model
INTERMEDIATE_LAYERS: List containing the size of the intermediate layers, [] gives a linear model
ADAPTIVE: Boolean to determine if an adaptive energy reference will be used

TRAINING_HYPERS: Hyperparameters specific to training
LEARNING_RATE: Initial Learning Rate
BATCH_SIZE: Batch Size
LR_DECAY: Learning rate decay, after a certain number of epochs (Patience) where the performance on the validation set does not improve, the learning rate will be decreased by this factor
MIN_LR: Minimum Learning Rate before training stops
PATIENCE: Number of epochs where the validation error does not improve before the learning rate decays
MAX_EPOCHS: Maximum number of epochs 
```

