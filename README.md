# FHI-DOS

Contains a python workflow capable of training a machine learning model from the output files of FHI-AIMS. The directory FHI-outputs comprise of folders which contain the FHI output of each structure in the training set based on hyperparameters defined in a file 'hypers.yaml'. As an example, in each of the folder, 1) adaptive and 2) fixed, run these code.

1. To train the model, run this code, the arguments are 1) the path to the FHI-AIMS outputs, 2) path to directory for training outputs, 3) path to hypers.yaml file. The outputs will be saved in the current directory under data (for the processed data for ML) and model (for the saved models and checkpoints)
```
python ../src/train_model.py ../FHI-outputs ./ ./hypers.yaml 
```
2. To use the model for inference, run this code, the arguments are 1) the path to the model state dictionary .pt file, 2) path to hypers.yaml file, 3) path to the inference structures .xyz file and 4) path to the directory to store inference outputs. The inference outputs will be saved as inference.pt. In this example, we will use the training set structures for inference as well but in practice any .xyz will do.
```
python ../src/inference_model.py ./model/best_model.pt ./hypers.yaml ./data/structures.xyz ./
```

The folders adaptive and fixed difference in terms of the energy reference used.
