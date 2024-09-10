import torch
from torch.utils.data import Dataset

class AtomicDataset(Dataset):
    def __init__(self, X, n_atoms_per_structure):
        self.X = X
        self.n_structures = len(n_atoms_per_structure)
        self.n_atoms_per_structure = n_atoms_per_structure
        self.index = self.generate_atomstructure_index(self.n_atoms_per_structure)
        assert (torch.sum(n_atoms_per_structure) == len(X))

    def __len__(self):
        return self.n_structures

    def __getitem__(self, idx):
        if type(idx) == list:
            x_indexes = []
            
            for i in idx:                
                x_indexes.append((self.index == i).nonzero(as_tuple = True)[0])
                
            x_indexes = torch.hstack(x_indexes)
            
            return self.X[x_indexes], idx, self.generate_atomstructure_index(self.n_atoms_per_structure[idx])
                
        else:
            x_indexes = (self.index == idx).nonzero(as_tuple = True)[0]
            return (self.X[x_indexes], idx, self.n_atoms_per_structure[idx])

    
    def generate_atomstructure_index(self, n_atoms_per_structure):
        total_index = []
        for i, atoms in enumerate(n_atoms_per_structure):
            indiv_index = torch.zeros(atoms) + i
            total_index.append(indiv_index)
        total_index = torch.hstack(total_index)
        return total_index.long()

def collate(batch):
    for x, idx, index in batch:
        return (x , idx, index)