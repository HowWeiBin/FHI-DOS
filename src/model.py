import torch

class SOAP_DOS(torch.nn.Module): 
    def __init__(self, input_dims, intermediate_layers, n_train, target_dims, adaptive):
        super(SOAP_DOS,self).__init__()
        if intermediate_layers:
            all_layers = [input_dims] + intermediate_layers + [target_dims]
            self.nn = torch.nn.Sequential()
            for index in range(len(all_layers)-1):
                next_layer = torch.nn.Linear(all_layers[index],all_layers[index + 1])
                self.nn.append(next_layer)
                self.nn.append(torch.nn.SiLU())
            self.nn.pop(-1)
        else:
            self.nn = torch.nn.Linear(input_dims, target_dims)

        if adaptive:
            initial_reference = torch.zeros(n_train, requires_grad = True)
            self.reference = torch.nn.parameter.Parameter(initial_reference)
            print ("Adaptive Energy Reference Used")
        else:
            print ("Fixed Energy Reference Used")
    def forward(self, x):
        result = self.nn(x)
        return result
