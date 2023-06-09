### IMPORTS
# Mathematical operations
import numpy as np

# Deep learning library + differentiable QP solver
import torch
import qpth

# Plots
import matplotlib.pyplot as plt

# Typing
from typing import Literal


### FUNCTION TO FIX THE SEED AND DEVICE
def fix_seed_and_device():
    np.random.seed(44)
    torch.manual_seed(44)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device('cpu')
    return device

device = fix_seed_and_device()


### FUNCTION TO TRAIN A MODEL
def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    n_epochs: int = 10,
    verbose: bool = False
):
    train_losses, val_losses = [], []
    best_model = model.state_dict()
    min_loss = torch.inf
    epochs_wo_improvement = 0
    
    for epoch in range(n_epochs):
        #Training
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            noised = batch['noised']
            target = batch['original']
            out = model(noised)
            loss = loss_fn(out, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # In the case of the OptNet, s0 has to stay non-negative
            #for name, parameter in model.named_parameters():
            #    if name == 's0':
            #        parameter = torch.clamp(parameter, min=0)

            
        if verbose and (epoch+1)%20 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, training loss: {train_loss/len(train_loader):.4g}')
        train_losses.append(train_loss/len(train_loader))

                
        # Validation
        model.eval()
        with torch.no_grad():
            loss = 0
            for batch in val_loader:
                noised = batch['noised']
                target = batch['original']
                out = model(noised)
                loss += loss_fn(out, target).item()
                
        if verbose and (epoch+1)%10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, val loss: {loss/len(val_loader):.4g}')
        val_losses.append(loss/len(val_loader))

        # Update of the best model if the validation loss is at its lowest:
        if val_losses[-1] < min_loss:
            min_loss = val_losses[-1]
            best_model = model.state_dict()
        
        # Keep track of the number of epochs val_loss has not been decreasing for
        if len(val_losses)>=2 and val_losses[-1] >= val_losses[-2]:
            epochs_wo_improvement += 1
        else:
            epochs_wo_improvement = 0
        
        if epochs_wo_improvement == 10:
            if verbose: print("10 epochs without improvement, stopping training")
            break
    
    # Returns best model wrt validation loss
    model.load_state_dict(best_model)
    return train_losses, val_losses, model


def plot_progression(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(train_losses, color='red', label='Train loss')
    ax.plot(val_losses, color='blue', label='Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.legend()
    return fig, ax


### FUNCTIONS TO BUILD THE MATRICES Γ AND D
def build_l1_constraint_matrix(d:int):
    """Builds the matrix Γ (version 1) of a given size."""
    I, J = np.indices((2**d, d))
    f = lambda i,j: (i//2**j)%2
    Gamma = 2*np.vectorize(f)(I, J) - 1
    return torch.from_numpy(Gamma).float()

def build_linf_constraint_matrix(d:int):
    """Builds the matrix Γ (version 2) of a given size."""
    return torch.cat(
        (torch.eye(d), -torch.eye(d)),
        dim = 0,
    )

def build_fd_matrix(T:int):
    """Builds the matrix D of a given size."""
    return torch.cat(
        (-torch.eye(T-1), torch.zeros(T-1,1)),
        dim = 1,
    ) + torch.cat(
        (torch.zeros(T-1,1), torch.eye(T-1)),
        dim = 1,
    )

### MODELS
class OptNetDenoiser(torch.nn.Module):
    def __init__(self, series_length:int, init_mode:Literal['random', 'tv']='random'):
        """
        Creates a network with one OptNet layer.
        `series_length' describes the length of the input time series.
        `init_mode' describes how the matrix G is initialized:
            - 'random' for a random initialization
            - 'tv' for the initialization with the matrix of the l-inf TV problem
        """
        super().__init__()
        n_ineq = 2*(series_length - 1)
        self.solver = qpth.qp.QPFunction(verbose=-1)

        # Fixed problem parameters
        self.Q = torch.autograd.Variable(torch.eye(series_length))
        self.A = torch.autograd.Variable(torch.Tensor())
        self.b = torch.autograd.Variable(torch.Tensor())
        self.Gamma = torch.autograd.Variable(
            build_linf_constraint_matrix(series_length-1)
        )

        # Learnable parameters
        if init_mode == 'random':
            self.D = torch.nn.Parameter(torch.normal(0, 1, (n_ineq, series_length)))
        elif init_mode == 'tv':
            self.D = torch.nn.Parameter(build_fd_matrix(series_length))
        self.theta = torch.nn.Parameter(torch.ones((1,)))
    
    def forward(self, y:torch.Tensor):
        G = torch.mm(self.Gamma, self.D)
        h = self.theta * torch.ones((G.shape[0],))
        return self.solver(self.Q, -y, G, h, self.A, self.b)


class DenseNetDenoiser(torch.nn.Module):
    def __init__(self, series_length:int, init_mode=Literal['random', 'ma']):
        """
        Creates a network with one dense layer.
        `series_length' describes the length of the input time series.
        `init_mode' describes how the weight matrix is initialized:
            - 'random' for a random initialization
            - 'ma' for a moving average : z*_t = (y_{t-1}+y_t+y_{t+1})/3
        """
        super().__init__()
        self.linear = torch.nn.Linear(series_length, series_length)
        if init_mode == 'ma':
            weight = torch.eye(series_length) / 3
            weight[1:,:-1] += torch.eye(series_length-1) / 3
            weight[:-1,1:] += torch.eye(series_length-1) / 3
            bias = torch.zeros(series_length)
            self.linear.weight = torch.nn.Parameter(weight)
            self.linear.bias = torch.nn.Parameter(bias)
        

    def forward(self, x:torch.Tensor):
        return self.linear(x)

