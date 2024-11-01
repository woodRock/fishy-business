import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_inner_functions: int = 10,
        dropout_rate: float = 0.1,
    ) -> None:
        """Kalomogorov-Arnold Neural Network (KAN) module.

        Args:
            input_dim (int): the number of dimensions in the input.
            output_dim (int): the number of dimensions in the output.
            hidden_dim (int): the number of dimensions in the hidden layer. Defaults to 64.
            num_inner_functions (int): the number of inner functions. Defaults to 10.
            dropout_rate (float): the dropout rate. Defaults to 0.1.

        References:
            1. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F.,
                Halverson, J., Soljačić, M., ... & Tegmark, M. (2024).
                Kan: Kolmogorov-arnold networks.
                arXiv preprint arXiv:2404.19756.
            2. Srivastava, N., Hinton, G., Krizhevsky, A.,
                Sutskever, I., & Salakhutdinov, R. (2014).
                Dropout: a simple way to prevent neural networks from overfitting.
                The journal of machine learning research, 15(1), 1929-1958.
            3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
                I., & Salakhutdinov, R. R. (2012).
                Improving neural networks by preventing co-adaptation of feature detectors.
                arXiv preprint arXiv:1207.0580.
            4. Hendrycks, D., & Gimpel, K. (2016).
                Gaussian error linear units (gelus).
                arXiv preprint arXiv:1606.08415.
            5. Loshchilov, I., & Hutter, F. (2017).
                Decoupled weight decay regularization.
                arXiv preprint arXiv:1711.05101.
            6. Loshchilov, I., & Hutter, F. (2017).
                Decoupled weight decay regularization.
                arXiv preprint arXiv:1711.05101.
            7. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
                Rethinking the inception architecture for computer vision.
                In Proceedings of the IEEE conference on computer vision
                and pattern recognition (pp. 2818-2826).
        """
        super(KANLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inner_functions = num_inner_functions

        # Inner functions (vectorized)
        self.inner_linear1 = nn.Linear(input_dim, hidden_dim)
        self.inner_bn1 = nn.BatchNorm1d(hidden_dim)
        self.inner_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.inner_bn2 = nn.BatchNorm1d(hidden_dim)
        self.inner_linear3 = nn.Linear(
            hidden_dim, num_inner_functions * (2 * input_dim + 1)
        )

        # Outer functions (vectorized)
        self.outer_linear1 = nn.Linear(num_inner_functions, hidden_dim)
        self.outer_bn1 = nn.BatchNorm1d(hidden_dim)
        self.outer_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.outer_bn2 = nn.BatchNorm1d(hidden_dim)
        self.outer_linear3 = nn.Linear(hidden_dim, output_dim)

        # Dropout layer (Srivastava 2014, Hinton 2012)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.size(0)

        # Inner functions (vectorized)
        # GELU activation function (Hendrycks 2016)
        inner = F.gelu(self.inner_bn1(self.inner_linear1(x)))
        # Dropout layer (Srivastava 2014, Hinton 2012)
        inner = self.dropout(inner)
        inner = F.gelu(self.inner_bn2(self.inner_linear2(inner)))
        inner = self.dropout(inner)
        inner = self.inner_linear3(inner)
        inner = inner.view(batch_size, 2 * self.input_dim + 1, self.num_inner_functions)

        # Add the constant term
        constant_term = torch.ones(
            batch_size, 1, self.num_inner_functions, device=x.device
        )
        inner = torch.cat([inner, constant_term], dim=1)

        # Sum across the inner functions
        summed = torch.sum(inner, dim=1)

        # Outer functions (vectorized)
        outer = F.gelu(self.outer_bn1(self.outer_linear1(summed)))
        outer = self.dropout(outer)
        outer = F.gelu(self.outer_bn2(self.outer_linear2(outer)))
        outer = self.dropout(outer)
        output = self.outer_linear3(outer)

        return output


class KAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_inner_functions: int = 10,
        dropout_rate: float = 0.1,
        num_layers: int = 5,
    ) -> None:
        """Stacked Kalomogorov-Arnold Neural Network (KAN) module.

        Args:
            input_dim (int): the number of dimensions in the input.
            output_dim (int): the number of dimensions in the output.
            hidden_dim (int): the number of dimensions in the hidden layer. Defaults to 64.
            num_inner_functions (int): the number of inner functions. Defaults to 10.
            dropout_rate (float): the dropout rate. Defaults to 0.1.
            num_layers (int): the number of layers. Defaults to 5.
        """
        super(KAN, self).__init__()
        self.layers = nn.ModuleList(
            [
                KANLayer(
                    input_dim,
                    output_dim if i == (num_layers - 1) else input_dim,
                    hidden_dim,
                    num_inner_functions,
                    dropout_rate,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """A forward pass through the encoder module.

        Args:
            x (torch.Tensor): the input tensor for the encoder.
            mask (torch.Tensor): the mask for the encoder.

        Returns:
            x (torch.Tensor): output tensorfrom a forward pass of the encoder.
        """
        for layer in self.layers:
            x = layer(x)
        return x


import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_epochs = 100_000

    # Set the device to cuda if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Make a dataset of sine waves.
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    x, y = x.to(device), y.to(device)

    model = StackedKAN(
        input_dim=1,
        output_dim=1,
        hidden_dim=64,
        num_inner_functions=10,
        dropout_rate=0.2,
        num_layers=1,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")

    y_pred = model(x)
    plt.plot(x.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    plt.savefig("figures/sine_wave.png")
