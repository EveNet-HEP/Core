from torch import Tensor, nn
import torch
from evenet.network.layers.norm import create_normalization
from evenet.network.layers.activation import create_activation, create_dropout
from evenet.network.layers.mask import FillingMasking
from evenet.network.layers.utils import LayerScale

def create_residual_connection(skip_connection: bool, input_dim: int, output_dim: int) -> nn.Module:
    if input_dim == output_dim or not skip_connection:
        return nn.Identity()

    return nn.Linear(input_dim, output_dim)


class GRUGate(nn.Module):
    def __init__(self, hidden_dim, gate_initialization: float = 2.0):
        super(GRUGate, self).__init__()

        self.linear_W_r = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.linear_W_z = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.linear_W_g = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_U_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.gate_bias = nn.Parameter(torch.ones(hidden_dim) * gate_initialization)

    def forward(self, vectors: Tensor, residual: Tensor) -> Tensor:
        r = torch.sigmoid(self.linear_W_r(vectors) + self.linear_U_r(residual))
        z = torch.sigmoid(self.linear_W_z(vectors) + self.linear_U_z(residual) - self.gate_bias)
        h = torch.tanh(self.linear_W_g(vectors) + self.linear_U_g(r * residual))

        return (1 - z) * residual + z * h


class GRUBlock(nn.Module):
    __constants__ = ['input_dim', 'output_dim', 'skip_connection', 'hidden_dim']

    # noinspection SpellCheckingInspection
    def __init__(
            self,
            input_dim: int,
            hidden_dim_scale: float,
            output_dim: int,
            normalization_type: str,
            activation_type: str,
            dropout: float,
            skip_connection: bool = False
    ):
        super(GRUBlock, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = int(round(hidden_dim_scale * input_dim))
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.normalization_type = normalization_type
        self.dropout = dropout
        self.activation_type = activation_type

        # Create normalization layer for keeping values in good ranges.
        self.normalization = create_normalization(self.normalization_type, input_dim)

        # The primary linear layers applied before the gate
        self.linear_1 = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            create_activation(self.activation_type, self.hidden_dim),
            create_dropout(self.dropout)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            create_activation(self.activation_type, output_dim),
            create_dropout(self.dropout)
        )

        # GRU layer to gate and project back to output. This will also handle the
        # self.gru = GRUGate(output_dim, input_dim)
        self.gru = GRUGate(output_dim)

        # Possibly need a linear layer to create residual connection.
        self.residual = create_residual_connection(skip_connection, input_dim, output_dim)

    def forward(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
        """

        :param x: shape: (batch_size, num_object, input_dim)
        :param sequence_mask: (batch_size, num_object, 1)
        :return:
            - output: shape (batch_size, num_object, output_dim)
            - sequence_mask: shape (batch_size, num_object)
        """
        batch_size, num_object, input_dim = x.shape

        # -----------------------------------------------------------------------------
        # Apply normalization first for this type of linear block.
        # output: (batch_size, num_object, input_dim)
        # -----------------------------------------------------------------------------
        output = self.normalization(x, sequence_mask)

        # -----------------------------------------------------------------------------
        # Flatten the data and apply the basic matrix multiplication and non-linearity.
        # output: (batch_size * num_object, input_dim)
        # -----------------------------------------------------------------------------
        output = output.reshape(num_object * batch_size, self.input_dim)

        # -----------------------------------------------------------------------------
        # Apply linear layer with expansion in the middle.
        # output: (batch_size * num_object, hidden_dim)
        # -----------------------------------------------------------------------------
        output = self.linear_1(output)
        output = self.linear_2(output)

        # --------------------------------------------------------------------------
        # Reshape the data back into the time-series and apply normalization.
        # output: (batch_size, num_object, output_dim)
        # --------------------------------------------------------------------------
        output = output.reshape(batch_size, num_object, self.output_dim)

        # -----------------------------------------------------------------------------
        # Apply gating mechanism and skip connection using the GRU mechanism.
        # output: (batch_size, num_object, output_dim)
        # -----------------------------------------------------------------------------
        if self.skip_connection:
            output = self.gru(output, self.residual(x))

        return output * sequence_mask


class ResNetDense(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 activation: str = 'silu',
                 dropout: float = 0.0,
                 layer_scale_init: float = 1.0):
        super(ResNetDense, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_scale_init = layer_scale_init

        # Define the layers
        self.residual_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
            create_activation(activation, hidden_dim),
            nn.Dropout(self.dropout))
            for i in range(num_layers)])
        self.layer_scale = LayerScale(self.layer_scale_init, hidden_dim)
        self.out_layer = create_residual_connection(
            skip_connection=True,
            input_dim=self.hidden_dim,
            output_dim=output_dim
        )
    def forward(self, x):
        residual = self.residual_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.layer_scale(x)
        return self.out_layer(residual + x)


def create_linear_block(
        linear_block_type: str,
        input_dim: int,
        hidden_dim_scale: float,
        output_dim: int,
        normalization_type: str,
        activation_type: str,
        dropout: float,
        skip_connection: bool
) -> nn.Module:
    if linear_block_type == "GRU":
        return GRUBlock(input_dim,
                        hidden_dim_scale,
                        output_dim,
                        normalization_type,
                        activation_type,
                        dropout,
                        skip_connection)
