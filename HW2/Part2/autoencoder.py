import numpy as np
import torch.nn as nn
import torch

class AutoEncoderNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        """
        Your autoencoder model definition should go here
        """


        # Input -> Hidden -> Bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 2), output_dim)
        )

        # Bottleneck -> Hidden -> Output
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, int(input_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 2), input_dim)
        )


    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function should map a given data matrix onto the bottleneck hidden layer
        :param x: the input data matrix of type torch.Tensor
        :return: the resulting projected data matrix of type torch.Tensor
        """
        
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Your autoencoder model's forward pass operations should go here
        :param x: the input data matrix of type torch array
        :return: the neural network output as torch array
        """

        encoded_data = self.project(x)
        return self.decoder(encoded_data)

class AutoEncoder:

    def __init__(self, input_dim: int, projection_dim: int, learning_rate: float, iteration_count: int):
        """
        Initializes the Auto Encoder method
        :param input_dim: the input data space dimensionality
        :param projection_dim: the projection space dimensionality
        :param learning_rate: the learning rate for the auto encoder neural network training
        :param iteration_count: the number epoch for the neural network training
        """
        self.input_dim = input_dim
        self.projection_matrix = projection_dim
        self.epochs = iteration_count
        self.autoencoder_model = AutoEncoder(input_dim, projection_dim)
        """
            Your optimizer and loss definitions should go here
        """
        
        self.optimzer = torch.optim.Adam(self.autoencoder_model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def fit(self, x: torch.Tensor) -> None:
        """
        Trains the auto encoder nn on the given data matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should train the auto encoder to minimize the reconstruction error
        please do not forget to put the neural network model into the training mode before training
        """

        self.autoencoder_model.train()
        for _ in range(self.epochs):
            self.optimzer.zero_grad()
            decoded_data = self.autoencoder_model(x)
            loss = self.loss(decoded_data, x)
            loss.backward()
            self.optimzer.step()


    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        After training the nn a given dataset,
        this function uses the learned model to project new data instances
        :param x: the data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        please do not forget to put the neural network model into the evaluation mode before projecting data instances
        """
        self.autoencoder_model.eval()
        with torch.no_grad():
            projected_data = self.autoencoder_model.project(x)
        return projected_data

