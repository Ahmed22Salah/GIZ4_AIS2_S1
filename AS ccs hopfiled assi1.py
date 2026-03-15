import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    """
    Hopfield Network for storing and recalling patterns.
    
    Parameters
    ----------
    size : int
        Number of neurons in the network (for 5x5 image = 25).
    """

    def __init__(self, size):
        """Initialize the network and create empty weight matrix."""
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Train the network using Hebbian learning.

        Parameters
        ----------
        patterns : list of numpy arrays
            Patterns that the network should memorize.
        """
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += p @ p.T

        np.fill_diagonal(self.weights, 0)
        self.weights = self.weights / self.size

    def recall(self, pattern, steps=5):
        """
        Recall a stored pattern from a noisy input.

        Parameters
        ----------
        pattern : numpy array
            Input pattern (may contain noise).
        steps : int
            Number of update iterations.

        Returns
        -------
        numpy array
            The recovered pattern.
        """
        state = pattern.copy()

        for _ in range(steps):
            state = np.sign(self.weights @ state)
            state[state == 0] = 1

        return state


def plot_image(image, title):
    """
    Display a 5x5 image.

    Parameters
    ----------
    image : numpy array
        Image vector of length 25.
    title : str
        Title of the plot.
    """
    plt.imshow(image.reshape(5, 5), cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

original = np.array([
    [1,-1,1,-1,1],
    [1,-1,1,-1,1],
    [1,-1,1,-1,1],
    [1,-1,1,-1,1],
    [-1,1,1,1,-1]
]).flatten()

# Add noise
noisy = original.copy()
noise_index = np.random.choice(25, 5, replace=False)
noisy[noise_index] *= -1

# Train network
net = HopfieldNetwork(25)
net.train([original])

# Recall pattern
recovered = net.recall(noisy)

plot_image(original, "Original")
plot_image(noisy, "Noisy")
plot_image(recovered, "Recovered")