# VAE-LSTM-PyTorch
 
# VAE-LSTM-PyTorch

This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) combined with Long Short-Term Memory (LSTM) layers for modelling sequential or time-series data.

## Features

- **VAE with LSTM architecture** – encodes sequences into a latent space using LSTM encoders and decoders.
- **Sequence data support** – designed for tasks involving sequential or temporal patterns (e.g., music generation, speech, sensor data).
- **Training notebook** – includes a Jupyter notebook (`VAE-LSTM-PyTorch.ipynb`) demonstrating training and evaluation.
- **Modular codebase** – separate modules for dataset loading, model definitions, and training loop.

## Project Structure

```
├── Dataset.py             # dataset loader
├── base.py                # base classes and utilities
├── models.py              # VAE and LSTM model definitions
├── VAE-LSTM-PyTorch.ipynb # training and evaluation notebook
├── README.md
└── LICENSE.txt
```

## Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/IamArmanNikkhah/VAE-LSTM-PyTorch.git
cd VAE-LSTM-PyTorch
```

2. **Install dependencies** (requires Python 3.7+ and PyTorch):

```bash
pip install torch torchvision
```

3. **Run the training notebook:**

Open `VAE-LSTM-PyTorch.ipynb` in Jupyter Notebook or JupyterLab to follow the training process and evaluate the model on sample data.

## Usage

You can import the model classes from `models.py` into your own project and train on your datasets. Example:

```python
from models import VAE_LSTM
model = VAE_LSTM(input_dim=128, hidden_dim=256, latent_dim=64, num_layers=2)
# define optimizer, loss, and training loop...
```

## Contributing

Contributions are welcome! If you have suggestions for improving the model, adding new features or datasets, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License – see [LICENSE.txt](LICENSE.txt) for details.
