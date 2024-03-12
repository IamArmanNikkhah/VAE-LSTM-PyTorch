import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, savefig
import torch
from torch.utils.data import DataLoader, TensorDataset
import time


######################################################
#                                                    #
#                   ENCODER                          #
#                                                    #
######################################################

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config
        num_hidden_units = self.config['num_hidden_units']
        n_channel = self.config['n_channel']
        code_size = self.config['code_size']
        self.l_win = config['l_win']

        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=num_hidden_units // 16, kernel_size=(3, n_channel), stride=(2, 1), padding='same')
        self.conv2 = nn.Conv2d(num_hidden_units // 16, out_channels=num_hidden_units // 8, kernel_size=(3, n_channel), stride=(2, 1), padding='same')
        self.conv3 = nn.Conv2d(num_hidden_units // 8, out_channels=num_hidden_units // 4, kernel_size=(3, n_channel), stride=(2, 1), padding='same')
        self.conv4 = nn.Conv2d(num_hidden_units // 4, out_channels=num_hidden_units, kernel_size=(4, n_channel), stride=1, padding='valid')
        self.conv5 = nn.Conv2d(in_channels=num_hidden_units // 4, out_channels=num_hidden_units, kernel_size=(6, n_channel), stride=1, padding='valid')

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_conv_output_shape(), code_size * 4)
        self.code_mean = nn.Linear(code_size * 4, code_size)
        self.code_std_dev = nn.Linear(code_size * 4, code_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension

        if self.l_win == 24:
            x = F.pad(x, (0, 0, 4, 4), mode='reflect')
            x = F.leaky_relu(self.conv1(x))
            print("conv_1:", x.size())
            x = F.leaky_relu(self.conv2(x))
            print("conv_2:", x.size())
            x = F.leaky_relu(self.conv3(x))
            print("conv_3:", x.size())
            x = F.leaky_relu(self.conv4(x))
            print("conv_4:", x.size())

        if self.l_win == 48:
            x = F.leaky_relu(self.conv)
            print("conv_1:", x.size())
            x = F.leaky_relu(self.conv2(x))
            print("conv_2:", x.size())
            x = F.leaky_relu(self.conv3(x))
            print("conv_3:", x.size())
            x = F.leaky_relu(self.conv5(x))
            print("conv_5:", x.size())

        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))

        code_mean = self.code_mean(x)
        code_std_dev = F.relu(self.code_std_dev(x)) + 1e-2  # Ensure std_dev is positive

        # Sampling from the distribution
        normal_dist = dist.Normal(loc=code_mean, scale=code_std_dev)
        code_sample = normal_dist.rsample()  # Reparameterization trick for backpropagation

        print("finish encoder: \n{}".format(self.code_sample))
        print("\n")

        return code_sample, code_mean, code_std_dev




######################################################
#                                                    #
#                   DECODER                          #
#                                                    #
######################################################



class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.is_code_input = config['is_code_input']  # Assuming this is determined elsewhere in your code
        self.num_hidden_units = config['num_hidden_units']
        self.n_channel = config['n_channel']
        self.l_win = config['l_win']
        self.code_size = config['code_size']
        self.TRAIN_sigma = config['TRAIN_sigma']
        self.sigma = config['sigma']
        self.sigma2_offset = config['sigma2_offset']

        # Layers
        self.fc1 = nn.Linear(self.code_size, self.num_hidden_units)
        self.conv1 = nn.Conv2d(self.num_hidden_units, self.num_hidden_units, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(self.num_hidden_units // 4, self.num_hidden_units // 4, kernel_size=(3, 1), stride=1, padding='same')
        self.conv3 = nn.Conv2d(self.num_hidden_units // 8, self.num_hidden_units // 8, kernel_size=(3, 1), stride=1, padding='same')
        self.conv4 = nn.Conv2d(self.num_hidden_units // 16, self.num_hidden_units // 16, kernel_size=(3, 1), stride=1, padding='same')
        self.conv5 = nn.Conv2d(self.num_hidden_units // 32, self.n_channel, kernel_size=(9, 1), stride=1, padding='valid') ## 16 -> 32 , valid -> same

        self.conv_1 = nn.Conv2d(self.num_hidden_units, 256 * 3, kernel_size=1, padding='same')
        self.conv_2 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding='same')
        self.conv_3 = nn.Conv2d(128, 128, kernel_size=(3, 1), stride=1, padding='same')
        self.conv_4 = nn.Conv2d(32, 32, kernel_size=(3, 1), stride=1, padding='same')
        self.conv_5 = nn.Conv2d(16, self.n_channel, kernel_size=(5, self.n_channel), stride=1, padding='same')

        # Assuming sigma2 and sigma2_offset are defined in config

    def forward(self, code_input=None, code_sample=None):

        encoded = code_input if self.is_code_input else code_sample

        decoded_1 = F.leaky_relu(self.fc1(encoded))
        decoded_1 = decoded_1.view(-1, self.num_hidden_units, 1, 1)
        print("decoded_1 is: {}".format(decoded_1.size()))

        if self.l_win == 24:
            decoded_2 = F.leaky_relu(self.conv1(decoded_1))
            decoded_2 = decoded_2.view(-1, self.num_hidden_units // 4, 4, 1)
            print("decoded_2 is: {}".format(decoded_2.size()))

            decoded_3 = F.leaky_relu(self.conv2(decoded_2))
            decoded_3 = F.pixel_shuffle(decoded_3, upscale_factor=2)
            decoded_3 = decoded_3.view(-1, self.num_hidden_units // 8, 8, 1)
            print("decoded_3 is: {}".format(decoded_3.size()))

            decoded_4 = F.leaky_relu(self.conv3(decoded_3))
            decoded_4 = F.pixel_shuffle(decoded_4, upscale_factor=2)
            decoded_4 = decoded_4.view(-1, self.num_hidden_units // 16, 16, 1)
            print("decoded_4 is: {}".format(decoded_4.size()))

            decoded_5 = F.leaky_relu(self.conv4(decoded_4))
            decoded_5 = F.pixel_shuffle(decoded_5, upscale_factor=2)
            decoded_5 = decoded_5.view(-1, 16, self.num_hidden_units // 16, 1) ## test
            print("decoded_5 is: {}".format(decoded_5.size()))

            decoded = self.conv5(decoded_5)
            print("decoded_6 is: {}".format(decoded.size()))

            decoded = decoded.view(-1, self.l_win, self.n_channel)

        if self.l_win == 48:
            decoded_2 = F.leaky_relu(self.conv_1(decoded_1))
            decoded_2 = decoded_2.view(-1, 256, 3, 1)
            print("decoded_2 is: {}".format(decoded_2.size()))

            decoded_3 = F.leaky_relu(self.conv_2(decoded_2))
            decoded_3 = F.pixel_shuffle(decoded_3, upscale_factor=2)
            decoded_3 = decoded_3.view(-1, 128, 6, 1)
            print("decoded_3 is: {}".format(decoded_3.size()))

            decoded_4 = F.leaky_relu(self.conv_3(decoded_3))
            decoded_4 = F.pixel_shuffle(decoded_4, upscale_factor=2)
            decoded_4 = decoded_4.view(-1, 32, 24, 1)
            print("decoded_4 is: {}".format(decoded_4.size()))

            decoded_5 = F.leaky_relu(self.conv_4(decoded_4))
            decoded_5 = F.pixel_shuffle(decoded_5, upscale_factor=2)
            decoded_5 = decoded_5.view(-1, 16, 48, 1)
            print("decoded_5 is: {}".format(decoded_5.size()))

            decoded = self.conv_5(decoded_5)
            print("decoded_6 is: {}".format(decoded.size()))

            decoded = decoded.view(-1, self.l_win, self.n_channel)


        #if self.TRAIN_sigma == 1:
        #  self.sigma2 = nn.Parameter(torch.tensor(self.sigma ** 2, dtype=torch.float32), requires_grad=True)

        #else:
        #  self.sigma2 = torch.tensor(self.sigma ** 2, dtype=torch.float32)

        #self.sigma2 = torch.add(self.sigma2, self.sigma2_offset)

        print("finish decoder: \n{}".format(self.decoded.size()))
        print('\n')

        return decoded




######################################################
#                                                    #
#                     LSTM                           #
#                                                    #
######################################################


class lstmPyTorchModel(nn.Module):
    def __init__(self, config):
        super(lstmPyTorchModel, self).__init__()

        self.config = config
        self.l_seq = config['l_seq']
        self.l_win = config['l_win']
        self.num_hidden_units_lstm = config['num_hidden_units_lstm']
        self.code_size = config['code_size']

        self.lstm1 = nn.LSTM(self.code_size, self.num_hidden_units_lstm, batch_first=True)
        self.lstm2 = nn.LSTM(self.num_hidden_units_lstm, self.num_hidden_units_lstm, batch_first=True)
        self.lstm3 = nn.LSTM(self.num_hidden_units_lstm, self.code_size, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        return x

    def produce_embeddings(self, model_vae, data, device):
        self.embedding_lstm_train = torch.zeros((data.n_train_lstm, self.l_seq, self.code_size), device=device)
        for i in range(data.n_train_lstm):
            with torch.no_grad():
                input_signal = data.train_set_lstm['data'][i].unsqueeze(0).to(device)
                code_input = torch.zeros((1, self.code_size), device=device)
                self.embedding_lstm_train[i] = model_vae(input_signal, False, code_input)[1]

        print("Finish processing the embeddings of the entire dataset.")
        print("The first a few embeddings are\n{}".format(self.embedding_lstm_train[0, 0:5]))

        self.x_train = self.embedding_lstm_train[:, : self.l_seq - 1]
        self.y_train = self.embedding_lstm_train[:, 1:]

        self.embedding_lstm_test = torch.zeros((data.n_val_lstm, self.l_seq, self.code_size), device=device)
        for i in range(data.n_val_lstm):
            with torch.no_grad():
                input_signal = data.val_set_lstm['data'][i].unsqueeze(0).to(device)
                code_input = torch.zeros((1, self.code_size), device=device)
                self.embedding_lstm_test[i] = model_vae(input_signal, False, code_input)[1]

        self.x_test = self.embedding_lstm_test[:, :self.l_seq - 1]
        self.y_test = self.embedding_lstm_test[:, 1:]

      
    def train(self, cp_callback):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate_lstm'])
        criterion = nn.MSELoss()

        for epoch in range(self.config['num_epochs_lstm']):
            self.train()
            running_loss = 0.0
            for i in range(0, self.x_train.size(0), self.config['batch_size_lstm']):
                batch_x = self.x_train[i:i+self.config['batch_size_lstm']]
                batch_y = self.y_train[i:i+self.config['batch_size_lstm']]

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / (self.x_train.size(0) // config['batch_size_lstm'])
            print(f'Epoch {epoch+1}/{config["num_epochs_lstm"]}, Loss: {epoch_loss:.4f}')

            if cp_callback is not None:
                cp_callback(self, epoch_loss)

            self.eval()
            with torch.no_grad():
                test_outputs = self(self.x_test)
                test_loss = criterion(test_outputs, self.y_test)
                print(f'Test Loss: {test_loss:.4f}')

        print('Training finished.')


    def plot_reconstructed_lt_seq(self, idx_test, model_vae, data, device):
        with torch.no_grad():
            # VAE reconstruction
            input_signal = torch.zeros((self.l_seq, self.l_win, self.config['n_channel']), device=device)
            code_input = self.embedding_lstm_test[idx_test].unsqueeze(0).to(device)
            decoded_seq_vae = model_vae(input_signal, True, code_input)[0].squeeze().cpu().numpy()
            print(f"Decoded seq from VAE: {decoded_seq_vae.shape}")

            # LSTM reconstruction
            input_signal = torch.zeros((self.l_seq - 1, self.l_win, self.config['n_channel']), device=device)
            lstm_embedding_test = self.x_test[idx_test].unsqueeze(0).to(device)
            decoded_seq_lstm = model_vae(input_signal, True, lstm_embedding_test)[0].squeeze().cpu().numpy()
            print(f"Decoded seq from lstm: {decoded_seq_lstm.shape}")

        fig, axs = plt.subplots(self.config['n_channel'], 2, figsize=(15, 4.5 * self.config['n_channel']), edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()
        for j in range(self.config['n_channel']):
            for i in range(2):
                axs[i + j * 2].plot(np.arange(0, self.l_seq * self.l_win),
                                    np.reshape(data.val_set_lstm['data'][idx_test, :, :, j],
                                               (self.l_seq * self.l_win)))
                axs[i + j * 2].grid(True)
                axs[i + j * 2].set_xlim(0, self.l_seq * self.l_win)
                axs[i + j * 2].set_xlabel('samples')
            if self.config['n_channel'] == 1:
                axs[0 + j * 2].plot(np.arange(0, self.l_seq * self.l_win),
                                    np.reshape(decoded_seq_vae, (self.l_seq * self.l_win)), 'r--')
                axs[1 + j * 2].plot(np.arange(self.l_win, self.l_seq * self.l_win),
                                    np.reshape(decoded_seq_lstm, ((self.l_seq - 1) * self.l_win)), 'g--')
            else:
                axs[0 + j * 2].plot(np.arange(0, self.l_seq * self.l_win),
                                    np.reshape(decoded_seq_vae[:, :, j], (self.l_seq * self.l_win)), 'r--')
                axs[1 + j * 2].plot(np.arange(self.l_win, self.l_seq * self.l_win),
                                    np.reshape(decoded_seq_lstm[:, :, j], ((self.l_seq - 1) * self.l_win)), 'g--')
            axs[0 + j * 2].set_title(f'VAE reconstruction - channel {j}')
            axs[1 + j * 2].set_title(f'LSTM reconstruction - channel {j}')
            for i in range(2):
                axs[i + j * 2].legend(('ground truth', 'reconstruction'))
        plt.savefig(f"{self.config['result_dir']}lstm_long_seq_recons_{idx_test}.pdf")
        plt.close()


    def plot_lstm_embedding_prediction(self, idx_test, config, model_vae, data, device):
        self.plot_reconstructed_lt_seq(idx_test, config, model_vae, data, device)

        fig, axs = plt.subplots(2, self.code_size // 2, figsize=(15, 5.5), edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()
        for i in range(self.code_size):
            axs[i].plot(torch.arange(1, self.l_seq), self.embedding_lstm_test[idx_test, 1:, i].squeeze().cpu().numpy())
            axs[i].plot(torch.arange(1, self.l_seq), self.x_test[idx_test, :, i].squeeze().cpu().numpy())
            axs[i].set_xlim(1, self.l_seq - 1)
            axs[i].set_ylim(-2.5, 2.5)
            axs[i].grid(True)
            axs[i].set_title(f'Embedding dim {i}')
            axs[i].set_xlabel('windows')
            if i == self.code_size - 1:
                axs[i].legend(('VAE\nembedding', 'LSTM\nembedding'))
        plt.savefig(f"{config['result_dir']}lstm_seq_embedding_{idx_test}.pdf")
        plt.close()




######################################################
#                                                    #
#                   VAE Model                        #
#                                                    #
######################################################
        

class VAEmodel(BaseModel):
    def __init__(self, config):
        super(VAEmodel, self).__init__(config)
        self.input_dims = self.config['l_win'] * self.config['n_channel']
        self.config = config
        self.define_iterator()
        self.build_model()
        self.define_loss()
        self.training_variables()
        self.compute_gradients()
        self.init_saver()

    def define_iterator(self):
        self.original_signal = tf.placeholder(tf.float32, [None, self.config['l_win'], self.config['n_channel']])
        self.seed = tf.placeholder(tf.int64, shape=())
        self.dataset = tf.data.Dataset.from_tensor_slices(self.original_signal)
        self.dataset = self.dataset.shuffle(buffer_size=60000, seed=self.seed)
        self.dataset = self.dataset.repeat(8000)
        self.dataset = self.dataset.batch(self.config['batch_size'], drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()
        self.input_image = self.iterator.get_next()
        self.code_input = tf.placeholder(tf.float32, [None, self.config['code_size']])
        self.is_code_input = tf.placeholder(tf.bool)
        self.sigma2_offset = tf.constant(self.config['sigma2_offset'])



class VAEmodel(BaseModel):
    def __init__(self, config):
        super(VAEmodel, self).__init__(config)
        self.input_dims = self.config['l_win'] * self.config['n_channel']

        # Initialize encoder and decoder networks
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.original_signal = torch.tensor(self.original_signal, dtype=torch.float32)

        # Create PyTorch dataset
        dataset = TensorDataset(self.original_signal)

        # Create PyTorch DataLoader with shuffling and batching
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True,
        )

        # Convert sigma2_offset to PyTorch tensor
        self.sigma2_offset = torch.tensor(self.config['sigma2_offset'], dtype=torch.float32)

        # define sigma2 parameter to be trained to optimize ELBO        
        if self.config['TRAIN_sigma'] == 1:
            sigma = torch.nn.Parameter(torch.tensor(self.config['sigma'], dtype=torch.float32))
        
        else:
            sigma = torch.tensor(self.config['sigma'], dtype=torch.float32)
        
        self.sigma2 = torch.square(sigma)
        
        if self.config['TRAIN_sigma'] == 1:
            self.sigma2 = self.sigma2 + self.sigma2_offset

        print("sigma2: \n{}\n".format(self.sigma2))

    

    def calculate_loss(self, ):
        # KL divergence loss - analytical result
        KL_loss = 0.5 * (torch.sum(self.code_mean ** 2, dim=1)
                         + torch.sum(self.code_std_dev ** 2, dim=1)
                         - torch.sum(torch.log(self.code_std_dev ** 2), dim=1)
                         - self.config['code_size'])
        self.KL_loss = torch.mean(KL_loss)

        # norm 1 of standard deviation of the sample-wise encoder prediction
        self.std_dev_norm = torch.mean(self.code_std_dev, dim=0)

        weighted_reconstruction_error_dataset = torch.sum(
            (self.original_signal - self.decoded) ** 2, dim=[1, 2])
        weighted_reconstruction_error_dataset = torch.mean(weighted_reconstruction_error_dataset)
        self.weighted_reconstruction_error_dataset = weighted_reconstruction_error_dataset / (2 * self.sigma2)

        # least squared reconstruction error
        ls_reconstruction_error = torch.sum(
            (self.original_signal - self.decoded) ** 2, dim=[1, 2])
        self.ls_reconstruction_error = torch.mean(ls_reconstruction_error)

        # sigma regularisor - input elbo
        self.sigma_regularisor_dataset = self.input_dims / 2 * torch.log(self.sigma2)
        two_pi = self.input_dims / 2 * self.two_pi

        self.elbo_loss = two_pi + self.sigma_regularisor_dataset + \
                         0.5 * self.weighted_reconstruction_error_dataset + self.KL_loss



    def forward(self, x, is_code_input, code_input):
        """
        Performs the forward pass through the VAE model.

        :param x: Input time series window.
        :param is_code_input: Boolean indicating whether to use code_input or sample from the latent space.
        :param code_input: Optional code input for the decoder.

        Returns the decoded time series window, code mean, and code standard deviation.
        """
        if is_code_input:
            code_sample = code_input
            code_mean = None
            code_std_dev = None
        else:
            code_sample, code_mean, code_std_dev = self.encoder(x)

        decoded, sigma2 = self.decoder(code_input=code_input, code_sample=code_sample)

        return decoded, code_mean, code_std_dev, sigma2