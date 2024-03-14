import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, savefig
import time



######################################################
#                                                    #
#                   BASE MODEL                       #
#                                                    #
######################################################


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config      = config
        self.two_pi      = torch.tensor(2 * np.pi)
        self.input_dims  = None
        self.global_step = 0
        self.cur_epoch   = 0

        self.encoder = None
        self.decoder = None
        self.sigma2  = None

        self.train_vars_VAE = self.training_variables()

        self.lr        = 1 ###CHECK
        self.optimizer = torch.optim.Adam(self.train_vars_VAE, lr=self.lr, betas=(0.9, 0.95))
        

    def save(self, checkpoint_path):
        print("Saving model...")
        torch.save(self.state_dict(), checkpoint_path)
        print("Model saved.")

    
    def load(self, checkpoint_path):
        print("checkpoint_dir at loading: {}".format(checkpoint_path))
        if os.path.exists(checkpoint_path):
            print("Loading model checkpoint {} ...\n".format(checkpoint_path))
            self.load_state_dict(torch.load(checkpoint_path))
            print("Model loaded.")
        else:
            print("No model loaded.")

    
    def calculate_loss(self, original_signal, decoded, code_mean, code_std_dev):
        # KL divergence loss - analytical result
        KL_loss = 0.5 * (torch.sum(code_mean ** 2, dim=1)
                         + torch.sum(code_std_dev ** 2, dim=1)
                         - torch.sum(torch.log(code_std_dev ** 2), dim=1)
                         - self.config['code_size'])
        KL_loss = torch.mean(KL_loss)

        # norm 1 of standard deviation of the sample-wise encoder prediction
        std_dev_norm = torch.mean(code_std_dev, dim=0) # Check here Later !!!!

        weighted_reconstruction_error_dataset = torch.sum(
            (original_signal - decoded) ** 2, dim=[1, 2])
        weighted_reconstruction_error_dataset = torch.mean(weighted_reconstruction_error_dataset)
        weighted_reconstruction_error_dataset = weighted_reconstruction_error_dataset / (2 * self.sigma2)

        # least squared reconstruction error
        ls_reconstruction_error = torch.sum(
            (original_signal - decoded) ** 2, dim=[1, 2])
        ls_reconstruction_error = torch.mean(ls_reconstruction_error)

        # sigma regularisor - input elbo
        sigma_regularisor_dataset = self.input_dims / 2 * torch.log(self.sigma2)
        two_pi = self.input_dims / 2 * self.two_pi

        elbo_loss = two_pi + sigma_regularisor_dataset + \
                         0.5 * weighted_reconstruction_error_dataset + KL_loss
        
        return elbo_loss

    
    def training_variables(self):
        encoder_vars = list(self.encoder.parameters())
        decoder_vars = list(self.decoder.parameters())
        sigma_vars = [self.sigma2]
        train_vars_VAE = encoder_vars + decoder_vars + sigma_vars

        num_encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        num_decoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        num_sigma2 = self.sigma2.numel()
        
        num_vars_total = num_decoder + num_encoder + num_sigma2
        
        print("Total number of trainable parameters in the VAE network is: {}".format(num_vars_total))
        
        return train_vars_VAE

    
    def compute_gradients(self):
        
        self.train_step_gradient = self.optimizer.step()
        print("Reach the definition of loss for VAE")

    
    def clip_grad(self, grad):
        if grad is None:
            return grad
        return torch.clamp(grad, -1, 1)



######################################################
#                                                    #
#                   BASE TRAIN                       #
#                                                    #
######################################################




class BaseTrain:
    def __init__(self, model, data, config):
        """
        Initializes the training session with the necessary components.

        :param model: The model to be trained.
        :param data: The dataset for training and validation.
        :param config: Configuration dictionary with training parameters and paths.

        Initializes PyTorch variables, prepares for training, and initializes
        various lists to keep track of training and validation losses, as well as
        other metrics like reconstruction and KL divergence losses.
        """
        self.model = model
        self.config = config
        self.data = data

        # Initialize lists to keep track of various training and validation metrics
        self.train_loss = []
        self.val_loss = []
        self.train_loss_ave_epoch = []
        self.val_loss_ave_epoch = []
        self.recons_loss_train = []
        self.recons_loss_val = []
        self.KL_loss_train = []
        self.KL_loss_val = []
        self.sample_std_dev_train = []
        self.sample_std_dev_val = []
        self.iter_epochs_list = []
        self.test_sigma2 = []

    def train(self):
        """
        Starts the training process over a specified number of epochs.

        Tracks and prints the elapsed time and estimated remaining time for training.
        Utilizes the `train_epoch` method (not defined in this snippet) for actual
        training logic specific to each epoch.
        """
        self.start_time = time.time()
        for cur_epoch in range(self.config['num_epochs_vae']):
            self.train_epoch()  # Train for one epoch, method to be defined elsewhere

            # Calculate and print time elapsed and estimated remaining training time
            self.current_time = time.time()
            elapsed_time = (self.current_time - self.start_time) / 60
            est_remaining_time = (self.current_time - self.start_time) / (cur_epoch + 1) * (self.config['num_epochs_vae'] - cur_epoch - 1) / 60
            print("Already trained for {} min; Remaining {} min.".format(elapsed_time, est_remaining_time))

            # Increment the current epoch counter in the model
            self.model.cur_epoch += 1

    def save_variables_VAE(self):
        """
        Saves important variables and metrics for inspection after training.

        Saves metrics such as training and validation losses, reconstruction and KL
        divergence losses, and model parameters to a file.
        """
        # Construct the filename from configuration parameters
        file_name = "{}{}-batch-{}-epoch-{}-code-{}-lr-{}.npz".format(self.config['result_dir'],
                                                                      self.config['exp_name'],
                                                                      self.config['batch_size'],
                                                                      self.config['num_epochs_vae'],
                                                                      self.config['code_size'],
                                                                      self.config['learning_rate_vae'])
        # Save metrics to the specified file
        np.savez(file_name,
                 iter_list_val=self.iter_epochs_list,
                 train_loss=self.train_loss,
                 val_loss=self.val_loss,
                 n_train_iter=self.n_train_iter,
                 n_val_iter=self.n_val_iter,
                 recons_loss_train=self.recons_loss_train,
                 recons_loss_val=self.recons_loss_val,
                 KL_loss_train=self.KL_loss_train,
                 KL_loss_val=self.KL_loss_val,
                 num_para_all=sum(p.numel() for p in self.model.parameters()),
                 sigma2=self.test_sigma2)

    def plot_train_and_val_loss(self):
        """
        Plots training and validation loss over time, as well as a breakdown of the
        validation loss into its components and the evolution of sigma^2.

        Generates three plots: overall training and validation losses, validation
        loss components (reconstruction and KL divergence losses), and sigma^2 values
        over the course of training.
        """
        # Plot overall training and validation losses
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plot(self.train_loss, 'b-')
        plot(self.iter_epochs_list, self.val_loss_ave_epoch, 'r-')
        plt.legend(('Training Loss (Total)', 'Validation Loss'))
        plt.title('Training Loss Over Iterations (Validation @ Epochs)')
        plt.ylabel('Total Loss')
        plt.xlabel('Iterations')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/loss.png')

        # Plot breakdown of validation loss into reconstruction and KL divergence losses
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plot(self.recons_loss_val, 'b-')
        plot(self.KL_loss_val, 'r-')
        plt.legend(('Reconstruction Loss', 'KL Divergence Loss'))
        plt.title('Validation Loss Breakdown')
        plt.ylabel('Loss')
        plt.xlabel('Number of Batches')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/val-loss.png')

        # Plot the evolution of sigma^2 over training
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plot(self.test_sigma2, 'b-')
        plt.title('Sigma^2 Over Training')
        plt.ylabel('Sigma^2')
        plt.xlabel('Iteration')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/sigma2.png')