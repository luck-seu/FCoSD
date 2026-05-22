import sys
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary

import warnings
warnings.filterwarnings('ignore')


sys.path.append('..')
from utils.log import print_log
from utils.metrics import RMSE_MSE_MAPE_MAE, MAE, MSE, RMSE, MAPE

class FCoSDLTSFRunner():
    def __init__(self, cfg:dict, device, scaler, log=None):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.scaler =scaler
        self.log = log

        self.clip_grad = cfg['OPTIM'].get('clip_grad')


    def train_one_epoch(self, model, train_loader, laplacian, optimizer, scheduler, criterion):

        model.train()

        batch_loss_list = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_batch = y_batch[..., 0]

            out_batch, cl, lol, mwl = model(x_batch, laplacian)
            loss = criterion(out_batch, y_batch) + self.cfg['MODEL_PARAM'].get('op_w')*lol + self.cfg['MODEL_PARAM'].get('op_w')*mwl
   
            batch_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg['OPTIM'].get('clip_grad'))
            optimizer.step()

        epoch_loss = np.mean(batch_loss_list)
        scheduler.step()

        return epoch_loss

    @torch.no_grad()
    def eval_model(self, model, val_loader, laplacian, criterion):

        model.eval()

        batch_loss_list = []
        for x_batch, y_batch in val_loader:

            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            y_batch = y_batch[..., 0]

            out_batch, cl, lol, mwl = model(x_batch, laplacian)
            loss = criterion(out_batch.detach().cpu(), y_batch.detach().cpu()) + self.cfg['MODEL_PARAM'].get('op_w')*lol
            
            batch_loss_list.append(loss.item())

        return np.mean(batch_loss_list)


    @torch.no_grad()
    def predict(self, model, loader, laplacian):

        model.eval()

        y = []
        out = []
        x_ave = []

        for x_batch, y_batch in loader:

            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            y_batch = y_batch[..., 0]

            # if x_batch.shape[0] != self.cfg['GENERAL'].get('batch_size'):
            #     # If the batch size is not equal to the expected batch size, skip this batch
            #     continue

            # out_batch = model(x_batch, laplacian)
            out_batch, cl, lol, mwl = model(x_batch, laplacian)
            
            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            
            # record average prediction
            cur_x = x_batch[..., 0]
            cur_x_ave = torch.mean(cur_x, dim=1, keepdim=True).cpu().numpy()
            x_ave.append(cur_x_ave)

            out.append(out_batch)
            y.append(y_batch)            

        # (samples, out_steps, num_nodes, output_dim)
        out = np.vstack(out)  
        y = np.vstack(y)

        return y, out
    
    @torch.no_grad()
    def predict_order(self, model, loader, laplacian):

        model.eval()

        y = []
        out = []
        x_ave = []

        for x_batch, y_batch in loader:

            x_batch = x_batch.float().to(self.device) # b, T, N, d
            y_batch = y_batch.float().to(self.device) # b, T, N, d
            y_batch = y_batch[..., 0] # b, T, N, 1

            # if x_batch.shape[0] != self.cfg['GENERAL'].get('batch_size'):
            #     # If the batch size is not equal to the expected batch size, skip this batch
            #     continue

            # out_batch = model(x_batch, laplacian)

            # y_batch shape: (samples, out_steps, num_nodes)
            # Randomly permute x_batch nodes and apply the same order to y_batch.
            num_nodes = y_batch.shape[2]
            # print(x_batch.shape, y_batch.shape)
            perm = np.random.permutation(num_nodes)
            y_batch = y_batch[:, :, perm]
            x_batch = x_batch[:, :, perm, :]
            
            # L_perm = laplacian[perm][:, perm]

            out_batch, cl, lol, mwl = model(x_batch, laplacian, perm)
            
            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            
            # record average prediction
            cur_x = x_batch[..., 0]
            cur_x_ave = torch.mean(cur_x, dim=1, keepdim=True).cpu().numpy()
            x_ave.append(cur_x_ave)

            out.append(out_batch)
            y.append(y_batch)            

        # (samples, out_steps, num_nodes, output_dim)
        out = np.vstack(out)  
        y = np.vstack(y)

        return y, out

    def train(
        self,
        model,
        train_loader,
        val_loader,
        laplacian,
        optimizer,
        scheduler,
        criterion,
        max_epochs=10,
        early_stop_patience=3,
        compile_model=False,
        verbose=1,
        save=None):

        if torch.__version__ >= '2.0.0' and compile_model:
            model = torch.compile(model)

        wait = 0
        min_val_loss = np.inf
        best_epoch = 0

        train_loss_list = []
        val_loss_list = []

        # Create a temporary file path for saving the best model.
        import tempfile
        import os
        temp_checkpoint = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt')
        temp_checkpoint_path = temp_checkpoint.name
        temp_checkpoint.close()

        start = time.time()
        for epoch in range(max_epochs):
            train_loss = self.train_one_epoch(
                model, train_loader, laplacian, optimizer, scheduler, criterion
            )
            train_loss_list.append(train_loss)

            val_loss = self.eval_model(model, val_loader, laplacian, criterion)
            val_loss_list.append(val_loss)

            if (epoch + 1) % verbose == 0:
                print_log(
                    datetime.datetime.now(),
                    'Epoch',
                    epoch + 1,
                    ' \tTrain Loss = %.5f' % train_loss,
                    'Val Loss = %.5f' % val_loss,
                    log=self.log,
                )

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                # Save directly to the temporary file instead of deep-copying in memory.
                torch.save(model.state_dict(), temp_checkpoint_path)
            else:
                wait += 1
                if wait >= early_stop_patience:
                    break
        end = time.time()

        # Load the best model from the temporary file.
        model.load_state_dict(torch.load(temp_checkpoint_path))
        # Delete the temporary file.
        os.unlink(temp_checkpoint_path)

        if save:
            torch.save(model.state_dict(), save)  

        train_rmse, train_mse, train_mape, train_mae = RMSE_MSE_MAPE_MAE(*self.predict(model, train_loader, laplacian))
        val_rmse, val_mse, val_mape, val_mae = RMSE_MSE_MAPE_MAE(*self.predict(model, val_loader, laplacian))

        out_str = f'Finish at epoch: {epoch+1}\n'
        out_str += f'Best model at epoch {best_epoch+1}:\n'
        out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
        out_str += "Train RMSE = %.5f, MSE = %.5f, MAPE = %.5f, MAE = %.5f\n" % (
            train_rmse,
            train_mse,
            train_mape,
            train_mae,
        )
        out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
        out_str += "Val RMSE = %.5f, MSE = %.5f, MAPE = %.5f, MAE = %.5f" % (
            val_rmse,
            val_mse,
            val_mape,
            val_mae,
        )
        print_log(out_str, log=self.log)
        print_log("Traing time per epoch: %.3f s" % ((end - start)/epoch), log=self.log)

        return model


    @torch.no_grad()
    def test_model(self, model, test_loader, laplacian):
        
        model.eval()

        print_log('--------- Test ---------', log=self.log)
        
        
        y_true, y_pred = self.predict(model, test_loader, laplacian)

        out_steps = y_pred.shape[1]

        steps = [12, 24, 48, 96]
        # steps = [1, 2, 4, 8]
        mae_met = 0
        mse_met = 0
        rmse_met = 0
        mape_met = 0

        for i in range(out_steps):
            mae_met = mae_met + MAE(y_true[:, i, ...], y_pred[:, i, ...])
            mse_met = mse_met + MSE(y_true[:, i, ...], y_pred[:, i, ...])
            rmse_met = rmse_met + RMSE(y_true[:, i, ...], y_pred[:, i, ...])
            mape_met = mape_met + MAPE(y_true[:, i, ...], y_pred[:, i, ...])

            if (i + 1) in steps:
                step = i + 1
                out_str = "Step %d: RMSE = %.5f, MSE = %.5f, MAPE = %.5f, MAE = %.5f" % (
                    step,
                    rmse_met/step,
                    mse_met/step,
                    mape_met/step,
                    mae_met/step
                )
                print_log(out_str, log=self.log)