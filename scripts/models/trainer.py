from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch

def train(num_epochs, model: nn.Module,\
 loss_func, optimizer,\
 train_loader, val_loader,\
 val_logg_freq,\
 metric_funtions\
 ): 
  '''
    Train a model for a given number of epochs.

    ## Params:
      * `metric_funtions`: A list of functions from [f1_score, recall_score, precision_score, accuracy_score]
  '''
  train_losses = []
  val_losses = []
  metrics = []
  logg_freq = range(0,num_epochs,val_logg_freq)

  # Train through epoches
  for epoch in range(num_epochs):
    model.train()
    avg_lost_batch = 0
    for batch_x, batch_y in train_loader:      
      # Stack the batches to one tensor and wrap them to Variables      
      x_train = Variable(batch_x)
      y_train = Variable(batch_y)     
      
      # Tranfer data to cuda if present
      if torch.cuda.is_available():
        x_train, y_train = x_train.cuda(), y_train.cuda()

      # Clear the gradients and compute the updated weights
      optimizer.zero_grad()

      # predict      
      pred_out = model(x_train)      
            
      # find loss      
      loss_train = loss_func(pred_out, y_train)    
      loss_train.backward()
      optimizer.step()
      avg_lost_batch += loss_train.item()      
    
    # Store loss
    train_losses.append(avg_lost_batch/len(train_loader))

    # logging prgoress and validating 
    if epoch in logg_freq or epoch == num_epochs-1:      
      print(f'Epoch: {epoch:3d}/{num_epochs} train_loss: {train_losses[-1]:.2f}', end='')

      # Test the model
      if val_loader != None:
        model.eval()
        avg_loss_val = 0
        metrics_avg = np.zeros(len(metric_funtions))
        with torch.no_grad():
          for batch_x, batch_y in val_loader: # Batch size 1
            x_test, y_test = Variable(batch_x), Variable(batch_y)
            
            # Tranfer data to cuda if present
            if torch.cuda.is_available():
              x_test, y_test = x_test.cuda(), y_test.cuda()

            # predict & softmax
            pred_out = model(x_test)            
            soft = nn.Softmax(dim=1)
            pred_out = soft(pred_out.cpu().detach())
            y_test = y_test.cpu().detach()       

            # find loss and f1          
            avg_loss_val += loss_func(pred_out, y_test)            
            for idx, func in enumerate(metric_funtions):
              metrics_avg[idx] += func(y_test.flatten(), (pred_out[:,1]>0.5).int().flatten())

          # Store loss
          val_losses.append(avg_loss_val/len(val_loader))
          metrics.append(metrics_avg/len(val_loader))
          
        # Create metrics string
        met_str = ' '.join([f'{metric_funtions[i].__name__}={metrics_avg[i]/len(val_loader):.2f}' for i in range(len(metric_funtions))] )
        print(f', val_loss:{avg_loss_val/len(val_loader):.2f},{met_str}')

      else:
        print() # For the '\n'
          

  return train_losses, val_losses, metrics