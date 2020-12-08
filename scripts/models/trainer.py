from sklearn.metrics import f1_score
from torch.autograd import Variable
import torch.nn as nn
import torch

def train(num_epochs, model: nn.Module, loss_func, optimizer, train_loader, val_loader):
  train_losses = []
  test_losses = []
  model.train()

  # Train through epoches
  for epoch in range(num_epochs):
    model.train()
    avg_lost_batch = 0
    for batch_x, batch_y in train_loader:      
      # Stack the batches to one tensor and wrap them to Variables
      x_train = Variable( torch.cat( [a for a in batch_x ]) )
      y_train = Variable( torch.cat( [a for a in batch_y ]) )     
      
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
    if epoch % 10 == 9 or epoch == 0:
      print(f'Epoch: {epoch+1:3d} train_loss: {avg_lost_batch/len(train_loader):.2f}',  end='')

      # Test the model
      model.eval()
      loss_test = 0
      metric = 0
      with torch.no_grad():          
        for batch_x, batch_y in val_loader: # Batch size 1
          x_test, y_test = Variable(batch_x), Variable(batch_y)
          
          # Tranfer data to cuda if present
          if torch.cuda.is_available():
            x_test, y_test = x_test.cuda().squeeze(0), y_test.cuda().squeeze(0)

          # predict & softmax
          pred_out = model(x_test)
          soft = nn.Softmax(dim=1)

          pred_out = soft( pred_out.cpu().detach() )
          y_test = y_test.cpu().detach()          

          # find loss and f1
          loss_test += loss_func(pred_out, y_test)
          metric += f1_score(y_test, pred_out[:,1]>0.5 )

        # Store loss
        test_losses.append(loss_test/len(val_loader))

      print(f', test_loss:{loss_test/len(val_loader):.2f}, {f1_score.__name__}:{metric/len(val_loader):.2f}')
          

  return train_losses