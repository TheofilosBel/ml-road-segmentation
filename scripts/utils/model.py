''' Saving & loading Pytorch models and optimizers util '''
import torch
import torch.nn as nn
import torch.optim as opt
import os
from typing import Dict


def save(model: nn.Module, optimizer, name_specifier: str=None, path: str='./cache/', **opt_args):
  ''' Save the model's state to the directory specified by `path`

      ## Parameters:
      * model: the model
      * optimizer: the optimizer
      * path: the directory to place the saved model
      * opt_args: other optional args you would like to save
  '''
  # model related dictionary
  model_dict = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }
  
  # save the device of the network amon with the other info  
  if torch.cuda.is_available():
    model_dict['model_device'] = 'cuda'
  else:
    model_dict['model_device'] = 'cpu'
      
  # Concatanate it with opt_args if any
  if len(opt_args) != 0:
    model_dict = dict(model_dict, **opt_args)

  # Create the name of the file based on the model + optimizer
  if name_specifier != None:
    file_name = model.__class__.__name__ + '_' + optimizer.__class__.__name__ + '_' + name_specifier + '.pt'
  else:
    file_name = model.__class__.__name__ + '_' + optimizer.__class__.__name__ + '.pt'

  # Save the model
  torch.save(model_dict, os.path.join(path, file_name))  


def load(model: nn.Module, optimizer=None, name_specifier: str=None, path: str='./cache/') -> Dict:
  ''' Load the model's state from the directory specified by `path`
      The load function will complain if model was saved using other
      otimizers or it was a different type o model

      ## Parameters:
       * model: the model
       * optimizer: the optimizer
       * path: the directory to place the saved model
       * opt_args: other optional args you would like to save 

      # Returns:
       * A dictionarry with other stats contained to the checkpoint loaded, 
         If not existing return None
  '''
  # Create possible model name
  saved_models = os.listdir(path)
  specifiers = [model.__class__.__name__]
  if optimizer != None:
    specifiers.append(optimizer.__class__.__name__)
  if name_specifier != None:        
    specifiers.append(name_specifier)
 
  model_file_name = None
  for model_name in saved_models:
    found_all = 0
    for spec in specifiers:    
      if spec in model_name:
        found_all+=1
    if found_all == len(specifiers):
      model_file_name = model_name
  
  # if not found then return and complain
  if model_file_name == None:
    raise Exception(f'Model with specifiers:"{specifiers}" not found')
  else:    
    # Load checkpoint on cpu or gpu
    if torch.cuda.is_available():
      checkpoint = torch.load(os.path.join(path, model_file_name))
    else:
      checkpoint = torch.load(os.path.join(path, model_file_name), map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer != None:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    dev = checkpoint['model_device']
    
    # To cuda if vailable    
    if dev in 'cuda':      
      if torch.cuda.is_available():
        model.to(torch.device("cuda"))        
      else:
        print("[INF] Model was saved in cuda mode, but now it's loaded in cpu")

    # Return a dict with other stats if they exist
    checkpoint.pop('model_state_dict')
    checkpoint.pop('optimizer_state_dict')
    if len(checkpoint) != 0:
      return checkpoint
    else:
      return None


