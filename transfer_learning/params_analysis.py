import pandas as pd

def save_params(name, params, metrics, cm=None,   elapsed_time=None, df=None, path=None):
  if df is None:
    df = pd.DataFrame(index=['accuracy', 'batch_size', 'bottleneck', 'check_points_path', 'cm',
       'cut_layer', 'data_augmentation', 'decay_rate', 'decay_steps', 'elapsed_time', 'f1_score', 
       'fine_tunning', 'hidden_layers', 'image_channels', 'image_size', 'initial_learning_rate', 
       'keep', 'loss[train]', 'loss[validation]', 'num_classes', 'num_epochs', 'precision', 'recall'])
   
  df[name] = pd.Series({**vars(params), **metrics})
  
  if cm:
    df.at['cm', name] = cm
    
  if elapsed_time:
    df.at['elapsed_time', name] = elapsed_time
    
  if path:
    df.to_json(path)
    
  return df  

def load_df(path):
  try:
    df = pd.read_json(path)
    
  except:
    open(path, 'w').close() 
    df = pd.DataFrame(index=['accuracy', 'batch_size', 'bottleneck', 'check_points_path', 'cm',
       'cut_layer', 'data_augmentation', 'decay_rate', 'decay_steps', 'elapsed_time', 'f1_score', 
       'fine_tunning', 'hidden_layers', 'image_channels', 'image_size', 'initial_learning_rate', 
       'keep', 'loss[train]', 'loss[validation]', 'num_classes', 'num_epochs', 'precision', 'recall'])
  
  return df
