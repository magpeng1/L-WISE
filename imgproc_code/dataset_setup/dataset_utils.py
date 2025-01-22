
def assign_split(group, proportion_val, proportion_test):
  """
  Assign train/val/test splits to rows of a dataframe. 
  To get even splits per class, call this function using something like: 
  data_df = data_df.groupby('class', group_keys=False).apply(lambda x: assign_split(x, proportion_val, proportion_test))
  """

  # Shuffle the group to ensure random sampling
  shuffled = group.sample(frac=1).reset_index(drop=True)
  size = len(shuffled)
  
  # Calculate the count for each split
  val_count = int(size * proportion_val)
  test_count = int(size * proportion_test)
  train_count = size - (val_count + test_count)  # Ensure all samples are accounted for
  
  # Assign the split values
  shuffled['split'] = ['train'] * train_count + ['val'] * val_count + ['test'] * test_count
  
  # Shuffle again to mix the split assignments before returning
  return shuffled.sample(frac=1).reset_index(drop=True)