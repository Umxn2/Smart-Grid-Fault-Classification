import pandas as pd

# Load the datasets
test_data = pd.read_csv('./Dataset/test_dataset_with_fault_type.csv')
train_data = pd.read_csv('./Dataset/train_dataset.csv')
updated_test_data = test_data[~test_data.apply(tuple, axis=1).isin(train_data.apply(tuple, axis=1))]
updated_test_data.to_csv('./Dataset/test_dataset_with_fault_type.csv', index=False)
