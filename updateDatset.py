import pandas as pd
test_file_path = './Dataset/test_dataset_with_fault_type.csv'
train_file_path = './Dataset/train_dataset.csv'
test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)
test_data_sample = test_data.sample(frac=0.5, random_state=42)
updated_train_data = pd.concat([train_data, test_data_sample], ignore_index=True)
# Save the updated train dataset to a new file
updated_train_file_path = './Dataset/train_dataset.csv'
updated_train_data.to_csv(updated_train_file_path, index=False)


