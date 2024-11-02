import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class LoggingDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        # Data Preprocessing
        # Step 1: Handling Missing Values
        for column in ["PE", "GR", "AC", "CNL", "DEN"]:
            missing_percentage = data[column].isna().mean()
            if missing_percentage < 0.05:
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                data.drop(column, axis=1, inplace=True)

        # Step 2: Outlier Detection and Handling
        for column in ["PE", "GR", "AC", "CNL", "DEN"]:
            mean = data[column].mean()
            std = data[column].std()
            outliers = (data[column] - mean).abs() > 3 * std
            data.loc[outliers, column] = data[column].median()

        # Step 3: Feature Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[["PE", "GR", "AC", "CNL", "DEN"]])
        self.features = scaled_features
        self.labels = data["classification"].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def prepare_data_loader(file_path, batch_size=32):
    dataset = LoggingDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
