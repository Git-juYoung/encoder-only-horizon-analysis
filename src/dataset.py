import torch
from torch.utils.data import Dataset


class ElectricityDataset(Dataset):
    def __init__(self, data, input_length, horizon, stride):

        self.data = data
        self.input_length = input_length
        self.horizon = horizon
        self.stride = stride

        self.samples = []
        self._build_windows()

    def _build_windows(self):
        T, num_households = self.data.shape

        for h_id in range(num_households):
            series = self.data[:, h_id]

            for start in range(0, T - self.input_length - self.horizon + 1, self.stride):
                end_input = start + self.input_length
                end_target = end_input + self.horizon

                x = series[start:end_input]
                y = series[end_input:end_target]

                self.samples.append((x, y, h_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, h_id = self.samples[index]

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y, h_id
