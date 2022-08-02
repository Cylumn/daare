from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler


class AKRDataset(Dataset):
    def __init__(self, simulate, n_dataset, brush_range=(None, None), scale: bool = True):
        self.img_true = []
        self.img_perturbed = []
        self.simulate = simulate
        brushes = simulate.read_brushes('data', False)[brush_range[0]:brush_range[1]]

        for i in range(n_dataset):
            y = simulate.ground_truth(brushes)
            x = simulate.noise(simulate.noise(y))

            if scale:
                sc = StandardScaler()
                shape = y.shape
                x = torch.tensor(sc.fit_transform(x.reshape(-1, 1)).reshape(-1, shape[0], shape[1]) * 0.2 + 0.2).float()
                y = torch.tensor(sc.transform(y.reshape(-1, 1)).reshape(-1, shape[0], shape[1]) * 0.2 + 0.2).float()
            else:
                x = torch.tensor(x[None, :, :]).float()
                y = torch.tensor(y[None, :, :]).float()

            self.img_true.append(y)
            self.img_perturbed.append(x)

    def __len__(self):
        return len(self.img_true)

    def __getitem__(self, idx):
        return self.img_perturbed[idx], self.img_true[idx]