import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image



"""
args: path
"""

# train transformer
dict = {'apartment': 0, 'church': 1, 'garage': 2, 'house': 3, 'industrial': 4, 'officebuilding': 5, 'retail': 6,
        'roof': 7}

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# evl and test transformer
eval_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()])

class BuildingDataset(Dataset):
    def __init__(self, data_dir, dict, transform):
        self.root = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.root]
        self.filenames = [os.listdir(filename) for filename in self.filenames]
        self.labels = [len(filename)*[dict[self.root[i]]] for i, filename in enumerate(self.filenames)]

        self.filenames = [j for filename in self.filenames for j in filename]
        self.labels = [j for label in self.labels for j in label]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]


# load a train, val, text in mini-batch size
def fetch_dataloader(types, data_dir, params, device='cpu'):
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(BuildingDataset(path, dict, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers, pin_memory=params.cuda)
            else:
                dl = DataLoader(BuildingDataset(path, dict, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                        num_workers=params.num_workers, pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

