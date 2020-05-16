import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

"""
args: path
"""

# train transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4793, 0.4921, 0.4731), (0.0670, 0.0837, 0.1140))])

# evl and test transformer
eval_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.4789, 0.4905, 0.4740), (0.2007, 0.2004, 0.2277))])

class BuildingDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.images = []
        self.labels = []
        self.root = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.root]

        for i, file in enumerate(self.filenames):
            for image in os.listdir(file):
                src = os.path.join(file, image)
                self.images.append(src)
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        return image, self.labels[idx]


# load a train, val, text in mini-batch size
def fetch_dataloader(types, data_dir, params, device='cpu'):
    dataloaders = {}

    for split in ['Building_labeled_train_data', 'Building_labeled_val_data', 'Building_labeled_test_data']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'Building_labeled_train_data':
                dl = DataLoader(BuildingDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers, pin_memory=params.cuda)
            else:
                dl = DataLoader(BuildingDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                        num_workers=params.num_workers, pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
