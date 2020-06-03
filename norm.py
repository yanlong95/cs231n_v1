import data_loader_v5 as data_loader
import utils
import torch

path = 'dataset/small_region_data'
json_path = 'model/params.json'
params = utils.Params(json_path)
params.cuda = torch.cuda.is_available()
params.batch_size = 1000000

dataloaders = data_loader.fetch_dataloader(['train', 'test'], path, params)

train_dl = dataloaders['train']
val_dl = dataloaders['val']
test_dl = dataloaders['test']

for i, (image1, _, _) in enumerate(train_dl):
    avg = [torch.mean(image1[:,0,:,:]), torch.mean(image1[:,1,:,:]), torch.mean(image1[:,2,:,:])]
    std = [torch.std(image1[:,0,:,:]), torch.std(image1[:,1,:,:]), torch.std(image1[:,2,:,:])]
    print('Train Avg:', avg)
    print('Train Std:', std)

for i, (image1, _, _) in enumerate(val_dl):
    avg = [torch.mean(image1[:,0,:,:]), torch.mean(image1[:,1,:,:]), torch.mean(image1[:,2,:,:])]
    std = [torch.std(image1[:,0,:,:]), torch.std(image1[:,1,:,:]), torch.std(image1[:,2,:,:])]
    print('Val Avg:', avg)
    print('Val Std:', std)

for i, (image1, _, _) in enumerate(test_dl):
    avg = [torch.mean(image1[:,0,:,:]), torch.mean(image1[:,1,:,:]), torch.mean(image1[:,2,:,:])]
    std = [torch.std(image1[:,0,:,:]), torch.std(image1[:,1,:,:]), torch.std(image1[:,2,:,:])]
    print('Test Avg:', avg)
    print('Test Std:', std)
