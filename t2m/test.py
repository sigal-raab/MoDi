from t2m.motion_loaders import dataset_motion_loader

dataset_motion_loader.get_dataset_motion_loader('', 8, 'cpu')
for i, data in enumerate(dataset_motion_loader.dataloader):
    print(data)
    break