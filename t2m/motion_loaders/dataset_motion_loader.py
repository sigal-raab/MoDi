from t2m.data.dataset import Text2MotionDatasetV2, collate_fn, MoDiDataset
from t2m.t2m_utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from t2m.t2m_utils.get_opt import get_opt

def get_dataset_motion_loader(opt_path, batch_size, device):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./t2m/glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test.txt')
        dataset = Text2MotionDatasetV2(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

if __name__ == '__main__':
    dataloader, dataset = get_dataset_motion_loader('config/kit.yaml', 8, 'cpu')
    for i, data in enumerate(dataloader):
        print(data)
        break

def get_dataset_modi_motion_loader(opt_path, batch_size, device, modi_folder_path):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./t2m/glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test.txt')
        dataset = MoDiDataset(opt, mean, std, split_file, w_vectorizer, modi_folder_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

if __name__ == '__main__':
    dataloader, dataset = get_dataset_motion_loader('config/kit.yaml', 8, 'cpu')
    for i, data in enumerate(dataloader):
        print(data)
        break