import os
import pickle
from natsort import natsorted
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    def __init__(self, images, targets, transform=None, filtered=False):
        if filtered:
            valid_indices = [i for i, t in enumerate(targets) if torch.any(torch.tensor(t) != 0.)]
            self.images = [images[i] for i in valid_indices]
            self.targets = [targets[i] for i in valid_indices]
        else:
            self.images = images
            self.targets = targets

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)
        return image, target

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
])

IMAGES_PATH = 'data/BrainMRISegmentation/kaggle_3m/'
DATAPATH = 'data/2D_segmentation_dataset'
SANITY_DATAPATH = 'data/2D_segmentation_sanity_dataset'
SMALL_DATAPATH = 'data/2D_segmentation_small_dataset'
FILTERED_DATAPATH = 'data/2D_segmentation_filtered_dataset'

def load_dataset():
    if not os.path.exists(DATAPATH):
        raise Exception('Datasets are missing. Run create_dataset_files function.')
    else:
        print('Loading 2D segmentation dataset...')
        data_file = open(DATAPATH, 'rb')
        data = pickle.load(data_file)
        data_file.close()
        return data

def load_filtered_dataset():
    if not os.path.exists(FILTERED_DATAPATH):
        raise Exception('Datasets are missing. Run create_dataset_files function.')
    else:
        print('Loading filtered 2D segmentation dataset...')
        filtered_file = open(FILTERED_DATAPATH, 'rb')
        data = pickle.load(filtered_file)
        filtered_file.close()
        return data


def load_sanity_dataset():
    if not os.path.exists(SANITY_DATAPATH):
        raise Exception('Datasets are missing. Run create_dataset_files function.')
    else:
        print('Loading 2D sanity dataset...')
        sanity_file = open(SANITY_DATAPATH, 'rb')
        sanity_data = pickle.load(sanity_file)
        sanity_file.close()
        return sanity_data


def load_small_dataset():
    if not os.path.exists(SMALL_DATAPATH):
        raise Exception('Datasets are missing. Run create_dataset_files function.')
    else:
        print('Loading 2D small dataset...')
        small_file = open(SMALL_DATAPATH, 'rb')
        small_data = pickle.load(small_file)
        small_file.close()
        return small_data


def load_train_test_split(test_size=0.2, random_state=42):
    dataset = load_dataset()

    train_images, test_images, train_targets, test_targets = train_test_split(
        dataset.images,
        dataset.targets,
        test_size=test_size,
        random_state=random_state
    )

    train_dataset = SegmentationDataset(train_images, train_targets, transform)
    test_dataset = SegmentationDataset(test_images, test_targets, transform)

    return train_dataset, test_dataset


def load_filtered_train_test_split(test_size=0.2, random_state=42):
    dataset = load_filtered_dataset()

    train_images, test_images, train_targets, test_targets = train_test_split(
        dataset.images,
        dataset.targets,
        test_size=test_size,
        random_state=random_state
    )

    train_dataset = SegmentationDataset(train_images, train_targets, transform)
    test_dataset = SegmentationDataset(test_images, test_targets, transform)

    return train_dataset, test_dataset


def load_small_train_test_split(test_size=0.2, random_state=42):
    dataset = load_small_dataset()

    train_images, test_images, train_targets, test_targets = train_test_split(
        dataset.images,
        dataset.targets,
        test_size=test_size,
        random_state=random_state
    )

    train_dataset = SegmentationDataset(train_images, train_targets, transform)
    test_dataset = SegmentationDataset(test_images, test_targets, transform)

    return train_dataset, test_dataset

def create_dataset_files():
    if not os.path.exists(IMAGES_PATH):
        raise Exception('No images found. Download link https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation.')

    print('Iterating over directories containing images...')
    images = []
    targets = []
    for patient in os.listdir(IMAGES_PATH):
        patient_path = os.path.join(IMAGES_PATH, patient)
        for img in natsorted(os.listdir(patient_path)):
            im = plt.imread(os.path.join(patient_path, img))
            if im.ndim == 3:
                images.append(im)
            elif im.ndim == 2:
                targets.append(im)
            else:
                raise Exception()

    print('Creating dataobjects...')
    # Full dataset
    data = SegmentationDataset(
        images,
        targets,
        transform)

    # Filtered dataset
    filtered_data = SegmentationDataset(
        images,
        targets,
        transform,
        filtered=True)

    # Arbitrary image to use in sanity test
    sanity_data = SegmentationDataset(
        [images[147]],
        [targets[147]],
        transform)
    
    # Small arbitrary training set to test that training works
    small_data = SegmentationDataset(
        images[147:157],
        targets[147:157],
        transform)

    print('Dumping objects to files...')
    data_file = open(DATAPATH, 'wb')
    sanity_file = open(SANITY_DATAPATH, 'wb')
    small_file = open(SMALL_DATAPATH, 'wb')
    filtered_file = open(FILTERED_DATAPATH, 'wb')

    pickle.dump(data, data_file)
    pickle.dump(sanity_data, sanity_file)
    pickle.dump(small_data, small_file)
    pickle.dump(filtered_data, filtered_file)

    data_file.close()
    sanity_file.close()
    small_file.close()
    filtered_file.close()
    print('Done')