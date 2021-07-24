
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from skimage.transform import rescale, resize
import SimpleITK as sitk
import os
import numpy as np


class CamusIterator(Dataset):
    def __init__(
            self,
            data_type='train',
            global_transforms=[],
            augment_transforms=[]
    ):
        super(CamusIterator, self).__init__()

        train_file = r'D:\Projects\Heart\camus\training'
        test_file = r'D:\Projects\Heart\camus\testing'

        if data_type == 'train':
            data_file = train_file
        elif data_type == 'test':
            data_file = test_file
        else:
            raise Exception('Wrong data_type for CamusIterator')

        self.data_type = data_type
        self.data_file = data_file
        self.global_transforms = global_transforms
        self.augment_transforms = augment_transforms

    def __read_image(self, patient_file, suffix):
        image_file = '{}/{}/{}'.format(self.data_file, patient_file, patient_file + suffix)
        # Stolen from a StackOverflow answer
        # https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_file, sitk.sitkFloat32))
        return image

    def __read_info(self, data_file):
        info = {}
        with open(data_file, 'r') as f:
            for line in f.readlines():
                info_type, info_details = line.strip('\n').split(': ')
                info[info_type] = info_details
        return info

    def __len__(self):
        return len(os.listdir(self.data_file))

    def __getitem__(self, index):
        patient_file = 'patient{}'.format(f'{index + 1:04}')  # patient{0001}, patient{0002}, etc

        image_2CH_ED = self.__read_image(patient_file, '_2CH_ED.mhd')
        image_2CH_ES = self.__read_image(patient_file, '_2CH_ES.mhd')
        image_4CH_ED = self.__read_image(patient_file, '_4CH_ED.mhd')
        image_4CH_ES = self.__read_image(patient_file, '_4CH_ES.mhd')
        image_2CH_sequence = self.__read_image(patient_file, '_2CH_sequence.mhd')
        image_4CH_sequence = self.__read_image(patient_file, '_4CH_sequence.mhd')

        if self.data_type == 'train':
            image_2CH_ED_gt = self.__read_image(patient_file, '_2CH_ED_gt.mhd')
            image_2CH_ES_gt = self.__read_image(patient_file, '_2CH_ES_gt.mhd')
            image_4CH_ED_gt = self.__read_image(patient_file, '_4CH_ED_gt.mhd')
            image_4CH_ES_gt = self.__read_image(patient_file, '_4CH_ES_gt.mhd')

        info_2CH = self.__read_info('{}/{}/{}'.format(self.data_file, patient_file, 'Info_2CH.cfg'))
        info_4CH = self.__read_info('{}/{}/{}'.format(self.data_file, patient_file, 'Info_4CH.cfg'))

        if self.data_type == 'train':
            data = {
                # 'patient': patient_file,
                '2CH_ED': image_2CH_ED,
                # '2CH_ES': image_2CH_ES,
                # '4CH_ED': image_4CH_ED,
                # '4CH_ES': image_4CH_ES,
                # '2CH_sequence': image_2CH_sequence,
                # '4CH_sequence': image_4CH_sequence,
                '2CH_ED_gt': image_2CH_ED_gt,
                # '2CH_ES_gt': image_2CH_ES_gt,
                # '4CH_ED_gt': image_4CH_ED_gt,
                # '4CH_ES_gt': image_4CH_ES_gt,
                # 'info_2CH': info_2CH,  # Dictionary of infos
                # 'info_4CH': info_4CH  # Dictionary of infos
            }
        elif self.data_type == 'test':
            data = {
                # 'patient': patient_file,
                '2CH_ED': image_2CH_ED,
                # '2CH_ES': image_2CH_ES,
                # '4CH_ED': image_4CH_ED,
                # '4CH_ES': image_4CH_ES,
                # '2CH_sequence': image_2CH_sequence,
                # '4CH_sequence': image_4CH_sequence,
                # 'info_2CH': info_2CH,  # Dictionary of infos
                # 'info_4CH': info_4CH  # Dictionary of infos
            }

        # Transforms
        for transform in self.global_transforms:
            data = transform(data)
        for transform in self.augment_transforms:
            data = transform(data)

        return data

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ResizeImagesAndLabels(object):
    '''
    Ripped out of Prof. Stough's code
    '''

    def __init__(self, size, fields=['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES',
                                     '2CH_ED_gt', '2CH_ES_gt', '4CH_ED_gt', '4CH_ES_gt']):
        self.size = size
        self.fields = fields

        self.fields = ['2CH_ED','2CH_ED_gt']

    def __call__(self, data):
        for field in self.fields:
            # transpose to go from chan x h x w to h x w x chan and back.
            data[field] = resize(data[field].transpose([1, 2, 0]),
                                 self.size, mode='constant',
                                 anti_aliasing=True)
            data[field] = data[field].transpose([2, 0, 1])

        # normalizing the pixel values
        data['2CH_ED'] /= 255.0
        # data['2CH_ES'] /= 255.0
        # data['4CH_ED'] /= 255.0
        # data['4CH_ES'] /= 255.0

        return data

global_transforms = [
    ResizeImagesAndLabels(size=[256, 256])
]
augment_transforms = [
    #AddSaltPepper(freq = .1)
]

train_iter = CamusIterator(
    data_type='train',
    global_transforms=global_transforms,
    augment_transforms=augment_transforms,
)

test_iter = CamusIterator(
    data_type='test',
    global_transforms=global_transforms,
    augment_transforms=augment_transforms,
)

def read_data(batch_size, validation_split):
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(train_iter)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(train_iter, batch_size=batch_size, sampler=train_sampler, num_workers=0, drop_last=True)
    valid_dataloader = DataLoader(train_iter, batch_size=batch_size, sampler=valid_sampler, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_iter, batch_size=batch_size, num_workers=0)

    return train_dataloader, valid_dataloader, test_dataloader