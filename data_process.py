from PIL import Image
from PIL import ImageFile
from PIL import TiffImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import json
import torch
from torch.utils.data import Dataset, DataLoader
import os

class SentenceDataset(Dataset):
    def __init__(self, opt, data_path, photo_path, image_transforms):
        self.dataset_type = opt.dataset
        self.photo_path = photo_path
        self.image_transforms = image_transforms
        
        # read original text
        with open(data_path, 'r', encoding='utf-8') as file_read:
            file_content = json.load(file_read)
        
        self.data_id_list = []
        self.text_list = []
        self.label_list = []
        for data in file_content:
            self.data_id_list.append(data['id'])
            self.text_list.append(data['text'])
            self.label_list.append(data['label'])
            self.image_id_list = self.data_id_list
          

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.data_id_list)

    def __getitem__(self, index):
        # read images
        image_path = os.path.join(self.photo_path, str(self.data_id_list[index]) + '.jpg')
        
        # If can not be found，try read in dataset-image2 
        if not os.path.exists(image_path):
            alternative_path = os.path.join('dataset-image2', str(self.data_id_list[index]) + '.jpg')
            if os.path.exists(alternative_path):
                image_path = alternative_path
            else:
                raise FileNotFoundError(f"Image {self.data_id_list[index]} not found in either {self.photo_path} or /kaggle/input/dataset-image2")
        
        # open and convert image
        main_image = Image.open(image_path)
        main_image = self.image_transforms(main_image)
      
        text = self.text_list[index]
        return text, main_image, self.label_list[index]        



def data_process(opt, data_path, photo_path, data_type):
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SentenceDataset(opt, data_path, photo_path, img_transforms)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size,
                             shuffle=True if data_type == 1 else False,
                             num_workers=opt.num_workers, pin_memory=True)
    
    return data_loader, dataset.__len__()
