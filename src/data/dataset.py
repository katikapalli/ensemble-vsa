import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, config=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            config (dict): Configuration containing class names and data processing info.
            split (string): One of 'train', 'valid', 'test' to determine the dataset split.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.config = config
        self.split = split
        self.classes = self.config['dataset']['emotions']  # List of class labels
        self.images = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        class_images = {emotion: [] for emotion in self.classes}
        image_extensions = ['.jpg', '.png']

        for emotion in self.classes:
            emotion_dir = os.path.join(self.root_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Warning: Directory {emotion_dir} does not exist.")
                continue
            class_images[emotion] = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) 
                                     if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        val_images, test_images, train_images = self._split_data(class_images)

        if self.split == 'train':
            self._set_images_and_labels(train_images)
        elif self.split == 'valid':
            self._set_images_and_labels(val_images)
        elif self.split == 'test':
            self._set_images_and_labels(test_images)
        else:
            raise ValueError("Split should be one of 'train', 'valid', or 'test'")

    def _split_data(self, class_images):
        val_images, test_images, train_images = {}, {}, {}
        
        for emotion in self.classes:
            all_images = class_images[emotion]

            random.seed(0)
            random.shuffle(all_images)
            
            val_count = min(self.config['dataset']['val_size_per_class'], len(all_images))
            test_count = min(self.config['dataset']['test_size_per_class'], len(all_images) - val_count)
            
            val_images[emotion] = all_images[:val_count]
            test_images[emotion] = all_images[val_count:val_count + test_count]
            remaining_images = all_images[val_count + test_count:]
            
            train_images[emotion] = self._balance_data(remaining_images, self.config['dataset']['train_size_per_class'])
        
        return val_images, test_images, train_images

    def _balance_data(self, images, target_size):
        if len(images) > target_size:
            return random.sample(images, target_size)
        elif len(images) < target_size and len(images) > 0:
            return images * (target_size // len(images)) + random.sample(images, target_size % len(images))
        else:
            return images
        
    def _set_images_and_labels(self, image_dict):
        for emotion in self.classes:
            images = image_dict.get(emotion, [])
            self.images.extend(images)
            self.labels.extend([self.classes.index(emotion)] * len(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
