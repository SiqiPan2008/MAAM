import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import math
import random

def resizeLongEdge(img, longEdgeSize = 0):
    width, height = img.size
    if longEdgeSize == 0:
        longEdgeSize = max(width, height)
    if width > height:
        newSize = (longEdgeSize, int(height * longEdgeSize / width))
        loc = (0, int((longEdgeSize - newSize[1]) / 2))
    else:
        newSize = (int(longEdgeSize * width / height), longEdgeSize)
        width, _ = img.size
        loc = (int((longEdgeSize - newSize[0]) / 2), 0)
    img = img.resize(newSize)
    blackBackground = Image.new("RGB", (longEdgeSize, longEdgeSize), "black")
    blackBackground.paste(img, loc)
    return blackBackground

class MultiTransformDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, num_transforms=5, exclude_classes=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.num_transforms = num_transforms
        if exclude_classes is not None:
            self.exclude_classes = set(exclude_classes)
            self._filter_classes()
        
    def _filter_classes(self):
        filtered_samples = []
        for sample in self.dataset.samples:
            path, class_idx = sample
            class_name = self.dataset.classes[class_idx]
            if class_name not in self.exclude_classes:
                filtered_samples.append(sample)
        self.dataset.samples = filtered_samples
        self.dataset.targets = [s[1] for s in filtered_samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        transformed_imgs = [self.transform(img) for _ in range(self.num_transforms)]
        return transformed_imgs, label

def delete_random_files(directory, num_files_to_delete):
    files = os.listdir(directory)
    files_to_delete = random.sample(files, num_files_to_delete)
    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        os.remove(file_path)
        print(f"Deleted file: {file}")



if __name__ == '__main__':
    
    sideLength = 224
    
    transformAndResize = transforms.Compose([
        transforms.Lambda(lambda x: resizeLongEdge(x, longEdgeSize = sideLength)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1),
        transforms.RandomAffine(degrees = 10, translate = (0.05, 0.05), scale = (0.8, 1.2)),
        transforms.ToTensor(),
    ])
    onlyResize = transforms.Compose([
        transforms.Lambda(lambda x: resizeLongEdge(x, longEdgeSize = sideLength)),
        transforms.ToTensor(),
    ]) # for debugging

    transformation = transformAndResize
    input_dir = 'OCT-Original' 
    output_dir = 'OCT'
    os.makedirs(output_dir, exist_ok=True)
    all_classes = [f.name for f in os.scandir(input_dir) if f.is_dir()]
    
    target_num = 5000
    
    for class_name in all_classes:
        include_classes = [class_name]
        exclude_classes = [f for f in all_classes if f != class_name]
        class_input_dir = input_dir + "/" + class_name
        class_output_dir = output_dir + "/" + class_name
        os.makedirs(class_output_dir, exist_ok=True)
        file_num = len(os.listdir(class_input_dir))
        num_transforms_per_image = math.ceil(target_num / file_num)
        
        dataset = MultiTransformDataset(root_dir=input_dir, transform=transformation, num_transforms=num_transforms_per_image, exclude_classes=exclude_classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        class_names = dataset.dataset.classes

        index = 0
        for i, (imgs, labels) in enumerate(dataloader):
            label = labels[0].item()
            for j, img in enumerate(imgs):
                img = transforms.ToPILImage()(img[0])
                img.save(os.path.join(output_dir + "/" + class_names[label], f'{index}.jpg'))
                index += 1
                if index % 10 == 0:
                    print(f"Saving image number {index}")
        print(f"Transformed images saved to {class_output_dir}")
        
        delete_random_files(class_output_dir, num_transforms_per_image * file_num - target_num)
        print(f"There are now 5000 files in {class_output_dir}")
        
        