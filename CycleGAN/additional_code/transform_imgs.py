import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

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

"""
def normalizeBrightness(img, target_mean, target_std):
       grayscale_img = transforms.Grayscale(num_output_channels=1)(img)
        grayscale_tensor = transforms.ToTensor()(grayscale_img)
        current_std = torch.std(grayscale_tensor)
    
        if current_std > 0:
            std_adjustment_factor = target_std / (current_std + 1e-6)
        else:
            std_adjustment_factor = 1.0
        
        img_tensor = transforms.ToTensor()(img)
        normalized_img_tensor = (img_tensor - img_tensor.mean(dim=[1,2], keepdim=True)) * std_adjustment_factor + target_mean
        normalized_img_tensor = torch.clamp(normalized_img_tensor, 0.0, 1.0)
        
        grayscale_img = transforms.Grayscale(num_output_channels=1)(img)
        grayscale_tensor = transforms.ToTensor()(grayscale_img)
        min_grayscale = torch.min(grayscale_tensor)
        max_grayscale = torch.max(grayscale_tensor)
        
        img = transforms.ToTensor()(img)
        normalized_img_tensor = (img - min_grayscale) / (max_grayscale - min_grayscale + 1e-6)
        
        return normalized_img_tensor
"""

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

if __name__ == '__main__':
    
    sideLength = 256
    
    transformAndResize = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1),
        transforms.RandomAffine(degrees = 10, translate = (0.05, 0.05), scale = (0.9, 1.2)),
        transforms.Lambda(lambda x: resizeLongEdge(x, longEdgeSize = sideLength)),
        transforms.ToTensor(),
    ])
    onlyResize = transforms.Compose([
        transforms.Lambda(lambda x: resizeLongEdge(x, longEdgeSize = sideLength)),
        transforms.ToTensor(),
    ]) # for debugging

    num_transforms_per_image = 15
    transformation = transformAndResize
    input_dir = 'datasets/Normal_ERM' 
    output_dir = 'datasets/Transformed_New_ERM'
    include_classes = ['trainB']
    exclude_classes = ['trainA', 'testA', 'testB']
    os.makedirs(output_dir, exist_ok=True)
    for class_name in include_classes:
        os.makedirs(output_dir + "/" + class_name, exist_ok=True)

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
            if index % 5 == 0:
                print(f"Saving image number {index}")
    print(f"Transformed images saved to {output_dir}")