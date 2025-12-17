import torchvision.transforms as transforms
import random
import numpy as np
import torch
import requests
import os
import io
import pandas as pd
from typing import Callable, Optional
from omegaconf import OmegaConf
from torch.utils.data import Subset
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_from_disk
from torchvision.datasets import CocoDetection
from io import BytesIO


image_column = 'image'
caption_column = 'text'


def tokenize_captions(examples, tokenizer, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )

    if not captions:
        captions.append('None')
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    return inputs.input_ids


def preprocess_train(examples, tokenizer):
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    images = [image.convert("RGB") for image in examples[image_column]]

    examples["pixel_values"] = [transform(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples, tokenizer)
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def load_pokemon_datasets(dataset_root, num_samples=415, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")
    dataset = load_from_disk(os.path.join(dataset_root, 'pokemon'))

    def preprocess_with_tokenizer(examples):
        return preprocess_train(examples, tokenizer)

    train_dataset = dataset["train"].select(range(num_samples)).with_transform(preprocess_with_tokenizer)
    test_dataset = dataset["test"].select(range(num_samples)).with_transform(preprocess_with_tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader


class CocoCaptionsDict(CocoDetection):

    def __init__(
            self,
            split,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            tokenizer=None,
    ) -> None:
        assert split in ['train', 'val']
        annFile = os.path.join(root, 'annotations/captions_val2017.json')
        self.split = split
        conf = OmegaConf.load(os.path.join(root, 'coco_split.yaml'))
        root = os.path.join(root, 'val2017')
        self._train_ids = conf['train']
        self._val_ids = conf['test']
        self.tokenizer = tokenizer

        super().__init__(root, annFile, transform, target_transform, transforms)

        if split == 'train':
            self.ids = [self.ids[i] for i in self._train_ids]
        elif split == 'val':
            self.ids = [self.ids[i] for i in self._val_ids]
        self._init_tokenize_captions()

    def _init_tokenize_captions(self):
        captions = []
        for id in self.ids:
            caption = [ann["caption"] for ann in super()._load_target(id)]
            # caption = ['None' for ann in super()._load_target(id)]s
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                # captions.append(random.choice(caption) if is_train else caption[0])
                captions.append(caption[0])
            else:
                raise ValueError()
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        self.input_ids = inputs.input_ids

    def _load_target(self, id: int):
        # return super()._load_target(id)
        return self.input_ids[id]

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(index)
        caption = super()._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # return image, target
        return {"pixel_values": image, "input_ids": target, 'caption': caption}


def load_coco_datasets(dataset_root, num_samples=100, tokenizer=None):
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = CocoCaptionsDict(split='train', transform=transform, tokenizer=tokenizer,
                                     root=os.path.join(dataset_root, 'coco2017val'))

    test_dataset = CocoCaptionsDict(split='val', transform=transform, tokenizer=tokenizer,
                                    root=os.path.join(dataset_root, 'coco2017val'))

    if num_samples is not None:
        train_indices = list(range(min(num_samples, len(train_dataset))))
        test_indices = list(range(min(num_samples, len(test_dataset))))

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def read_image_from_url(url):
    try:
        response = requests.get(url, timeout=1)
        image = Image.open(BytesIO(response.content))
        return image
    except:
        return None


def load_laion5_dataset(dataset_root, num_samples=100, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")
    dataset = load_dataset(dataset_root + '/laion_mi')

    members_dataset = dataset["members"]
    nonmembers_dataset = dataset["nonmembers"]

    def preprocess_with_tokenizer(examples):
        return preprocess_train(examples, tokenizer)

    def select_valid_samples(dataset, num_samples):
        valid_images = []
        valid_captions = []
        index = 0
        count = 0
        while len(valid_images) < num_samples and index < len(dataset):
            sample = dataset[index]
            url = sample['url']
            caption = sample['caption']

            image = read_image_from_url(url)
            if image:
                valid_images.append(image)
                valid_captions.append(caption)
                count += 1
                print(count)
            index += 1
        data_dict = {
            "image": valid_images,
            "text": valid_captions
        }
        return Dataset.from_dict(data_dict)

    train_dataset = select_valid_samples(members_dataset, num_samples).with_transform(preprocess_with_tokenizer)
    test_dataset = select_valid_samples(nonmembers_dataset, num_samples).with_transform(preprocess_with_tokenizer)

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader


def load_flickr_datasets(dataset_root, num_samples=100, tokenizer=None):
    folder_path = os.path.join(dataset_root, 'Flickr')
    dfs = []

    for file in os.listdir(folder_path):
        if file.endswith('.parquet'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_parquet(file_path)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    train_images = []
    train_texts = []
    test_images = []
    test_texts = []

    for index, row in combined_df.iterrows():
        split = row['split']
        image_dict = row['image']
        image_bytes = image_dict['bytes']

        image = Image.open(io.BytesIO(image_bytes))
        caption = row['caption']

        if split == 'train':
            train_images.append(image)
            train_texts.append(caption)
        elif split == 'test':
            test_images.append(image)
            test_texts.append(caption)

    train_data_dict = {
        "image": train_images,
        "text": train_texts
    }
    test_data_dict = {
        "image": test_images,
        "text": test_texts
    }

    def preprocess_with_tokenizer(examples):
        return preprocess_train(examples, tokenizer)

    train_dataset = Dataset.from_dict(train_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)

    train_dataset = train_dataset.select(range(min(num_samples, len(train_dataset)))).with_transform(preprocess_with_tokenizer)
    test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset)))).with_transform(preprocess_with_tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader


def load_ti_datasets(dataset_root, num_samples=100, tokenizer=None):
    folder_path = os.path.join(dataset_root, 'text-to-image-2m')
    dfs = []

    for file in os.listdir(folder_path):
        if file.endswith('new_data.parquet'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_parquet(file_path)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    train_images = []
    train_texts = []
    test_images = []
    test_texts = []

    for index, row in combined_df.iterrows():
        split = row['split']
        image_dict = row['image']
        image_bytes = image_dict['bytes']

        image = Image.open(io.BytesIO(image_bytes))
        caption = row['caption']

        if split == 'train':
            train_images.append(image)
            train_texts.append(caption)
        elif split == 'test':
            test_images.append(image)
            test_texts.append(caption)

    train_data_dict = {
        "image": train_images,
        "text": train_texts
    }
    test_data_dict = {
        "image": test_images,
        "text": test_texts
    }

    def preprocess_with_tokenizer(examples):
        return preprocess_train(examples, tokenizer)

    train_dataset = Dataset.from_dict(train_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)

    train_dataset = train_dataset.select(range(min(num_samples, len(train_dataset)))).with_transform(preprocess_with_tokenizer)
    test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset)))).with_transform(preprocess_with_tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader
