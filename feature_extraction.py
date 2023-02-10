import gzip
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import (AlexNet_Weights, ViT_L_16_Weights, alexnet,
                                vit_l_16)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


def setup_logging() -> logging.Logger:
    res = logging.getLogger('feature_extraction')
    res.setLevel(logging.DEBUG)

    if not res.hasHandlers():
        # We can avoid double handling when reloading the module
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        res.addHandler(handler)

    return res


logger = setup_logging()

class SquaredCentered(nn.Module):
    """Makes a rectangle image to a square image of a fixed size"""
    def __init__(self, fill: Union[int, float] = 0):
        super().__init__()
        self.fill = fill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ignored, h, w = x.shape
        left_pad   = max(h-w, 0) // 2
        right_pad  = max(h-w, 0) // 2 + max(h-w, 0) % 2
        top_pad    = max(w-h, 0) // 2
        bottom_pad = max(w-h, 0) // 2 + max(w-h, 0) % 2
        return F.pad(x, (left_pad, top_pad, right_pad, bottom_pad), self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fill={self.fill})"

def square_resize(size: int, fill: int = 255) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        SquaredCentered(fill),
        T.Resize(size)
    )

TransformType = Callable[[torch.Tensor], torch.Tensor]

# The default image transform for feature extraction
DEFAULT_TRANSFORM = square_resize(size=400, fill=255)


def get_default_device() -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu")

class ImageFeatureExtractionDataset(Dataset):
    """Image dataset from a directory"""

    _repr_indent = 4

    def __init__(self,
        root: Path,
        feature_extension: str,
        device: Optional[str] = None,
        transform: Optional[TransformType] = None,
        force: bool = False,
    ):
        """
        Args:
            root: Directory with all the images
            feature_extension: Extension for the feature files
            force: If true consider all images, despite them already having
                   generated features
            transform: Optional transform to be applied on a sample.
        """
        self.root = root
        if device is None:
            device = get_default_device()
        self.device = device
        self.transform = transform
        self.feature_extension = feature_extension
        self.force = force

        # Both of them have names like '41DlkYa9DtL.jpg'  (for pending images)
        # or '41DlkYa9DtL.something' (for feature files)
        self.pending_images, self.unnecessary = self.list_files()

    def list_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Finds the pending files needing extraction
        """
        images = {}
        feature_files = {}

        # We don't use pathlib here because is slower
        for filename in os.listdir(self.root):
            stem, extension = os.path.splitext(filename)
            if extension == '.jpg':
                if os.path.getsize(os.path.join(self.root, filename)) > 0:
                    images[stem] = filename
            elif extension == f'.{self.feature_extension}':
                feature_files[stem] = filename

        if self.force:
            unecessary = feature_files.values()
            pending = images.values()
        else:
            unecessary = [
                p
                for stem, p in feature_files.items()
                if stem not in images
            ]
            pending = [
                p
                for stem, p in images.items()
                if stem not in feature_files
            ]

        return pending, unecessary


    def __len__(self):
        return len(self.pending_images)

    def get_image(self, filename: str) -> torch.Tensor:

        # We don't use pathlib here because is slower
        rel_path = os.path.join(self.root, filename)
        img = read_image(rel_path, mode=ImageReadMode.RGB)
        if self.device:
            img = img.to(device=self.device)
        if self.transform:
            img = self.transform(img)
        return {
            'img': img,
            'path': rel_path
        }

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idxs = idx.tolist()
            images_paths = [self.pending_images[idx] for idx in idxs]
        else:
            images_paths = self.pending_images[idx]

        if isinstance(images_paths, list):
            images = [
                self.get_image(image_path) for image_path in images_paths
            ]
        else:
            images = self.get_image(images_paths)

        return images

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Force: {self.force}")
        body.append(f"Device: {self.device}")
        body.append(f"Suffix Name: {self.feature_extension}")
        if self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


class AlexNetFeatureExtractor(nn.Module):
    """
    An AlexNet with extracted features using the last cell before the
    fully connected net used for classifying, following SEM-MacridVAE.
    """
    def __init__(self):
        super().__init__()
        weights = AlexNet_Weights.DEFAULT
        classifier = alexnet(weights=weights, progress=True).eval()

        self.transforms = weights.transforms()

        self.body = create_feature_extractor(
            classifier, return_nodes={
                'flatten': 'features',
                'classifier.6': 'output'
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            return self.body(x)


class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
        classifier = vit_l_16(weights=weights).eval()

        self.transforms = weights.transforms()

        self.body = create_feature_extractor(
            classifier, return_nodes={
                'getitem_5': 'features',
                'heads.head': 'output'
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            return self.body(x)


def save_features_to_file(dest: Path, features: torch.Tensor, progress: tqdm):
    nparray: npt.NDArray = features.numpy(force=True)
    assert nparray.dtype == np.float32
    with gzip.open(dest, mode='wb') as dest_gzip:
        dest_gzip.write(nparray.tobytes(order='C'))
    progress.update()


def load_features_from_file(src: Path):
    with gzip.open(src, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.float32)


def extract_features(
    asset_path: Path,
    feature_extractor: nn.Module,
    suffix_name: str,
    force: bool,
    batch_size: int,
    transform: Optional[TransformType] = None,
    device: Optional[str] = None,
):
    if device is None:
        device = get_default_device()

    logger.info('Listing images...')
    dataset = ImageFeatureExtractionDataset(
        asset_path,
        device=device,
        feature_extension=suffix_name,
        force=force,
        transform=transform
    )

    # Remove unnecessary files
    if len(dataset.unnecessary) > 0:
        for p in tqdm(dataset.unnecessary, unit='file', unit_scale=True,
                      desc='Removing files'):
            p.unlink(missing_ok=True)

    if len(dataset) == 0:
        logger.info('All images have extracted features :)')
        return

    dataloader = DataLoader(dataset, batch_size=batch_size)
    feature_extractor = feature_extractor.to(device)

    with tqdm(total=len(dataset), unit_scale=True, unit='image', smoothing=0,
              desc='Extracting features') as progress, \
            ThreadPoolExecutor(max_workers=8) as executor:
        for batch in dataloader:
            result = feature_extractor(batch['img'])
            combined_results = zip(batch['path'], result['features'])

            for img_path, features in combined_results:
                img_path = Path(img_path)
                dest = img_path.parent / f'{img_path.stem}.{suffix_name}'
                executor.submit(save_features_to_file, dest, features, progress)


def extract_features_one_image(
    image_path: Path,
    feature_extractor: nn.Module,
    device: Optional[str] = None,
    transform: Optional[TransformType] = DEFAULT_TRANSFORM,
) -> torch.Tensor:
    """Extract features for one image"""
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    if transform:
        image = transform(image)

    # The transform ONLY works with batches, so we feed it a batch with
    # a single element
    stacked_images = torch.stack([image]).to(device)
    return feature_extractor(stacked_images)['features']


def extract_alexnet_features(
    asset_path: Path,
    force: bool = False,
    batch_size: int = 1024,
    transform: Optional[TransformType] = DEFAULT_TRANSFORM,
    device: Optional[str] = None
):
    alexnet_feature_extractor = AlexNetFeatureExtractor()
    extract_features(
        asset_path=asset_path,
        force=force,
        feature_extractor=alexnet_feature_extractor,
        suffix_name='alexnet',
        batch_size=batch_size,
        transform=transform,
        device=device
    )


def extract_vit_features(
    asset_path: Path,
    force: bool = False,
    batch_size: int = 16,
    transform: Optional[TransformType] = DEFAULT_TRANSFORM,
    device: Optional[str] = None
):
    alexnet_feature_extractor = ViTFeatureExtractor()
    extract_features(
        asset_path=asset_path,
        force=force,
        feature_extractor=alexnet_feature_extractor,
        suffix_name='vit',
        batch_size=batch_size,
        transform=transform,
        device=device
    )
