from pathlib import Path
import feature_extraction_image
import amazon_dataset
import numpy as np

dataset = 'Clothing_Shoes_and_Jewelry'
asset_path = amazon_dataset.images_dir(dataset)

print(f"Loading products")

products_df = amazon_dataset.items_df(dataset)

print(f"Extracting into {asset_path}")

vit_features = feature_extraction_image.extract_vit_features(
    products_df,
    asset_path=asset_path,
    batch_size=16
)

print(f'Done! Features available in {asset_path}')

np.savez_compressed(
    amazon_dataset.BASE_DATA_FOLDER / f'{dataset}_vit_features.npz',
    **vit_features
)

print(f'Done! Extracted features')
