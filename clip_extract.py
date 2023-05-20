from pathlib import Path
import feature_extraction_image
import amazon_dataset

dataset = 'Home_and_Kitchen'
asset_path = amazon_dataset.product_images_dir(dataset)

print(f"Extracting into {asset_path}")

feature_extraction_image.extract_clip_features(
    # Not necessary to extract features. Only necessary to save the "mean"
    # product image
    products_df=None,
    asset_path=asset_path,
    batch_size=16)

print(f'Done! Features available in {asset_path}')
