from pathlib import Path
import feature_extraction_image
import amazon_dataset

dataset = 'Clothing_Shoes_and_Jewelry'
asset_path = amazon_dataset.product_images_dir(dataset)
products_df = amazon_dataset.products_df(dataset)

print(f"Extracting into {asset_path}")

feature_extraction_image.extract_clip_features(
    products_df=products_df,
    asset_path=asset_path,
    batch_size=16)

print(f'Done! Features available in {asset_path}')
