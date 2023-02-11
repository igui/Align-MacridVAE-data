from pathlib import Path
import feature_extraction

asset_path = Path('data/amazon/Clothing_Shoes_and_Jewelry_product_images/')
feature_extraction.extract_vit_features(asset_path=asset_path, batch_size=16)
