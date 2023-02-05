import gzip
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import count, islice
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import jsonlines
import pandas as pd
import requests
import tqdm
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, \
    String, create_engine
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship
import urllib3


BASE_DATA_FOLDER = Path('data/amazon/')
IMAGE_DOWNLOAD_PROCESSES = 4
READ_CHUNK_SIZE = 16384
DB_CHUNK_SIZE = 2000
BASE_SOURCE_URL = 'https://jmcauley.ucsd.edu/data/amazon_v2'
DB_PATH = BASE_DATA_FOLDER / 'db.sqlite'


def setup_logging() -> logging.Logger:
    res = logging.getLogger('amazon_dataset')
    res.setLevel(logging.DEBUG)

    if not res.hasHandlers():
        # We can avoid double handling when reloading the module
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        res.addHandler(handler)

    return res


logger = setup_logging()

# Not super necessary, avoid noisy warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# This Part only defines the required DB for importing data using SQLAlchemy


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "product"

    id = Column(Integer, primary_key=True)
    asin = Column(
        String(20),
        nullable=False,
        index=True,
        unique=True
    )
    description = Column(String())
    title = Column(String(), nullable=False)
    brand = Column(String())
    main_cat = Column(String())
    # details = Column(String()) # Has long HTML text that is not important
    # fit = Column(String())     # Noisy text
    # date = Column(String())    # Does not contain dates mostly
    rank = Column(String())
    price = Column(String())

    product_categories = relationship(
        "ProductCategory",
        cascade="all, delete-orphan",
        backref="product"
    )

    product_features = relationship(
        "ProductFeature",
        cascade="all, delete-orphan",
        backref="product"
    )

    product_images = relationship(
        "ProductImage",
        cascade="all, delete-orphan",
        backref="product"
    )

    related_products = relationship(
        "RelatedProduct",
        cascade="all, delete-orphan",
        backref="product"
    )

    technical_details = relationship(
        "TechnicalDetail",
        cascade="all, delete-orphan",
    )


class ProductCategory(Base):
    __tablename__ = "product_category"

    id = Column(Integer, primary_key=True)
    name = Column(String(), nullable=False, index=True)

    product_id = Column(ForeignKey("product.id"))


class ProductImage(Base):
    __tablename__ = "product_image"

    id = Column(Integer, primary_key=True)
    url = Column(String(), nullable=False, index=True)

    product_id = Column(ForeignKey("product.id"))


class ProductFeature(Base):
    __tablename__ = "product_feature"

    id = Column(Integer, primary_key=True)
    name = Column(String(), nullable=False, index=True)

    product_id = Column(ForeignKey("product.id"))


class RelatedProduct(Base):
    __tablename__ = "related_product"

    id = Column(Integer, primary_key=True)
    asin = Column(String(), nullable=False, index=True)
    kind = Column(String(), nullable=False, index=True)
    product_id = Column(ForeignKey("product.id"))


class TechnicalDetail(Base):
    __tablename__ = "technical_detail"

    id = Column(Integer, primary_key=True)
    name = Column(String(), nullable=False)
    value = Column(String(), nullable=False)
    kind = Column(String(), nullable=False, index=True)
    product_id = Column(ForeignKey("product.id"))


PRODUCT_TABLES = [Product, ProductCategory, ProductImage, ProductFeature,
                  RelatedProduct, TechnicalDetail]


class Review(Base):
    __tablename__ = "review"

    id = Column(Integer, primary_key=True)
    asin = Column(String(20), nullable=False, index=True)
    reviewerID = Column(String(), nullable=False, index=True)
    reviewerName = Column(String())
    overall = Column(Float, nullable=False)
    text = Column(String)
    reviewTime = Column(DateTime, nullable=False, index=True)
    summary = Column(String)
    verified = Column(Boolean, nullable=False)
    vote = Column(Integer)

    review_images = relationship(
        "ReviewImage",
        cascade="all, delete-orphan",
        backref="review"
    )

    review_styles = relationship(
        "ReviewStyle",
        cascade="all, delete-orphan",
        backref="review"
    )


class ReviewImage(Base):
    __tablename__ = "review_image"

    id = Column(Integer, primary_key=True)
    url = Column(String(), nullable=False, index=True)

    review_id = Column(ForeignKey("review.id"))


class ReviewStyle(Base):
    __tablename__ = "review_style"

    id = Column(Integer, primary_key=True)
    name = Column(String(), nullable=False)
    value = Column(String(), nullable=False)

    review_id = Column(ForeignKey("review.id"))


REVIEW_TABLES = [Review, ReviewImage, ReviewStyle]


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    This hook allows using foreign keys ALL the time when using the Database
    https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support

    Also, use the "unsafe" but faster memory journal mode
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=MEMORY")
    cursor.close()


def initialize_db() -> Path:
    """
    Initializes the DB using WAL, for a little faster DB writes
    """
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=True)
    Base.metadata.create_all(engine)

    # Use WAL for faster connections
    cursor = engine.raw_connection().cursor()
    cursor.execute("PRAGMA journal_mode=MEMORY")
    cursor.close()

    return DB_PATH


def get_product_description(obj: Dict) -> str:
    """
    Flattens the product description from a parsed JSON
    """
    description = obj.get('description')
    if isinstance(description, list):
        description = '\n'.join(description)
    return description


def get_product_rank(obj: Dict) -> str:
    """
    Flattens the rank from a parsed JSON
    """
    rank = obj.get('rank')
    if isinstance(rank, list):
        rank = ' '.join(rank)
    return rank


def create_session() -> Session:
    """Returns a session inside the SQLAlchemy"""
    return Session(create_engine(f"sqlite:///{DB_PATH}"))


def process_product_chunk(chunk: List[Tuple[Dict, int]], session: Session):
    """
    Process one chunk of products. The chunks are parsed but can have
    some dodgy information.

    Each chunk has a suggested primary id (the line number in the original file)
    """
    # Only preserve objects with title
    obj_with_title = [(obj, obj_id) for obj, obj_id in chunk if 'title' in obj]

    products = [
        {
            'id': obj_id,
            'asin': obj['asin'],
            'description': get_product_description(obj),
            'title': obj['title'],
            'brand': obj.get('brand'),
            'main_cat': obj.get('main_cat'),
            'rank': get_product_rank(obj),
            'price': obj.get('price')
        }
        for obj, obj_id in obj_with_title
    ]

    # We should check which objects we inserted to avoid inserting unnecessary
    # objects
    result = session.execute(
        insert(Product.__table__).on_conflict_do_nothing().returning(Product.id),
        products
    )
    inserted_ids = {obj_id for (obj_id, ) in result.fetchall()}
    session.commit()

    # Here, we only consider related objects to the ones we inserted
    inserted_objs = [
        (obj, obj_id) for (obj, obj_id) in chunk if obj_id in inserted_ids
    ]

    product_images = [
        {'url': url, 'product_id': obj_id}
        for obj, obj_id in inserted_objs
        for url in obj.get('image', [])
    ]
    product_categories = [
        {'name': name, 'product_id': obj_id}
        for obj, obj_id in inserted_objs
        for name in obj['category']
    ]
    product_features = [
        {'name': name, 'product_id': obj_id}
        for obj, obj_id in inserted_objs
        for name in obj.get('feature', [])
    ]

    # The ones that are plain. Those are list of ASINs
    related_products = [
        {'asin': asin, 'kind': rel_product_key, 'product_id': obj_id}
        for obj, obj_id in inserted_objs
        for rel_product_key in ('also_view', 'also_buy')
        for asin in obj.get(rel_product_key, [])
    ]

    # The ones that are a dictionary. Those are list of dictionaries.
    related_products += [
        {'asin': rel['asin'], 'kind': 'similar_item', 'product_id': obj_id}
        for obj, obj_id in inserted_objs
        for rel in obj.get('similar_item', [])
        if rel['asin']
    ]

    technical_details = [
        {
            'name': key,
            'value': value,
            'kind': technical_detail_key,
            'product_id': obj_id
        }
        for obj, obj_id in inserted_objs
        for technical_detail_key in ('tech1', 'tech2')
        for key, value in obj.get(technical_detail_key, {}).items()
    ]

    # If any of the products fail, then we have to
    session.bulk_insert_mappings(ProductImage, product_images)
    session.bulk_insert_mappings(ProductCategory, product_categories)
    session.bulk_insert_mappings(ProductFeature, product_features)
    session.bulk_insert_mappings(RelatedProduct, related_products)
    session.bulk_insert_mappings(TechnicalDetail, technical_details)
    session.commit()


def recreate_product_tables():
    """
    Drop and create all tables related to Product data. So, we
    can recreate tables correctly. Usually necessary before a full import
    """
    engine = create_engine(f"sqlite:///{DB_PATH}")
    tables = [x.__table__ for x in PRODUCT_TABLES]
    Base.metadata.drop_all(engine, tables)
    Base.metadata.create_all(engine, tables)


def load_metadata_into_db(src: Path, force = False, max_workers: int = 2):
    """
    Loads all the product metadata into the DB. It will remove all previous
    products!

    max_workers can be ignored
    """
    if force:
        recreate_product_tables()

    with create_session() as session:
        if session.query(Review).first():
            print('There are records. Use force=True to force removal')
            return

    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm.tqdm(unit='product', unit_scale=True) as progress, \
            create_session() as session, \
            gzip.open(src) as file, \
            jsonlines.Reader(file) as reader:
        # Here we read the total lines in parallel to not slow down
        # the file download
        def set_total():
            line_total = line_count_gzip(src)
            progress.total = line_total
        executor.submit(set_total)

        for chunk in chunked_iterator(reader, DB_CHUNK_SIZE):
            process_product_chunk(chunk, session)
            progress.update(len(chunk))


def download_file(url: str, dest_file: str):
    resp = requests.get(url, stream=True, verify=False)
    resp.raise_for_status()

    with open(dest_file, 'wb') as f:
        progress = tqdm.tqdm(unit="B", unit_scale=True, unit_divisor=1024)

        if 'content-length' in resp.headers:
            progress.total = int(resp.headers['content-length'])

        for data in resp.iter_content(chunk_size=READ_CHUNK_SIZE):
            if data:
                progress.update(len(data))
                f.write(data)


def get_metafile(dataset: str) -> str:
    BASE_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    filename = f'meta_{dataset}.json.gz'
    if not filename or '/' in filename:
        raise ValueError('invalid filename')

    dest_file = BASE_DATA_FOLDER / filename
    if os.path.exists(dest_file):
        logger.info(f'Not downloading {filename}. File already exist')
    else:
        url = f'{BASE_SOURCE_URL}/metaFiles/{filename}'
        download_file(url, dest_file)

    return dest_file


def get_categoryfile(dataset: str):
    BASE_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    filename = f'{dataset}.json.gz'
    if not filename or '/' in filename:
        raise ValueError('invalid filename')

    dest_file = BASE_DATA_FOLDER / filename
    if os.path.exists(dest_file):
        logger.info(f'Not downloading {filename}. File already exist')
    else:
        download_file(f'{BASE_SOURCE_URL}/categoryFiles/{filename}', dest_file)

    return dest_file


def get_duplicated_product_list() -> str:
    BASE_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    filename = 'duplicates.txt'
    dest_file = BASE_DATA_FOLDER / filename

    if os.path.exists(dest_file):
        logger.info(f'Not downloading {filename}. File already exist')
    else:
        url = f'{BASE_SOURCE_URL}/metaFiles/{filename}'
        download_file(url, dest_file)

    return dest_file


def line_count_gzip(filename_gz: Path) -> int:
    with gzip.open(str(filename_gz)) as f:
        lines = 0
        while True:
            buf = f.read(1024 * 1024)
            if not buf:
                return lines
            lines += buf.count(b'\n')


def line_count(filename: Path) -> int:
    with open(filename, 'rb') as f:
        lines = 0
        while True:
            buf = f.read(1024 * 1024)
            if not buf:
                return lines
            lines += buf.count(b'\n')


def read_duplicate_products(src: Path) -> List[Set[str]]:
    res = []

    total_lines = line_count(src)
    with open(src, 'r') as file,\
            tqdm.tqdm(unit='line', unit_scale=True, total=total_lines) as progress:
        for line in file:
            line_set = {
                 asin for asin in line.split()
            }
            progress.update()
            if len(line_set) > 1:
                # Empty sets don't matter
                # sets with one item mean the duplicates are outside
                # our product set
                res.append(line_set)

    return res


def collect_image_paths(
        images: List[str],
        max_dimension: int = 400
) -> Dict[str, str]:
    image_re = re.compile(
        r'(?P<prefix>https:\/\/images-na\.ssl-images-amazon\.com\/images\/I\/'
        r'(?P<name>.*)\.)'
        r'(?P<dimensions>_((AC_)?(SX\d+_SY\d+_CR(,\d+)+)|(SR\d+,\d+)|(SS\d+))_)'
        r'(?P<suffix>\.jpg)'
    )
    image_urls = {}
    for image in images:
        match = image_re.match(image)
        if not match:
            raise ValueError('Invalid Image')

        filename = f'{match.group("name")}{match.group("suffix")}'
        dest_path = BASE_DATA_FOLDER / 'metadata' / filename

        if dest_path.exists():
            continue  # Avoid downloading the image again

        url = (f'{match.group("prefix")}_SX{max_dimension}'
               f'_{match.group("suffix")}')

        if url in image_urls:
            logger.info(f'Duplicated URL {url} (should not happen)')

        image_urls[url] = dest_path

    return image_urls


def download_an_image(item: Tuple[str, str]):
    url, dest = item
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(dest, 'wb') as dest_file:
        for chunk in resp.iter_content(chunk_size=READ_CHUNK_SIZE):
            dest_file.write(chunk)


def download_images(metadata_df: pd.DataFrame):
    logger.info('Collecting URLs')
    image_urls: Dict[str, Path] = {}
    for _, item in metadata_df.iterrows():
        item_urls = collect_image_paths(item['image'])

        # Multiple products have the same image. Download it once
        for url, dest_file in item_urls.items():
            old_dest_file = image_urls.get(url)
            if old_dest_file is None:
                image_urls[url] = dest_file
            else:
                # Same URLs should have the same destination file
                assert old_dest_file == dest_file

    if not image_urls:
        logger.info('Nothing to download')
        return

    logger.info(f'I need to download {len(image_urls)} files')

    # It might not exist the first time we download the images
    (BASE_DATA_FOLDER / 'metadata').mkdir(exist_ok=True)

    pool = ThreadPool(processes=IMAGE_DOWNLOAD_PROCESSES)
    results = pool.imap_unordered(download_an_image, image_urls.items())
    for _ in tqdm.tqdm(results, total=len(image_urls), unit='image'):
        pass


def process_review_chunk(chunk: List[Tuple[Dict, int]], session: Session):
    reviews = [
        {
            'id': obj_id,
            'asin': obj['asin'],
            'reviewerID': obj['reviewerID'],
            'reviewerName': obj.get('reviewerName'),
            'overall': obj['overall'],
            'text': obj.get('reviewText'),
            'reviewTime': datetime.fromtimestamp(obj['unixReviewTime']),
            'summary': obj.get('summary'),
            'verified': obj['verified'],
            'vote': obj.get('vote'),
        }
        for obj, obj_id in chunk
    ]
    review_images = [
        {'url': image_url, 'review_id': obj_id}
        for obj, obj_id in chunk
        for image_url in obj.get('image', [])
    ]
    review_style = [
        {'name': k, 'value': v, 'review_id': obj_id}
        for obj, obj_id in chunk
        for k, v in obj.get('style', {}).items()
    ]

    session.bulk_insert_mappings(Review, reviews)
    session.bulk_insert_mappings(ReviewImage, review_images)
    session.bulk_insert_mappings(ReviewStyle, review_style)
    session.commit()


def chunked_iterator(it: Iterable, chunk_size: int, init_count=1) -> Iterable:
    """
    Generates an iterable that goes into chunk of size chunk_size, each
    element gets an unique index

    For example list(chunked_iterator('abc', chunk_size=2)) gets two elements:
    [
     (('a', 1), ('b', 2)),
     (('c', 3),)
    ]

    """
    enumerated_reader = zip(it, count(init_count))
    return iter(lambda: tuple(islice(enumerated_reader, chunk_size)), ())


def recreate_reviews_tables():
    """
    Drop and create all tables related to review data (only review). So, we
    can recreate tables correctly
    """
    engine = create_engine(f"sqlite:///{DB_PATH}")
    tables = [x.__table__ for x in REVIEW_TABLES]
    Base.metadata.drop_all(engine, tables)
    Base.metadata.create_all(engine, tables)


def load_reviews_into_db(src: Path, force=False, max_workers: int = 2):
    if force:
        recreate_reviews_tables()

    # Use this to get the ID
    ## s.query(func.max(Review.id)).scalar()

    with create_session() as session:
        if session.query(Review).first():
            print('There are reviews. Use force=True to force removal')
            return

    progress_options = {
        'unit': 'review', 'unit_scale': True, 'smoothing': 0.01
    }
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm.tqdm(**progress_options) as progress, \
            create_session() as session, \
            gzip.open(src) as file, \
            jsonlines.Reader(file) as reader:

        # Here we read the total lines in parallel to not slow down
        # the file download
        def set_total():
            line_total = line_count_gzip(src)
            progress.total = line_total
        executor.submit(set_total)

        for chunk in chunked_iterator(reader, DB_CHUNK_SIZE):
            process_review_chunk(chunk, session)
            progress.update(len(chunk))

