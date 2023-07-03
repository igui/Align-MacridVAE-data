import gzip
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from itertools import count, islice
from multiprocessing.pool import ThreadPool
from pathlib import Path
from shutil import COPY_BUFSIZE
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import jsonlines
import pandas as pd
import requests
import tqdm
import urllib3
from requests import ConnectionError
from sqlalchemy import (Boolean, Column, Date, Float, ForeignKey, Integer, Select,
                        String, create_engine, delete, event, func, select,
                        text, update)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (DeclarativeBase, Session, declarative_mixin,
                            relationship, aliased)
from sqlalchemy.schema import CreateIndex, DropIndex, DropTable
from sqlalchemy.sql.functions import GenericFunction

BASE_DATA_FOLDER = Path('data/amazon/')
IMAGE_DOWNLOAD_PROCESSES = 4
DB_CHUNK_SIZE = 2000
BASE_SOURCE_URL = 'https://jmcauley.ucsd.edu/data/amazon_v2'
# Prefix for images in the dataset
IMAGE_PREFIX = 'https://images-na.ssl-images-amazon.com/images/I/'


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
    title = Column(String())
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

    product_id = Column(ForeignKey("product.id", ondelete='CASCADE'))


class ProductImage(Base):
    __tablename__ = "product_image"

    id = Column(Integer, primary_key=True)
    url = Column(String(), nullable=False, index=True)
    main = Column(Boolean(), nullable=False, default=False)
    slug = Column(String(), index=True)

    product_id = Column(ForeignKey("product.id"))


class ProductFeature(Base):
    __tablename__ = "product_feature"

    id = Column(Integer, primary_key=True)
    name = Column(String(), nullable=False, index=True)

    product_id = Column(ForeignKey("product.id", ondelete='CASCADE'))


class RelatedProduct(Base):
    __tablename__ = "related_product"

    id = Column(Integer, primary_key=True)
    asin = Column(String(), nullable=False, index=True)
    kind = Column(String(), nullable=False, index=True)
    product_id = Column(ForeignKey("product.id", ondelete='CASCADE'))


class TechnicalDetail(Base):
    __tablename__ = "technical_detail"

    id = Column(Integer, primary_key=True)
    name = Column(String(), nullable=False)
    value = Column(String(), nullable=False)
    kind = Column(String(), nullable=False, index=True)
    product_id = Column(ForeignKey("product.id", ondelete='CASCADE'))


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
    reviewTime = Column(Date, nullable=False, index=True)
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


@declarative_mixin
class ReviewImageMixin:
    id = Column(Integer, primary_key=True)
    url = Column(String(), nullable=False)
    review_id = Column(ForeignKey("review.id", ondelete='CASCADE'))


class ReviewImage(ReviewImageMixin, Base):
    __tablename__ = "review_image"


class TempReviewImage(ReviewImageMixin, Base):
    __tablename__ = "review_image_temp"


@declarative_mixin
class ReviewStyleMixin:
    id = Column(Integer, primary_key=True)
    name = Column(String(), nullable=False)
    value = Column(String(), nullable=False)

    review_id = Column(ForeignKey("review.id", ondelete='CASCADE'))


class ReviewStyle(ReviewStyleMixin, Base):
    __tablename__ = "review_style"


class TempReviewStyle(ReviewStyleMixin, Base):
    __tablename__ = "review_style_temp"


REVIEW_TABLES = [
    Review, ReviewImage, ReviewStyle, TempReviewImage, TempReviewStyle
]


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, _connection_record):
    """
    This hook allows using foreign keys ALL the time when using the Database
    https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()



class group_concat(GenericFunction):
    # Avoid re-registering the function if we reload the module
    _register = not hasattr(func, 'group_concat')

    type = String()

class json_object(GenericFunction):
    # Avoid re-registering the function if we reload the module
    _register = not hasattr(func, 'json_object')

    type = String()



def dataset_db_path(dataset: str) -> Path:
    """
    Convenient method to get the database path for a dataset
    """
    return BASE_DATA_FOLDER / f'{dataset}.sqlite'


def initialize_db(dataset: str) -> Path:
    """
    Initializes the DB
    """
    db_path = dataset_db_path(dataset)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return db_path


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


def create_session(dataset: str, echo=False) -> Session:
    """Returns a session inside the SQLAlchemy"""
    engine = create_engine(f"sqlite:///{dataset_db_path(dataset)}", echo=echo)
    return Session(engine)


def product_title_cleaned(title: Optional[str]) -> Optional[str]:
    if title is None:
        return None
    if 'function(' in title:
        return None  # it has javascript in 99% of cases
    return title


def get_image_urls(product: Dict) -> List[str]:
    return product.get('image', [])

def process_product_chunk(
        chunk: List[Tuple[Dict, int]],
        asins: Set[int],
        session: Session) -> int:
    """
    Process one chunk of products. The chunks are parsed but can have
    some dodgy information.

    Each chunk has a suggested primary id (the line number in the original file)
    """
    # Only preserve objects with title
    filtered_objs = [
        (obj, obj_id)
        for obj, obj_id in chunk
        if obj['asin'] in asins
    ]

    products = [
        {
            'id': obj_id,
            'asin': obj['asin'],
            'description': get_product_description(obj),
            'title': product_title_cleaned(obj.get('title')),
            'brand': obj.get('brand'),
            'main_cat': obj.get('main_cat'),
            'rank': get_product_rank(obj),
            'price': obj.get('price')
        }
        for obj, obj_id in filtered_objs
    ]

    # We should check which objects we inserted to avoid inserting unnecessary
    # objects
    result = session.execute(
        insert(Product.__table__).on_conflict_do_nothing().returning(
            Product.id),
        products
    )
    inserted_ids = set(result.scalars().all())
    session.commit()

    # Here, we only consider related objects to the ones we inserted
    inserted_objs = [
        (obj, obj_id)
        for (obj, obj_id) in filtered_objs
        if obj_id in inserted_ids
    ]

    product_images = [
        {
            'url': url,
            'slug': get_image_slug(url),
            'product_id': obj_id
        }
        for obj, obj_id in inserted_objs
        for url in get_image_urls(obj)
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

    # They are unused :)
    related_products = []
    technical_details = []

    # If any of the products fail, then we have to
    session.bulk_insert_mappings(ProductImage, product_images)
    session.bulk_insert_mappings(ProductCategory, product_categories)
    session.bulk_insert_mappings(ProductFeature, product_features)
    session.bulk_insert_mappings(RelatedProduct, related_products)
    session.bulk_insert_mappings(TechnicalDetail, technical_details)
    session.commit()

    return len(inserted_objs)


def recreate_product_tables(dataset: str):
    """
    Drop and create all tables related to Product data. So, we
    can recreate tables correctly. Usually necessary before a full import
    """
    engine = create_engine(f"sqlite:///{dataset_db_path(dataset)}")
    tables = [x.__table__ for x in PRODUCT_TABLES]
    Base.metadata.drop_all(engine, tables)
    Base.metadata.create_all(engine, tables)


def load_metadata_into_db(dataset: str, force=False):
    """
    Loads all the product metadata into the DB. It will remove all previous
    products!
    """
    if force:
        recreate_product_tables(dataset)

    with create_session(dataset) as session:
        if session.query(Product).first():
            logger.info('There are records. Use force=True to force removal')
            return

        asins = set(
            session.execute(select(Review.asin).distinct()).scalars().all()
        )
        if len(asins) == 0:
            raise ValueError("No reviews in DB!")

    src = get_metafile(dataset)

    progress_options = {
        'unit': 'product', 'unit_scale': True, 'desc': 'Loading prods   '
    }

    with ThreadPoolExecutor(max_workers=1) as executor, \
            tqdm.tqdm(**progress_options) as progress, \
            create_session(dataset) as session, \
            gzip.open(src) as file, \
            jsonlines.Reader(file) as reader:
        # Here we read the total lines in parallel to not slow down
        # the file download
        def set_total():
            line_total = line_count_gzip(src)
            progress.total = line_total
        executor.submit(set_total)

        added = 0
        for chunk in chunked_iterator(reader, DB_CHUNK_SIZE):
            added += process_product_chunk(chunk, asins, session)
            progress.set_postfix_str(f'Added {added} products', refresh=False)
            progress.update(len(chunk))


def download_file(url: str, dest_file: Path):
    resp = requests.get(url, stream=True, verify=False)
    resp.raise_for_status()

    with open(dest_file, 'wb') as f:
        progress = tqdm.tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_file.name
        )

        if 'content-length' in resp.headers:
            progress.total = int(resp.headers['content-length'])

        for data in resp.iter_content(chunk_size=COPY_BUFSIZE):
            if data:
                progress.update(len(data))
                f.write(data)


def get_metafile(dataset: str) -> str:
    BASE_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    filename = f'meta_{dataset}.json.gz'
    if not filename or '/' in filename:
        raise ValueError('invalid filename')

    dest_file = BASE_DATA_FOLDER / filename
    if not os.path.exists(dest_file):
        download_file(f'{BASE_SOURCE_URL}/metaFiles/{filename}', dest_file)

    return dest_file


def get_categoryfile(dataset: str):
    BASE_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    filename = f'{dataset}.json.gz'
    if not filename or '/' in filename:
        raise ValueError('invalid filename')

    dest_file = BASE_DATA_FOLDER / filename
    if not os.path.exists(dest_file):
        download_file(f'{BASE_SOURCE_URL}/categoryFiles/{filename}', dest_file)

    return dest_file


def get_duplicated_product_list() -> str:
    BASE_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    filename = 'duplicates.txt'
    dest_file = BASE_DATA_FOLDER / filename

    if not os.path.exists(dest_file):
        download_file(f'{BASE_SOURCE_URL}/metaFiles/{filename}', dest_file)

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


def process_duplicate_product_chunk(
    chunk: List[Tuple[str, int]],
    session: Session
):
    chunk_asins = {}

    for line, _ in chunk:
        line_set = {asin for asin in line.split()}
        for asin in line_set:
            chunk_asins[asin] = line_set

    stmt = select(Review.asin).where(Review.asin.in_(chunk_asins)).distinct()
    relevant_asins = set(session.execute(stmt).scalars().all())

    filtered_chunk = {
        asin: chunk_asins[asin] & relevant_asins
        for asin in relevant_asins
        if len(chunk_asins[asin] & relevant_asins) > 1
    }
    if len(filtered_chunk) > 0:
        print(filtered_chunk)


def read_duplicate_products(dataset: str) -> List[Set[str]]:
    src = get_duplicated_product_list()

    total_lines = line_count(src)
    with open(src, 'r') as file, \
            tqdm.tqdm(unit='line', unit_scale=True, smoothing=0.01) as progress, \
            create_session(dataset, echo=False) as session:
        progress.total = total_lines
        progress.desc = 'Detect dups   '
        for chunk in chunked_iterator(file, DB_CHUNK_SIZE):
            # TODO: do something if we find duplicates :)
            process_duplicate_product_chunk(chunk, session)
            progress.update(len(chunk))


def get_image_slug(image_url: str) -> Optional[str]:
    image_re = re.compile(
        f'(?P<prefix>https:\/\/.*amazon.com\/images\/I\/'
        # The common name like 71vAyOySUqL.
        r'(?P<name>.*)\.)'
        # Dimension like _SR400_
        r'(?P<dimensions>_?((AC_)?(SX\d+_SY\d+_CR(,\d+)+_?)|(SR\d+,\d+_?)|'
        r'(SS\d+_?)|(US\d+_?)|(SY\d+)))'
        r'(?P<suffix>\.jpg)'
    )
    match = image_re.match(image_url)
    if match:
        return match.group("name")


def images_dir(dataset: str) -> Path:
    return BASE_DATA_FOLDER / f'{dataset}_product_images'


def get_image_url(slug: str, max_dimension: int = 400) -> str:
    return f'{IMAGE_PREFIX}{slug}._SS{max_dimension}_.jpg'


def image_webservice_url(asin: str, max_dimension: int = 400) -> str:
    return f"https://ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN={asin}&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=SS{max_dimension}"


def save_image_webservice(args: Tuple[int, str, str, Any]) -> Tuple[str, Optional[str]]:
    product_id, asin, dataset, thread_data = args
    url = image_webservice_url(asin)
    try:
        resp = thread_data.session.get(url, allow_redirects=True, stream=True)
    except ConnectionError as ex:
        # Some name resolution failure or stuff like that
        return str(ex), None

    if resp.status_code == 404: # Not found
        return url, None
    resp.raise_for_status()

    # After redirect
    url = resp.request.url
    slug = get_image_slug(url)
    if not slug:
        return url, None

    dest_dir = images_dir(dataset)
    dest_dir.mkdir(exist_ok=True)

    dest = dest_dir / f'{slug}.jpg'
    with dest.open('wb') as dest_file:
        for chunk in resp.iter_content(chunk_size=COPY_BUFSIZE):
            dest_file.write(chunk)

    with create_session(dataset) as sql_session:
        sql_session.add(ProductImage(
            product_id=product_id, url=url, slug=slug, main=True
        ))
        sql_session.commit()

    return url, slug


def products_with_no_images_query() -> Select:
    join_table = Product.__table__.join(
        ProductImage.__table__,
        (Product.id == ProductImage.product_id) & (ProductImage.main == True),
        isouter=True
    )
    query = select(Product).select_from(join_table)
    return query.where(ProductImage.id == None).distinct()

def products_with_no_main_image_df(dataset: str) -> pd.DataFrame:
    with create_session(dataset) as session:
        return pd.read_sql_query(
            products_with_no_images_query(),
            session.bind.connect(),
            index_col='id'
        )

def download_main_product_images_webservice(
    dataset: str, max_workers=IMAGE_DOWNLOAD_PROCESSES
):
    # This includes products with NO MAIN image and products with no image at all
    no_images = products_with_no_main_image_df(dataset)

    # Use a thread-exclussive request session to speed up image download
    thread_storage = threading.local()
    def init_thread_storage():
        thread_storage.session = requests.Session()

    pool = ThreadPool(processes=max_workers, initializer=init_thread_storage)

    with tqdm.tqdm(total=len(no_images), unit='image', smoothing=0.01) as progress:
        errors = 0
        args = [
            (row.Index, row.asin, dataset, thread_storage)
            for row in no_images.itertuples()
        ]
        for url, slug in pool.imap_unordered(save_image_webservice, args):
            if not slug:
                errors += 1
                progress.set_postfix_str(
                    f'Errors {errors} {url}',
                    refresh=False
                )
            progress.update()


def save_image_heuristic(
    args: Tuple[pd.DataFrame, str, Any]
) -> Tuple[int, Optional[int]]:
    image_group, dataset, thread_data = args

    dest_dir = images_dir(dataset)

    # sort_index ensures the first image (I guess the first image is the one that counts
    # gets tried first)
    for image_id, image in image_group.sort_index().iterrows():
        product_id = image['product_id']
        slug = image['slug']

        if slug is None:
            continue # Not interesting

        url = get_image_url(image['slug'])

        resp = thread_data.session.get(url, stream=True)
        dest = dest_dir / f'{slug}.jpg'

        if resp.status_code == 404:
            continue

        resp.raise_for_status()

        try:
            # We can retrieve the image :)
            with dest.open('wb') as dest_file:
                for chunk in resp.iter_content(chunk_size=COPY_BUFSIZE):
                    dest_file.write(chunk)

            # We can update the image and mark it as Main!
            with create_session(dataset) as sql_session:
                update_stmt = update(ProductImage)
                update_stmt = update_stmt.where(ProductImage.id == image_id)
                update_stmt = update_stmt.values(main=True)
                sql_session.execute(update_stmt)
                sql_session.commit()

                return product_id, image_id

        except KeyboardInterrupt:
            # Do not keep files if we are cancelling downloads
            dest.unlink(missing_ok=True)
            raise

    return product_id, None


def download_main_image_heuristic(
    dataset: str, max_workers=IMAGE_DOWNLOAD_PROCESSES
):
    """
    Add images for remaining products using the first product image.
    it can have several false positives, though
    """
    product_ids = products_with_no_images_query().with_only_columns(Product.id)
    product_images = select(ProductImage).where(
        ProductImage.product_id.in_(product_ids)
    )

    with create_session(dataset) as session:
        product_images = pd.read_sql_query(product_images,
            session.bind.connect(),
            index_col='id'
        )

    # All those product should not have main images
    assert len(product_images.loc[product_images['main'] == True]) == 0

    no_images_by_product = product_images.groupby('product_id')


    # Use a thread-exclussive request session to speed up image download
    thread_storage = threading.local()
    def init_thread_storage():
        thread_storage.session = requests.Session()

    pool = ThreadPool(processes=max_workers, initializer=init_thread_storage)

    args = [
        (image_group, dataset, thread_storage)
        for _, image_group in no_images_by_product
    ]
    with tqdm.tqdm(
        total=len(no_images_by_product), unit='product', smoothing=0.01
    ) as progress:
        errors = 0
        for product_id, image_id in pool.imap_unordered(save_image_heuristic, args):
            if image_id is None:
                errors += 1
                progress.set_postfix_str(
                    f'Errors {errors} {product_id=}', refresh=False
                )
            progress.update()


def check_all_images_are_ok(dataset: str):
    product_images = product_images_df(dataset)

    # Slugs stored in the DB which we will expect
    main_slugs = product_images.loc[product_images['main'] == True]['slug']

    # Slugs in the images folder
    downloaded_slugs = [
        p.stem
        for p in images_dir(dataset).glob('*.jpg')
    ]

    assert set(main_slugs) == set(downloaded_slugs)


def process_review_chunk(
        chunk: List[Tuple[Dict, int]],
        min_date: date,
        max_date: date,
        session: Session) -> int:
    """
    Adds a chunk of reviews depending on the amount of reviews we added
    """
    filtered_chunk = [
        (obj, obj_id)
        for obj, obj_id in chunk
        if min_date <= date.fromtimestamp(obj['unixReviewTime']) < max_date
    ]

    reviews = [
        {
            'id': obj_id,
            'asin': obj['asin'],
            'reviewerID': obj['reviewerID'],
            'reviewerName': obj.get('reviewerName'),
            'overall': obj['overall'],
            'text': obj.get('reviewText'),
            'reviewTime': date.fromtimestamp(obj['unixReviewTime']),
            'summary': obj.get('summary'),
            'verified': obj['verified'],
            'vote': obj.get('vote'),
        }
        for obj, obj_id in filtered_chunk
    ]
    review_images = [
        {'url': image_url, 'review_id': obj_id}
        for obj, obj_id in filtered_chunk
        for image_url in obj.get('image', [])
    ]
    review_style = [
        {'name': k, 'value': v, 'review_id': obj_id}
        for obj, obj_id in filtered_chunk
        for k, v in obj.get('style', {}).items()
    ]

    session.bulk_insert_mappings(Review, reviews)
    session.bulk_insert_mappings(TempReviewImage, review_images)
    session.bulk_insert_mappings(TempReviewStyle, review_style)
    session.commit()

    # How many reviews we actually added
    return len(filtered_chunk)


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


def recreate_reviews_tables(dataset: str):
    """
    Drop and create all tables related to review data (only review). So, we
    can recreate tables correctly
    """
    engine = create_engine(f"sqlite:///{dataset_db_path(dataset)}")
    tables = [x.__table__ for x in REVIEW_TABLES]
    Base.metadata.drop_all(engine, tables)
    Base.metadata.create_all(engine, tables)


def drop_review_indexes(dataset: str):
    with create_session(dataset) as session:
        for idx in Review.__table__.indexes:
            session.execute(DropIndex(idx, if_exists=True))


def create_review_indexes(dataset: str):
    with create_session(dataset) as session:
        for idx in Review.__table__.indexes:
            session.execute(CreateIndex(idx, if_not_exists=True))


def load_reviews_into_db(
        dataset: str,
        force: bool = False,
        min_date: date = date.min,
        max_date: date = date.max,
):
    initialize_db(dataset)
    if force:
        recreate_reviews_tables(dataset)

    with create_session(dataset) as session:
        if session.query(Review).first():
            raise ValueError(
                'There are reviews. Use force=True to force removal'
            )

    src = get_categoryfile(dataset)

    progress_options = {
        'unit': 'review', 'unit_scale': True, 'smoothing': 0.01,
        'desc': 'Loading reviews '
    }

    drop_review_indexes(dataset)

    with ThreadPoolExecutor(max_workers=1) as executor, \
            tqdm.tqdm(**progress_options) as progress, \
            create_session(dataset) as session, \
            gzip.open(src) as file, \
            jsonlines.Reader(file) as reader:

        # Here we read the total lines in parallel to not slow down
        # the file download
        def set_total():
            line_total = line_count_gzip(src)
            progress.total = line_total

        executor.submit(set_total)

        added = 0

        for chunk in chunked_iterator(reader, DB_CHUNK_SIZE):
            progress.set_postfix_str(f'Added {added} reviews', refresh=False)
            added += process_review_chunk(
                chunk,
                session=session,
                min_date=min_date,
                max_date=max_date
            )
            progress.update(len(chunk))

    create_review_indexes(dataset)


def find_k_core(
        dataset: str,
        min_reviews_per_reviewer: int = 1,
        min_reviews_per_asin: int = 1,
        iterations=1):
    """
    Deletes reviews and products to ensure all reviewers and products have
    at least a minimum amount (bipartite graph k-core)

    Ideally we should run it multiple iterations, but in experiments
    people usually only run it once
    """
    with create_session(dataset) as session:
        # This makes the thing run MUCH faster
        session.execute(text("PRAGMA foreign_keys=OFF;"))
        for i in range(iterations):
            deleted_asins = delete_products_with_few_reviews(
                session, min_reviews_per_asin
            )
            deleted_reviews = delete_reviewers_with_few_reviews(
                session, min_reviews_per_reviewer
            )
            session.commit()

            if deleted_asins == 0 and deleted_reviews == 0:
                break


def reviewer_count(dataset: str) -> int:
    with create_session(dataset) as session:
        stmt = select(func.count(Review.reviewerID.distinct()))
        return session.execute(stmt).scalar_one()


def asin_count(dataset: str) -> int:
    with create_session(dataset) as session:
        stmt = select(func.count(Review.asin.distinct()))
        return session.execute(stmt).scalar_one()


def dataset_count(dataset: str) -> int:
    with create_session(dataset) as session:
        stmt = select(func.count(Review.id))
        return session.execute(stmt).scalar_one()


def dataset_density(dataset: str) -> float:
    with create_session(dataset) as session:
        stmt = select(
            func.count(Review.id) / (
                func.count(Review.reviewerID.distinct()) *
                func.count(Review.asin.distinct())
            )
        )
        return session.execute(stmt).scalar_one()


def log_dataset_metrics(dataset: str) -> float:
    reviewers = reviewer_count(dataset)
    asins = asin_count(dataset)
    length = dataset_count(dataset)
    density = dataset_density(dataset)

    logger.info(f'Total reviews: {length} Reviewers: {reviewers} '
                f'Products: {asins} Density: {100*density:.4f}%')


def move_dependent_review_items(dataset: str):
    """
    Moves reviewImage and reviewStyles from "temporary" tables to the
    final tables.

    Useful because we delete many review records during `find_k_core`. Its
    faster to copy a minor portion than removing records from a table
    """
    with create_session(dataset) as session:
        # Copy reviewImage
        session.execute(
            ReviewImage.__table__.insert().from_select(
                TempReviewImage.__table__.columns,
                select(TempReviewImage.__table__.columns)
                .where(TempReviewImage.review_id.in_(select(Review.id)))
            )
        )
        session.execute(DropTable(TempReviewImage.__table__))

        # Copy reviewStyle
        session.execute(
            ReviewStyle.__table__.insert().from_select(
                TempReviewStyle.__table__.columns,
                select(TempReviewStyle.__table__.columns)
                .where(TempReviewStyle.review_id.in_(select(Review.id)))
            )
        )
        session.execute(DropTable(TempReviewStyle.__table__))
        session.commit()


def delete_reviewers_with_few_reviews(session: Session, min_amount: int):
    """
    Removes from the DB the review records corresponding to reviewers with
    less than a minumum amount. This generates the "k-core" graph
    """
    select_cond = (
        select(Review.reviewerID)
        .group_by(Review.reviewerID)
        .having(func.count() < min_amount)
    )

    # Counting necessary removals
    count_del = session.execute(
        select(func.count(Review.id)).where(Review.reviewerID.in_(select_cond))
    ).scalar()

    progress_options = {
        'unit_scale': True, 'unit': 'row', 'total': count_del,
        'desc': 'Deleting reviews', 'disable': count_del < 10_000
    }
    with tqdm.tqdm(**progress_options) as progress:
        result = session.execute(
            select(Review.id).where(Review.reviewerID.in_(select_cond))
        ).scalars().partitions(DB_CHUNK_SIZE)
        for ids in result:
            session.execute(delete(Review).where(Review.id.in_(ids)))

            progress.update(len(ids))

    return count_del


def delete_products_with_few_reviews(session: Session, min_amount: int):
    """
    Removes from the DB the review records corresponding to products with
    less than a minumum amount. This generates the “k-core” graph
    """
    select_cond = (
        select(Review.asin)
        .group_by(Review.asin)
        .having(func.count() < min_amount)
    )

    # Counting necessary removals
    count_del = session.execute(
        select(func.count(Review.id)).where(Review.asin.in_(select_cond))
    ).scalar()

    progress_options = {
        'unit_scale': True, 'unit': 'row', 'total': count_del,
        'desc': 'Deleting asins  ', 'disable': count_del < 10_000
    }
    with tqdm.tqdm(**progress_options) as progress:
        result = session.execute(
            select(Review.id).where(Review.asin.in_(select_cond))
        ).scalars().partitions(DB_CHUNK_SIZE)
        for ids in result:
            session.execute(delete(Review).where(Review.id.in_(ids)))

            progress.update(len(ids))

    return count_del

def delete_non_relevant_images(dataset: str):
    """Delete duplicate and non main images"""
    with create_session(dataset) as session:
        # Non main are not important
        print('Deleting non main product images')
        delete_non_main = delete(ProductImage).where(ProductImage.main != True)
        session.execute(delete_non_main)

        # Products with duplicate slug and product id, and main
        # should not happen but we delete them anyway
        print('Getting duplicated product images')
        product_image2 = aliased(ProductImage)
        duplicate_product_images_ids = select(ProductImage.id).where(
            (ProductImage.id > product_image2.id) &
            (ProductImage.product_id == product_image2.product_id)
        ).distinct()
        product_ids = [
            row[0]
            for row in session.execute(duplicate_product_images_ids).all()
        ]

        print('Deleting duplicated')
        delete_duplicated = delete(ProductImage).where(
            ProductImage.id.in_(product_ids)
        )
        session.execute(delete_duplicated)
        session.commit()


def vacuum_dataset(dataset: str):
    with create_session(dataset) as session:
        session.execute(text("VACUUM"))


def load_amazon_dataset(
    dataset: str,
    force: bool = False,
    min_date: date = date.min,
    max_date: date = date.max,
    min_reviews_per_reviewer: int = 1,
    min_reviews_per_asin: int = 1,
    kcore_iterations: int = 1,
):
    # Filter out reviews when loading
    load_reviews_into_db(dataset,
        force=force,
        min_date=min_date,
        max_date=max_date
    )
    # Remove irrelevant data
    find_k_core(
        dataset,
        min_reviews_per_reviewer=min_reviews_per_reviewer,
        min_reviews_per_asin=min_reviews_per_asin,
        iterations=kcore_iterations
    )
    # (re)move unecessary review images/styles
    move_dependent_review_items(dataset)
    # Only load necessary asins
    load_metadata_into_db(dataset, force=force)
    # To reduce space
    vacuum_dataset(dataset)
    # Totally optional
    log_dataset_metrics(dataset)


def reviews_df(dataset: str, limit: Optional[int] = None) -> pd.DataFrame:
    with create_session(dataset) as session:
        query = session.query(Review)
        if limit is not None:
            query = query.limit(100)
        stmt = query.statement

    result = pd.read_sql_query(stmt, session.bind.connect())

    # Add columns to be compatible with other datasets
    result['user_id'] = result['reviewerID']
    result['item_id'] = result['asin']
    result['rating'] = result['overall']

    return result


def split_line_str(s: Optional[str], separator: str) -> List[str]:
    """Splits a (nullable) string with a separator"""
    if s is None:
        return []
    else:
        return s.split(separator)

def items_df(dataset: str, limit: Optional[int] = None) -> pd.DataFrame:
    # We use as sparator something we don't use anywhere else (ESCAPE (U+001B))
    sep = '\x1b'

    images = select(
            ProductImage.product_id,
            func.group_concat(ProductImage.slug, sep).label('image_slug'),
            func.group_concat(ProductImage.url, sep).label('image_url')
    ).group_by(ProductImage.product_id)

    product_features = select(
        ProductFeature.product_id,
        func.group_concat(ProductFeature.name, sep).label('feature')
    ).group_by(ProductFeature.product_id)

    product_categories = select(
        ProductCategory.product_id,
        func.group_concat(ProductCategory.name, sep).label('category')
    ).group_by(ProductCategory.product_id)

    tech_detail = select(
        TechnicalDetail.product_id,
        func.group_concat(func.json_object(
            'kind', TechnicalDetail.kind,
            'name', TechnicalDetail.name,
            'value', TechnicalDetail.value
        ), sep).label('tech_detail')
    ).group_by(TechnicalDetail.product_id)

    from_table = Product.__table__
    for join_cte in [images, product_features, product_categories, tech_detail]:
        from_table = from_table.join(
            join_cte,
            Product.id == join_cte.c.product_id,
            isouter=True
    )

    query = select(
        Product,
        images.c.image_slug,
        images.c.image_url,
        product_features.c.feature,
        product_categories.c.category,
        tech_detail.c.tech_detail,
    ).select_from(from_table)

    if limit is not None:
        query = query.limit(limit)

    with create_session(dataset) as session:
        df = pd.read_sql_query(query, session.bind.connect(), index_col='id')

        for col in ['image_url', 'image_slug', 'feature', 'category']:
            df[col] = df[col].apply(split_line_str, args=(sep,))

        # For compatibility with other datasets
        df['item_id'] = df['asin']

        return df


def product_images_df(dataset: str, limit: Optional[int] = None) -> pd.DataFrame:
    query = select(ProductImage)

    if limit is not None:
        query = query.limit(limit)

    with create_session(dataset) as session:
        return pd.read_sql_query(query, session.bind.connect(), index_col='id')

def product_categories_df(dataset: str, limit: Optional[int] = None) -> pd.DataFrame:
    query = select(ProductCategory)

    if limit is not None:
        query = query.limit(limit)

    with create_session(dataset) as session:
        return pd.read_sql_query(query, session.bind.connect(), index_col='id')
