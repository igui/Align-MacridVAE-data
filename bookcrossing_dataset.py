
import csv
from multiprocessing.pool import ThreadPool
from pathlib import Path
from shutil import COPY_BUFSIZE
import threading
from typing import Optional, Tuple

import requests
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
import pandas as pd

BASE_DATA_FOLDER = Path('data/bookcrossing/')

# Original datasets
CSV_DUMP_URL = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'

IMAGE_FOLDER = BASE_DATA_FOLDER / 'images'
BOOKS_CSV_FILE = BASE_DATA_FOLDER / 'BX-Books.csv'
ITEMS_PROCESSED_FILE = BASE_DATA_FOLDER / 'items_processed.pickle'
RATINGS_CSV_FILE = BASE_DATA_FOLDER / 'BX-Book-Ratings.csv'
REVIEWS_PROCESSED_FILE = BASE_DATA_FOLDER / 'ratings_processed.pickle'

DOWNLOAD_PROCESSES = 8

SOME_UA = 'Mozilla/5.0 (Linux; Android 7.0) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Focus/1.0 Chrome/59.0.3029.83 Mobile Safari/537.36'


MIN_IMAGE_SIZE = 1000
MIN_REVIEW_COUNT = 5

def download_file(
        url: str,
        dest_file: Path, show_progress = True,
        session: Optional[requests.Session] = None
    ):
    if session is None:
        resp = requests.get(url, stream=True)
    else:
        resp = session.get(url, stream=True)

    resp.raise_for_status()

    with open(dest_file, 'wb') as f:
        if show_progress:
            progress = tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest_file.name
            )

        if show_progress and 'content-length' in resp.headers:
            progress.total = int(resp.headers['content-length'])

        for data in resp.iter_content(chunk_size=COPY_BUFSIZE):
            if data and show_progress:
                    progress.update(len(data))
            if data:
                f.write(data)


def download_dataset(overwrite: bool = False):
    BASE_DATA_FOLDER.mkdir(exist_ok=True, parents=True)

    csv_dump_file = BASE_DATA_FOLDER / 'BX-Books.csv'
    if csv_dump_file.exists() and not overwrite:
        return

    with NamedTemporaryFile(suffix=f'_bookcrossing.zip') as temp_zipfile:
        download_file(CSV_DUMP_URL, Path(temp_zipfile.name))
        with ZipFile(temp_zipfile.name, 'r') as temp_zipfile_obj:
            print(f'Extracting {temp_zipfile.name} to {BASE_DATA_FOLDER}')
            temp_zipfile_obj.extractall(BASE_DATA_FOLDER)


def process_dataset(max_workers=DOWNLOAD_PROCESSES, resume=True):
    download_dataset()
    print('Reading items...')
    raw_items = raw_items_df()
    print('Downloading images...')
    image_sizes = download_images(
        raw_items, max_workers=max_workers, resume=resume
    )
    raw_items['image_size'] = image_sizes
    raw_reviews = raw_reviews_df()

    print('Filtering items and reviews...')
    filtered_items, filtered_reviews = filter_items_and_reviews(
        raw_items, raw_reviews
    )

    filtered_items.to_pickle(ITEMS_PROCESSED_FILE)
    filtered_reviews.to_pickle(REVIEWS_PROCESSED_FILE)


def filter_items_and_reviews(items_df, reviews_df):
    # Filter out items that do not have an image
    items_df = only_items_with_image(items_df)

    # Do not consider reviews for items that are not in the items_df
    reviews_df = reviews_df.loc[
        reviews_df['item_id'].isin(items_df.index)
    ]

    # Filter out items with less than 5 reviews or not in items_df
    reviews_df = only_with_enough_reviews(reviews_df)

    # Do not consider items that are not in the reviews_df
    items_df = items_df.filter(
        reviews_df['item_id'].unique(), axis='index'
    )

    return items_df, reviews_df


def only_items_with_image(items_df):
    item_has_image = items_df.loc[items_df['image_size'] >= MIN_IMAGE_SIZE].index
    return items_df.loc[item_has_image]


def only_with_enough_reviews(reviews_df):
    item_id_sizes = reviews_df.groupby('item_id').size()
    items_with_enough_reviews = item_id_sizes.loc[item_id_sizes >= MIN_REVIEW_COUNT].index

    # Filter out items with less than 5 reviews or not in items_df
    reviews_df = reviews_df.loc[
        reviews_df['item_id'].isin(items_with_enough_reviews)
    ]

    user_id_sizes = reviews_df.groupby('user_id').size()
    users_with_enough_reviews = user_id_sizes.loc[user_id_sizes >= MIN_REVIEW_COUNT].index
    reviews_df = reviews_df.loc[
        reviews_df['user_id'].isin(users_with_enough_reviews)
    ]

    return reviews_df


def items_df():
    return pd.read_pickle(ITEMS_PROCESSED_FILE)


def raw_items_df():
    res = pd.read_csv(
        BOOKS_CSV_FILE,
        quoting=csv.QUOTE_ALL,
        sep=';',
        escapechar='\\',
        encoding='latin-1',
        quotechar='"',
        )
    res['item_id'] = res['ISBN']
    res['title'] = res['Book-Title']
    res['description'] = None
    res['image_slug'] = res['ISBN'].apply(lambda x: [ x ])
    res.set_index('item_id', inplace=True)
    # Just for compatibility with other datasets
    res['item_id'] = res.index
    return res


def raw_reviews_df():
    res = pd.read_csv(
        RATINGS_CSV_FILE,
        quoting=csv.QUOTE_ALL,
        sep=';',
        escapechar='\\',
        encoding='latin-1',
        quotechar='"'
    )
    # Normalize ratings to 0-5 for compatibility with other datasets
    res['Book-Rating'] = 5 * res['Book-Rating'] / res['Book-Rating'].max()

    return res.rename(columns={
        'User-ID': 'user_id',
        'ISBN': 'item_id',
        'Book-Rating': 'rating'
    })


def reviews_df():
    return pd.read_pickle(REVIEWS_PROCESSED_FILE)


def download_item_image(args: Tuple[str, str, threading.local]):
    isbn, url, thread_storage = args
    dest_file = IMAGE_FOLDER / f'{isbn}.jpg'

    session = thread_storage.session
    # print(f'Downloading {url} to {image_file}')
    if session is None:
        resp = requests.get(url, stream=True)
    else:
        resp = session.get(url, stream=True)

    if resp.status_code == 404:
        # We don't have an image for this item
        fp = open(dest_file, 'wb')
        fp.close()
        return isbn, None

    resp.raise_for_status()

    with open(dest_file, 'wb') as f:
        for data in resp.iter_content(chunk_size=COPY_BUFSIZE):
            if data:
                f.write(data)

    # Many images are missing or are single pixels. Truncate them
    if dest_file.stat().st_size < 1000:
        fp = open(f, 'wb')
        fp.close()

    return isbn, dest_file


def list_downloaded_images() -> pd.Series:
    downloaded_images_arr = [
        (x.stem, x.stat().st_size) for x in
        IMAGE_FOLDER.glob('*.jpg')
    ]
    return pd.Series(
        [size for (_, size) in downloaded_images_arr],
        index=[isbn for (isbn, _) in downloaded_images_arr]
    )


def download_images(
    items: pd.DataFrame,
    max_workers=DOWNLOAD_PROCESSES,
    resume=True
    ):
    IMAGE_FOLDER.mkdir(exist_ok=True)

    local_images = list_downloaded_images()

    if not resume:
        for stem in local_images.index:
            (IMAGE_FOLDER / f'{stem}.jpg').unlink()
        local_images = pd.Series()

    pending_items = items.loc[~items['ISBN'].isin(local_images.index)]
    if pending_items.empty:
        return local_images

    # Use a thread-exclussive request session to speed up image download
    thread_storage = threading.local()
    def init_thread_storage():
        session = requests.Session()
        session.headers.update({
            'User-Agent': SOME_UA,
            'Accept-Language': 'en-US,en;q=0.8'
        })
        thread_storage.session = session

    with ThreadPool(processes=max_workers, initializer=init_thread_storage) as pool, \
            tqdm(total=len(pending_items), unit='book', smoothing=0) as progress:
        args = [
            (row['ISBN'], row['Image-URL-L'], thread_storage)
            for _, row in pending_items.iterrows()
        ]
        async_results = pool.imap_unordered(download_item_image, args)

        errors = 0
        for url, dest_file in async_results:
            if dest_file is None:
                errors += 1
                progress.set_postfix_str(
                    f'Errors {errors} {url=}', refresh=False
                )
            local_images[dest_file.stem] = dest_file.stat().st_size
            progress.update()

    return local_images
