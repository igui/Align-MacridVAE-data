
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
    items_df = raw_items_df()
    download_images(items_df, max_workers=max_workers, resume=resume)

    # Filter out items that do not have an image
    valid_images = [
        (x.stem, x.stat().st_size >= 1000) for x in
        IMAGE_FOLDER.glob('*.jpg')
    ]
    has_image = pd.Series(
        [size for (_, size) in valid_images],
        index=[isbn for (isbn, _) in valid_images]
    )
    filtered_items = items_df.loc[has_image]

    reviews_df = raw_reviews_df()

    # Do not consider reviews for items that are not in the items_df
    filtered_reviews = reviews_df.loc[
        reviews_df['item_id'].isin(filtered_items.index)
    ]

    # Also, do not consider items that are not in the reviews_df
    filtered_items = filtered_items.filter(
        filtered_reviews['item_id'].unique(), axis='index'
    )


    filtered_reviews.to_pickle(REVIEWS_PROCESSED_FILE)
    filtered_items.to_pickle(ITEMS_PROCESSED_FILE)


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

    return isbn, dest_file

def download_images(
    items: pd.DataFrame,
    max_workers=DOWNLOAD_PROCESSES,
    resume=True
    ):
    IMAGE_FOLDER.mkdir(exist_ok=True)
    downloaded_isbns = set()
    if resume:
        downloaded_isbns = set((
            f.stem for f in IMAGE_FOLDER.glob('*.jpg')
        ))

    pending_items = items.loc[~items['ISBN'].isin(downloaded_isbns)]
    if pending_items.empty:
        return

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
            progress.update()

    # Truncate images that are too small. They correspond to missing images
    for f in IMAGE_FOLDER.glob('*.jpg'):
        if f.stat().st_size < 1000:
            fp = open(f, 'wb')
            fp.close()
