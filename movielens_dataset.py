
import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
from shutil import COPY_BUFSIZE
import threading
from typing import NamedTuple, Optional, Tuple

import requests
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
import pandas as pd
from bs4 import BeautifulSoup

BASE_DATA_FOLDER = Path('data/movielens/')
IMG_FOLDER = BASE_DATA_FOLDER / 'images'
# Original ML-1M dataset
ML_1M_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

# Generated links to from MovieLens movies IMDB
ML_1M_IMDB = 'https://raw.githubusercontent.com/vectorsss/movielens_100k_1m_extension/main/data/ml-1m/links_artificial.csv'

# Fallback CSV with images (low quality)
ML_1M_CSV_IMAGES = 'https://raw.githubusercontent.com/antonsteenvoorden/ml1m-images/master/ml1m_images.csv'

DONWLOAD_PROCESSES = 8

IMDB_DATA_FILE = BASE_DATA_FOLDER / 'imdb_data.pkl'
ML1M_REVIEWS_FILE = BASE_DATA_FOLDER / 'ml-1m/reviews.pkl'

SOME_UA = 'Mozilla/5.0 (Linux; Android 7.0) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Focus/1.0 Chrome/59.0.3029.83 Mobile Safari/537.36'

def download_file(url: str, dest_file: Path, show_progress = True):
    resp = requests.get(url, stream=True)
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

def download_movielens_dataset(overwrite: bool = False):
    BASE_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    dest_folder = BASE_DATA_FOLDER / 'ml-1m'
    if not dest_folder.exists() or overwrite:
        with NamedTemporaryFile(suffix='_ml-1m.zip') as temp_zipfile:
            download_file(ML_1M_URL, Path(temp_zipfile.name))
            with ZipFile(temp_zipfile.name, 'r') as temp_zipfile_obj:
                print(f'Extracting {temp_zipfile.name} to {BASE_DATA_FOLDER}')
                temp_zipfile_obj.extractall(BASE_DATA_FOLDER)

    movies_file = BASE_DATA_FOLDER / 'ml-1m/movies.dat'
    movies_df = pd.read_csv(
        movies_file,
        sep='::',
        encoding='latin-1',
        engine='python',
        names=['movie_id', 'title', 'genres'],
        index_col='movie_id'
    )
    movies_df['genres'] = movies_df['genres'].str.split('|')

    # Movielens have the title and year in the same column
    # (e.g. 'Toy Story (1995)')
    title_regex = r'^(.*)\s*\((\d{4})\)\s*$'
    movies_df[['title', 'year']] = movies_df['title'].str.extract(title_regex)

    original_title_regex = r'^(.*)\s*(\((.*)\))?\s*$'
    original_title_df = movies_df['title'].str.extract(original_title_regex)
    # The second and third captures can be empty
    movies_df[['title', 'original_title']] = original_title_df[[0,2]]

    return movies_df


def download_artificial_links(overwrite: bool = False):
    dest_file = BASE_DATA_FOLDER / 'ml-1m' / 'links_artificial.csv'
    if not dest_file.exists() or overwrite:
        download_file(
            ML_1M_IMDB,
            BASE_DATA_FOLDER / 'ml-1m' / 'links_artificial.csv'
        )

    links_artificial = pd.read_csv(
        BASE_DATA_FOLDER / 'ml-1m/links_artificial.csv',
        usecols=['movie_id', 'imdbId'],
        index_col='movie_id'
    )
    return links_artificial.rename(columns={'imdbId': 'imdb_id'})

def items_df() -> pd.DataFrame:
    movies_df = download_movielens_dataset()
    artificial_links = download_artificial_links()

    movies_df_linked = movies_df.join(artificial_links, validate='1:1')
    imdb_data = download_imdb_data(movies_df_linked['imdb_id'])
    result =  movies_df_linked.join(imdb_data, validate='1:1')

    # Rename summary to description to match the schema
    result.rename(columns={'summary': 'description'}, inplace=True)
    result.reset_index(inplace=True)

    # Add the item id for compatibility with other datasets
    result['item_id'] = result['movie_id'].astype(str)
    return result


def find_image_url(soup: BeautifulSoup) -> Optional[str]:
    og_image_meta = soup.find('meta', { 'property': "og:image" })
    return og_image_meta['content']


def find_synopsis(soup: BeautifulSoup) -> Optional[str]:
    synopsis = soup.find('div', {'data-testid': 'sub-section-synopsis'})
    if synopsis:
        return synopsis.text


def find_summary(soup: BeautifulSoup) -> Optional[str]:
    summaries_parent = soup.find(
        'div', {'data-testid': 'sub-section-summaries'}
    )
    if summaries_parent is None:
        return None
    summaries =  summaries_parent.find_all(
        'div', { 'class': 'ipc-html-content-inner-div' }
    )
    if len(summaries) == 0:
        return None

    summary_idx = 1 if len(summaries) > 1 else 0

    return summaries[summary_idx].text

class MovieDataResult(NamedTuple):
    movie_id: int
    image_url: Optional[str] = None
    synopsis: Optional[str] = None
    summary: Optional[str] = None


def download_movie_data(
        args: Tuple[int, int, threading.local]
    ) -> MovieDataResult:
    movie_id, imdb_id, thread_data = args

    url = f'https://www.imdb.com/title/tt{imdb_id:07d}/plotsummary/'
    resp = thread_data.session.get(url)

    if resp.status_code == 404:
        # Ignore 404 errors for now
        image_url = synopsis = summary = None
    else:
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        image_url=find_image_url(soup)
        synopsis=find_synopsis(soup)
        summary=find_summary(soup)

    if image_url is not None and not image_url.endswith('imdb_logo.png'):
        IMG_FOLDER.mkdir(exist_ok=True)
        dest_file = IMG_FOLDER / f'{movie_id}.jpg'
        download_file(image_url, dest_file, show_progress=False)

    return MovieDataResult(
        movie_id=movie_id,
        image_url=image_url,
        synopsis=synopsis,
        summary=summary
    )


def init_result_df(imdb_ids: pd.Series, resume: bool) -> pd.DataFrame:
    result_df = pd.DataFrame(index=imdb_ids.index)
    result_df['imdb_id'] = imdb_ids
    result_df['processed'] = False

    if IMDB_DATA_FILE.exists() and resume:
        result_df = pd.read_pickle(IMDB_DATA_FILE)

    return result_df

def download_imdb_data(
        imdb_ids: pd.Series,
        max_workers=DONWLOAD_PROCESSES,
        resume=True
    ):
    result_df = init_result_df(imdb_ids, resume)

    pending_imdb_id = result_df.loc[~result_df['processed']]['imdb_id']
    if len(pending_imdb_id) == 0:
        return result_df.drop(columns=['processed', 'imdb_id'])

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
            tqdm(total=len(pending_imdb_id), unit='movie') as progress:
        args = [
            (movie_id, imdb_id, thread_storage)
            for (movie_id, imdb_id) in pending_imdb_id.items()
        ]
        async_results = pool.imap_unordered(download_movie_data, args)

        for process_result in async_results:
            progress.update()
            # Write the result in the correct row
            movie_id = process_result.movie_id
            result_df.at[movie_id, 'synopsis'] = process_result.synopsis
            result_df.at[movie_id, 'summary'] = process_result.summary
            result_df.at[movie_id, 'image_url'] = process_result.image_url
            result_df.at[movie_id, 'processed'] = True

            # Persist the result in case of interruption
            result_df.to_pickle(IMDB_DATA_FILE)

    # Drop unnecessary columns
    return result_df.drop(columns=['processed', 'imdb_id'])


def reviews_df() -> pd.DataFrame:
    if ML1M_REVIEWS_FILE.exists():
        return pd.read_pickle(ML1M_REVIEWS_FILE)

    result = pd.read_csv(
        BASE_DATA_FOLDER / 'ml-1m/ratings.dat',
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
    )
    # Transform timestamp to datetime from epoch seconds
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s')

    # It is much faster to read the pickle file directly instead of the DAT
    # file, so we save it for later
    result.to_pickle(ML1M_REVIEWS_FILE)

    return result
