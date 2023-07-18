
from multiprocessing.pool import ThreadPool
from pathlib import Path
import re
from shutil import COPY_BUFSIZE
import threading
from typing import Literal, NamedTuple, Optional, Tuple
import gdown

import requests
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
import pandas as pd
from bs4 import BeautifulSoup

BASE_DATA_FOLDER = Path('data/movielens/')

# Original datasets
ML_1M_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
ML_25M_URL = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'

# Generated links to from MovieLens movies IMDB
ML_1M_IMDB = 'https://raw.githubusercontent.com/vectorsss/movielens_100k_1m_extension/main/data/ml-1m/links_artificial.csv'

# Another set of generated links to from MovieLens movies IMDB
ML_25M_IMDB = 'https://drive.google.com/uc?id=1fz8WjLy0_UYioFbMirYrjhM00EYnCaWP'

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


def fix_movie_title(original_title: str) -> str:
    # Remove original title
    without_original_title = re.sub(r'\s*\(.*\)\s*$', '', original_title)

    # Move 'Article' to the end
    res = re.sub(r'^(.*), \s*(\w+)\s*$', r'\2 \1', without_original_title)
    return res


def download_movielens_dataset(dataset: str, overwrite: bool = False):
    BASE_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    dest_folder = BASE_DATA_FOLDER / dataset
    if not dest_folder.exists() or overwrite:
        with NamedTemporaryFile(suffix=f'_{dataset}.zip') as temp_zipfile:
            download_file(ML_1M_URL, Path(temp_zipfile.name))
            with ZipFile(temp_zipfile.name, 'r') as temp_zipfile_obj:
                print(f'Extracting {temp_zipfile.name} to {BASE_DATA_FOLDER}')
                temp_zipfile_obj.extractall(BASE_DATA_FOLDER)

    if dataset == 'ml-1m':
        movies_file = BASE_DATA_FOLDER / dataset / 'movies.dat'
        movies_df = pd.read_csv(
            movies_file,
            sep='::',
            encoding='latin-1',
            engine='python',
            names=['movie_id', 'title', 'genres'],
        )
    else:
        movies_file = BASE_DATA_FOLDER / dataset / 'movies.csv'
        movies_df = pd.read_csv(
            movies_file,
        )
        # To be compatible with ML-1m
        movies_df.rename(columns={'movieId': 'movie_id'}, inplace=True)

    movies_df.set_index('movie_id', inplace=True)
    movies_df['genres'] = movies_df['genres'].str.split('|')

    # Movielens have the title and year in the same column
    # (e.g. 'Toy Story (1995)')
    title_regex = r'^(.*)\s*\((\d{4})\)\s*$'
    movies_df[['title', 'year']] = movies_df['title'].str.extract(title_regex)

    movies_df['title'] = movies_df['title'].fillna('Unknown').apply(fix_movie_title)

    return movies_df


def download_artificial_links(dataset: str, overwrite: bool = False):
    if dataset == 'ml-1m':
        return download_imdb_data_1m(overwrite)
    elif dataset == 'ml-25m':
        return download_imdb_data_25m(overwrite)
    else:
        raise ValueError(f'Unknown dataset {dataset}')


def download_imdb_data_1m(overwrite: bool = False):
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

    # Use the canonical name for the IMDB id "1234" to "tt0001234"
    # TODO: check this
    links_artificial['imdbId'] = (
        'tt' + links_artificial['imdbId'].astype(str).str.zfill(7)
    )

    return links_artificial.rename(columns={'imdbId': 'imdb_id'})


def download_imdb_data_25m(overwrite: bool = False):
    dest_file = BASE_DATA_FOLDER / 'ml-25m-imdb/movie_ml_imdb.csv'
    if not dest_file.exists() or overwrite:
        with NamedTemporaryFile(suffix='_ml-25m.zip', delete=False) as temp_zipfile:
            gdown.download(ML_25M_IMDB, temp_zipfile.name, quiet=False)
            ZipFile(temp_zipfile.name, 'r').extractall(BASE_DATA_FOLDER)

    res = pd.read_csv(dest_file, sep='\t')

    # Make it compatible with the other artificial links file for ML-1M
    res.rename(
        columns={
            'tconst': 'imdb_id', 'movieId': 'movie_id'
        },
        inplace=True
    )
    res.set_index('movie_id', inplace=True)

    return res

def items_df(dataset: Literal['ml-1m', 'ml-25m']) -> pd.DataFrame:
    movies_df = download_movielens_dataset(dataset)
    artificial_links = download_artificial_links(dataset)

    movies_df_linked = movies_df.join(
        artificial_links,
        validate='1:1',
        rsuffix='_imdb'
    )

    imdb_data = download_imdb_data(dataset, movies_df_linked['imdb_id'])
    result =  movies_df_linked.join(imdb_data, validate='1:1')

    # Add a description column
    result['description'] = result['summary'].fillna(result['synopsis'])

    # Add the item id for compatibility with other datasets
    result['item_id'] = result.index.astype(str)

    # Add the image slug, when the image is available
    result['image_slug'] = result['item_id'].apply(lambda x: [ x ])

    # Remove the image slug when the image is not available
    missing_image = result['image_url'].isna()
    missing_images_loc = result.loc[missing_image, 'image_slug']
    result.loc[missing_image, 'image_slug'] = missing_images_loc.apply(
        lambda _: []
    )

    return result


def find_image_url(soup: BeautifulSoup) -> Optional[str]:
    og_image_meta = soup.find('meta', { 'property': "og:image" })
    result = og_image_meta['content']
    if not result.endswith('imdb_logo.png'):
        return result


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
        args: Tuple[int, str, str, threading.local]
    ) -> MovieDataResult:
    movie_id, imdb_id, dataset, thread_data = args

    url = f'https://www.imdb.com/title/{imdb_id}/plotsummary/'
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

    if image_url is not None:
        img_folder = BASE_DATA_FOLDER / dataset / 'images'
        img_folder.mkdir(exist_ok=True)
        dest_file = img_folder / f'{movie_id}.jpg'
        if not dest_file.exists():
            download_file(
                image_url,
                dest_file,
                show_progress=False,
                session=thread_data.session
            )

    return MovieDataResult(
        movie_id=movie_id,
        image_url=image_url,
        synopsis=synopsis,
        summary=summary
    )

def get_imdb_data_file(dataset: str) -> Path:
    return BASE_DATA_FOLDER / dataset / 'imdb_data.pkl'


def init_result_df(
        dataset: str, imdb_ids: pd.Series, resume: bool
) -> pd.DataFrame:
    result_df = pd.DataFrame(index=imdb_ids.index)
    result_df['imdb_id'] = imdb_ids
    result_df['processed'] = False

    imdb_data_file = get_imdb_data_file(dataset)
    if imdb_data_file.exists() and resume:
        result_df = pd.read_pickle(imdb_data_file)

    return result_df

def download_imdb_data(
        dataset: str,
        imdb_ids: pd.Series,
        max_workers=DOWNLOAD_PROCESSES,
        resume=True
    ):
    result_df = init_result_df(dataset, imdb_ids, resume)

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
            tqdm(total=len(pending_imdb_id), unit='movie', smoothing=0) as progress:
        args = [
            (movie_id, imdb_id, dataset, thread_storage)
            for (movie_id, imdb_id) in pending_imdb_id.items()
        ]
        async_results = pool.imap_unordered(download_movie_data, args)

        try:
            for process_result in async_results:
                progress.update()
                # Write the result in the correct row
                movie_id = process_result.movie_id
                result_df.at[movie_id, 'synopsis'] = process_result.synopsis
                result_df.at[movie_id, 'summary'] = process_result.summary
                result_df.at[movie_id, 'image_url'] = process_result.image_url
                result_df.at[movie_id, 'processed'] = True
        finally:
            print('Saving...')
            # Persist the result in case of interruption
            result_df.to_pickle(get_imdb_data_file(dataset))

    # Drop unnecessary columns
    return result_df.drop(columns=['processed', 'imdb_id'])


def get_reviews_file(dataset: str) -> Path:
    return BASE_DATA_FOLDER / dataset / 'reviews.pkl'


def images_dir(dataset: str) -> Path:
    return BASE_DATA_FOLDER / dataset / 'images'

def get_csv_reviews(dataset: str) -> pd.DataFrame:
    if dataset == 'ml-1m':
        sep = '::'
        engine = 'python'
        filename = 'ratings.dat'
        names = ['user_id', 'movie_id', 'rating', 'timestamp'],
    else:
        sep = ','
        engine = None
        filename = 'ratings.csv'
        names = None

    return pd.read_csv(
        BASE_DATA_FOLDER / dataset / filename,
        sep=sep,
        engine=engine,
        names=names,
    )

def reviews_df(dataset: str, used_cached: bool = True) -> pd.DataFrame:
    reviews_file = get_reviews_file(dataset)

    # It is much faster to read the pickle file directly instead of the DAT/CSV
    # file, so we save it for later
    if used_cached and reviews_file.exists():
        return pd.read_pickle(reviews_file)

    reviews = get_csv_reviews(dataset)

    # Transform timestamp to datetime from epoch seconds
    reviews['timestamp'] = pd.to_datetime(reviews['timestamp'], unit='s')

    # Rename movie_id to item_id for compatibility with other datasets
    reviews.rename(columns={
        'userId': 'user_id',
        'movieId': 'item_id',
        'movie_id': 'item_id'
    }, inplace=True)
    reviews.to_pickle(reviews_file)

    return reviews
