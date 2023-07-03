import argparse
from pathlib import Path
from typing import Dict, Optional

import clip
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, PreTrainedTokenizerBase

# How many products do we store in a single batch
ITEM_BATCH_SIZE=2
# How many sequences do we store in a single sample
MAX_SEQ_SIZE = 256


def get_default_device() -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu")


def chunker(df: pd.DataFrame, sz: int):
    """Split df into chunks of a determined size sz"""
    for pos in range(0, len(df), sz):
        yield df.iloc[pos:pos + sz]


def mean_pool(
    tokens: torch.Tensor,        # batch_size x seq_size
    hidden_states: torch.Tensor, #
    tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """Output the mean value only taking into account non-padding tokens"""
    assert hidden_states.dim() == 3, "We work only in batches"
    assert tokens.dim() == 2, "We work only in batches"

    # size: seq_size x batch_size
    mask = (tokens != tokenizer.pad_token_id).t()

    # size: batch_size
    seq_length = mask.sum(axis=0)

    # size: hidden_state_size x seq_size x batch_size
    hidden_rearranged = hidden_states.transpose(dim0=0, dim1=2)
    hidden_masked = hidden_rearranged * mask

    # size: hidden_state_size x batch_size
    return hidden_masked.sum(axis=1) / seq_length


def cls_pool(
    tokens: torch.Tensor,        # batch_size x seq_size
    hidden_states: torch.Tensor, #
    tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """Output the CLS token"""
    assert hidden_states.dim() == 3, "We work only in batches"
    assert (tokens[:,0] == tokenizer.cls_token_id).all(), "First tokens must be CLS"

    # size: hidden_state_size x batch_size (for compatibility with the mean_pool)
    return hidden_states[:,0].T


def get_full_texts(df: pd.DataFrame):
    res = df['title'].fillna('') + ' ' +  df['description'].fillna('')
    res = res.apply(
        lambda x: BeautifulSoup(x, features='html.parser').get_text()
    )
    res = res.apply(lambda x: x[:MAX_SEQ_SIZE])
    return res

def tokenize_products(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """Makes a Tensor from the input string"""
    full_texts = get_full_texts(df)

    return tokenizer(
        full_texts.to_list(),
        padding=True,
        max_length=MAX_SEQ_SIZE,
        truncation=True,
        return_tensors='pt'
    )

def extract_bert_text_features(
    df: pd.DataFrame,
    device: Optional['str'] = None
) -> Dict[str, npt.NDArray]:
    """
    Returns a dictionary (item_id -> np.ndarray) with the textual features per
    product
    """
    if device is None:
        device = get_default_device()

    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained("bert-large-cased")

    extracted_features: Dict[str, npt.NDArray] = {}
    model = model.to(device)

    with tqdm(total=len(df), unit='item', unit_scale=True, smoothing=1e-2) as progress:
        for batch in chunker(df, ITEM_BATCH_SIZE):
            batch_item_id = batch['item_id'].to_list()
            encoded = tokenize_products(batch, tokenizer)
            encoded.to(device)
            output = model(**encoded)

            pooled = cls_pool(
                tokens=encoded['input_ids'],
                hidden_states=output.last_hidden_state,
                tokenizer=tokenizer
            )

            for idx in range(len(batch)):
                key = batch_item_id[idx]
                extracted_features[key] = pooled[:,idx].numpy(force=True)

            progress.update(len(batch))

    return extracted_features


def extract_clip_text_features(
    df: pd.DataFrame, device: Optional['str'] = None
) -> Dict[str, npt.NDArray]:
    if device is None:
        device = get_default_device()
    model, _preprocess = clip.load('ViT-L/14', device)

    extracted_features: Dict[str, npt.NDArray] = {}

    with tqdm(total=len(df), unit='product', unit_scale=True, smoothing=1e-2) as progress:
        for batch in chunker(df, ITEM_BATCH_SIZE):
            texts = get_full_texts(batch)

            batch_tokenized = clip.tokenize(texts.to_list(), truncate=True)
            batch_tokenized = batch_tokenized.to(device)

            with torch.no_grad():
                batch_encoded_tensor = model.encode_text(batch_tokenized)

            batch_encoded = batch_encoded_tensor.numpy(force=True)
            batch_encoded = batch_encoded.astype(np.float32)

            extracted_features.update(zip(batch['item_id'], batch_encoded))
            progress.update(len(batch))

    return extracted_features
