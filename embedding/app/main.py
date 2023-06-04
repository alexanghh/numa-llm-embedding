import json
import os
from itertools import islice
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
import numpy as np
import tiktoken
from pynumaflow.function import Messages, Message, Datum, Server

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 2048
EMBEDDING_ENCODING = 'cl100k_base'


# let's make sure to not retry on an invalid request, because that is what we want to demonstrate
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6),
       retry=retry_if_not_exception_type(openai.InvalidRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    openai.api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    openai.api_key = os.getenv("OPENAI_API_KEY", "sk-123")
    openai.api_type = os.getenv("OPENAI_API_TYPE", "open_ai")
    openai.debug = os.getenv("OPENAI_DEBUG", False)
    return openai.Embedding.create(
        input=text_or_tokens,
        model=model
    )["data"][0]["embedding"]


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH,
                           encoding_name=EMBEDDING_ENCODING, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings


# ref: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
def map_handler(_: list[str], datum: Datum) -> Messages:
    val = datum.value
    _ = datum.event_time
    _ = datum.watermark

    data = json.loads(val.decode("utf-8"))
    content = data['content']
    print('compute embedding for : ' + data['id'])
    messages = Messages()
    try:
        # openai.api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        # openai.api_key = os.getenv("OPENAI_API_KEY", "sk-123")
        # openai.api_type = os.getenv("OPENAI_API_TYPE", "open_ai")
        # openai.debug = os.getenv("OPENAI_DEBUG", False)
        #
        # response = openai.Embedding.create(
        #     model="text-embedding-ada-002",
        #     input=content
        # )
        # data['embedding'] = response['data'][0]['embedding']
        data['embedding'] = len_safe_get_embedding(content, average=True)
        messages.append(Message(str.encode(json.dumps(data)), keys=['success']))
    except Exception as e:
        print(e)
        data['exception'] = str(e)
        messages.append(Message(str.encode(json.dumps(data)), keys=['error']))
    return messages


if __name__ == "__main__":
    grpc_server = Server(map_handler=map_handler)
    grpc_server.start()
