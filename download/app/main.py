import json
import requests
import re
from pynumaflow.function import Messages, Message, Datum, Server
from tika import parser


def map_handler(_: list[str], datum: Datum) -> Messages:
    val = datum.value
    _ = datum.event_time
    _ = datum.watermark

    data = json.loads(val.decode("utf-8"))
    url = data['url']
    print('downloading: ' + json.dumps(data))
    messages = Messages()
    try:
        response = requests.get(url)
        tika_content = parser.from_buffer(response.content)
        clean_text = re.sub(r"\s+", " ", tika_content['content']).strip()
        data['content'] = clean_text
        print(f'len of doc {len(clean_text)}')
        messages.append(Message(str.encode(json.dumps(data)), keys=['success']))
    except Exception as e:
        print(e)
        data['exception'] = str(e)
        messages.append(Message(str.encode(json.dumps(data)), keys=['error']))
    return messages


if __name__ == "__main__":
    grpc_server = Server(map_handler=map_handler)
    grpc_server.start()
