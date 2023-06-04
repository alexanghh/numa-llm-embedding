import datetime
import json
import unittest
from pynumaflow.function import Datum, Messages
from app.main import map_handler


class test_main(unittest.TestCase):

    def test_download(self):
        ok_data = {}
        ok_data['url'] = 'https://www.channelnewsasia.com/singapore/shangri-la-dialogue-singapore-vested-interest-us-china-communication-ng-eng-hen-3537466'
        datum = Datum(keys=[],
                      event_time=datetime.datetime.now(),
                      watermark=datetime.datetime.now(),
                      metadata=None,
                      value=json.dumps(ok_data).encode("utf-8"))
        msgs = map_handler(None, datum)
        self.assertTrue(len(msgs.items()) > 0)
        data = json.loads(msgs.items()[0].value.decode("utf-8"))
        print(data['content'])
        self.assertTrue('content' in data and len(data['content']) > 0)  # add assertion here


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.getcwd())

    unittest.main()
