import warcio
from warcio.capture_http import capture_http
import requests

with capture_http('test.warc.gz'):
    requests.get('https://scp-wiki-cn.wikidot.com/')