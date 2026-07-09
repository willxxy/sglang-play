import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

session = requests.Session()
session.trust_env = False

response = session.post(url, json=data)

print("status:", response.status_code)
print("headers:", response.headers)
print("text:", repr(response.text))

try:
    print(response.json())
except Exception as e:
    print("JSON parse failed:", e)



'''
(sglang-play) bash-4.4$ uv run python3 src/online_demo.py 
Traceback (most recent call last):
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/connection.py", line 204, in _new_conn
    sock = connection.create_connection(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/util/connection.py", line 85, in create_connection
    raise err
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/util/connection.py", line 73, in create_connection
    sock.connect(sa)
ConnectionRefusedError: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/connectionpool.py", line 788, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/connectionpool.py", line 493, in _make_request
    conn.request(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/connection.py", line 500, in request
    self.endheaders()
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/http/client.py", line 1298, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/http/client.py", line 1058, in _send_output
    self.send(msg)
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/http/client.py", line 996, in send
    self.connect()
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/connection.py", line 331, in connect
    self.sock = self._new_conn()
                ^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/connection.py", line 219, in _new_conn
    raise NewConnectionError(
urllib3.exceptions.NewConnectionError: HTTPConnection(host='localhost', port=30000): Failed to establish a new connection: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/requests/adapters.py", line 696, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/connectionpool.py", line 842, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/urllib3/util/retry.py", line 543, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='localhost', port=30000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='localhost', port=30000): Failed to establish a new connection: [Errno 111] Connection refused"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/p01/whan/sglang-play/src/online_demo.py", line 13, in <module>
    response = session.post(url, json=data)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/requests/sessions.py", line 712, in post
    return self.request("POST", url, data=data, json=json, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/requests/sessions.py", line 651, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/requests/sessions.py", line 784, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/requests/adapters.py", line 729, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=30000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='localhost', port=30000): Failed to establish a new connection: [Errno 111] Connection refused"))

[2026-07-09 12:34:50] max_total_num_tokens=2642170, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=32768, available_gpu_mem=3.55 GB
[2026-07-09 12:34:50] INFO:     Started server process [2816856]
[2026-07-09 12:34:50] INFO:     Waiting for application startup.
[2026-07-09 12:34:50] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
[2026-07-09 12:34:50] INFO:     Application startup complete.
[2026-07-09 12:34:50] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
[2026-07-09 12:36:51] Initialization failed. warmup error: Traceback (most recent call last):
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/http_server.py", line 1770, in _execute_server_warmup
    assert res.status_code == 200, f"{res=}, {res.text=}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: res=<Response [502]>, res.text='<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n<html>\n<!-- FileName: index.html\n     Language: [en]\n-->\n<!--Head-->\n<head>\n  <meta content="text/html; charset=UTF-8" http-equiv="Content-Type">\n  <meta http-equiv="X-UA-Compatible" content="IE=7" />\n  <title>McAfee Web Gateway - Notification</title>\n  <script src="/mwg-internal/de5fs23hu73ds/files/javascript/sw.js" type="text/javascript" ></script>\n  <link rel="stylesheet" href="/mwg-internal/de5fs23hu73ds/files/default/stylesheet.css" />\n</head>\n<!--/Head-->\n<!--Body-->\n<body onload="swOnLoad();">\n  <table class=\'bodyTable\'>\n    <tr>\n      <td class=\'bodyData\' background=\'/mwg-internal/de5fs23hu73ds/files/default/img/bg_body.gif\'>\n<!--Logo-->\n<table class=\'logoTable\'>\n  <tr>\n    <td class=\'logoData\'>\n      <a href=\'http://www.ahn.org\'>\n        <img src=\'/mwg-internal/de5fs23hu73ds/files/default/img/logo_mwg.png\'></a>\n    </td>\n  </tr>\n</table>\n<!--/Logo-->\n<!--Contents-->\n<!-- FileName: cannotconnect.html\n     Language: [en]\n-->\n<!--Title-->\n<table class=\'titleTable\' background=\'/mwg-internal/de5fs23hu73ds/files/default/img/bg_navbar.jpg\'>\n  <tr>\n    <td class=\'titleData\'>\n      Cannot Connect\n    </td>\n  </tr>\n</table>\n<!--/Title-->\n\n<!--Content-->\n<table class="contentTable">\n  <tr>\n    <td class="contentData">\n      The proxy could not connect to the destination in time.\n    </td>\n  </tr>\n</table>\n<!--/Content-->\n\n<!--Info-->\n<table class="infoTable">\n  <tr>\n    <td class="infoData">\n      <b>URL: </b><script type="text/javascript">break_line("http://127.0.0.1:30000/model_info");</script><br />\n      <p class="proxyErrorData">Failure Description: :cannotconnect:server state 1:state 9:Application response 502 cannotconnect</p>\n    </td>\n  </tr>\n</table>\n<!--/Info-->\n\n<!--/Contents-->\n<!--Foot-->\n<table class=\'footTable\'>\n  <tr>\n    <td class=\'helpDeskData\' background=\'/mwg-internal/de5fs23hu73ds/files/default/img/bg_navbar.jpg\'>\n      For assistance, please contact your system administrator. <br />\nTo request a website be unblocked, please follow the link to the <a href="https://highmark.service-now.com/now/nav/ui/classic/params/target/com.glideapp.servicecatalog_cat_item_view.do%3Fv%3D1%26sysparm_id%3D100f4176473082507e8e5cfc416d43f1%26searchTerm%3Dexception">ServiceNow Request Catalog</a>\n    </td>\n  </tr>\n  <tr>\n    <td class=\'footData\'>\n      generated <span id="time">2026-07-09 12:36:51</span> by McAfee Web Gateway\n      <br />\n      python-requests/2.34.2\n    </td>\n  </tr>\n</table>\n<!--/Foot-->\n      </td>\n    </tr>\n  </table>\n</body>\n<!--/Body-->\n</html>\n'

scripts/demo_run.sh: line 10: 2816856 Killed                  sglang serve --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000


'''