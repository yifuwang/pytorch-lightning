# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import threading
from functools import partial, wraps
from http.server import SimpleHTTPRequestHandler

import pytest
import torch.distributed
import torch.multiprocessing as mp


@pytest.fixture(scope="function", autouse=True)
def preserve_global_rank_variable():
    """ Ensures that the rank_zero_only.rank global variable gets reset in each test. """
    from pytorch_lightning.utilities.distributed import rank_zero_only
    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


@pytest.fixture(scope="function", autouse=True)
def restore_env_variables():
    """ Ensures that environment variables set during the test do not leak out. """
    env_backup = os.environ.copy()
    yield
    # restore environment as it was before running the test
    os.environ.clear()
    os.environ.update(env_backup)


@pytest.fixture(scope="function", autouse=True)
def teardown_process_group():
    """ Ensures that the distributed process group gets closed before the next test runs. """
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def pytest_configure(config):
    config.addinivalue_line("markers", "spawn: spawn test in a separate process using torch.multiprocessing.spawn")


@pytest.mark.tryfirst
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("spawn"):
        testfunction = pyfuncitem.obj
        funcargs = pyfuncitem.funcargs
        testargs = tuple([funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames])

        mp.spawn(wraps, (testfunction, testargs))
        return True


@pytest.fixture
def tmpdir_server(tmpdir):
    if sys.version_info >= (3, 7):
        Handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
        from http.server import ThreadingHTTPServer
    else:
        # unfortunately SimpleHTTPRequestHandler doesn't accept the directory arg in python3.6
        # so we have to hack it like this

        class Handler(SimpleHTTPRequestHandler):

            def translate_path(self, path):
                # get the path from cwd
                path = super().translate_path(path)
                # get the relative path
                relpath = os.path.relpath(path, os.getcwd())
                # return the full path from root_dir
                return os.path.join(str(tmpdir), relpath)

        # ThreadingHTTPServer was added in 3.7, so we need to define it ourselves
        from http.server import HTTPServer
        from socketserver import ThreadingMixIn

        class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

    with ThreadingHTTPServer(('localhost', 0), Handler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        yield server.server_address
        server.shutdown()
