from pathlib import Path
from contextlib import contextmanager
import sys

class Tee:
    """Mirrors writes to both a file and the original stream."""

    def __init__(self, stream, filepath: Path):
        self._stream = stream
        self._file = open(filepath, "a", buffering=1, encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)
    

@contextmanager
def tee_logging(output_path: Path):
    log_file = output_path / "output.log"
    with Tee(sys.stdout, log_file) as tee_out, Tee(sys.stderr, log_file) as tee_err:
        sys.stdout = tee_out
        sys.stderr = tee_err
        try:
            yield
        finally:
            sys.stdout = tee_out._stream
            sys.stderr = tee_err._stream