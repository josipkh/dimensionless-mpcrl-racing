from pathlib import Path

class Tee:
    """Mirrors writes to both a file and the original stream."""
 
    def __init__(self, stream, filepath: Path):
        self._stream = stream
        self._file = open(filepath, "a", buffering=1, encoding="utf-8")
 
    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
 
    def flush(self):
        self._stream.flush()
        self._file.flush()
 
    def close(self):
        self._file.close()
 
    # Proxy everything else (e.g. fileno, isatty) to the original stream
    def __getattr__(self, name):
        return getattr(self._stream, name)