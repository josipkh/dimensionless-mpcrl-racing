from pathlib import Path
from contextlib import contextmanager
import sys
import os, shutil, atexit, tempfile
from acados_template import AcadosOcp, AcadosSim

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
        except Exception as e:
            import traceback
            print(f"Run crashed with: {e}")
            traceback.print_exc()
            raise
        finally:
            sys.stdout = tee_out._stream
            sys.stderr = tee_err._stream


def create_acados_temp_dir(ocp: AcadosOcp | AcadosSim) -> str:
    """Creates a temporary directory for acados files, which is automatically cleaned up."""
    temp_dir = tempfile.mkdtemp(prefix="jkh_acados_")
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

    ocp.code_gen_opts.code_export_directory = os.path.join(temp_dir, "codegen")
    return os.path.join(temp_dir, "acados_data.json")