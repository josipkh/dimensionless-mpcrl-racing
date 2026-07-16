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


def use_acados_temp_dir(ocp_or_sim: AcadosOcp | AcadosSim) -> None:
    """Creates a temporary directory for acados files, which is automatically cleaned up."""
    temp_dir = tempfile.mkdtemp(prefix="acados_")
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

    ocp_or_sim.code_gen_opts.code_export_directory = os.path.join(temp_dir, "codegen")
    ocp_or_sim.code_gen_opts.json_file = os.path.join(temp_dir, "problem_description.json")

    return