import os
import subprocess
import sys
import platform


CLIPPY_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))


def _get_venv_python_executable(venv_path: str) -> str:
    """
    Determines the conventional path to the Python executable within a virtual environment.
    """
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        # In a venv, 'bin/python' is typically the interpreter for that venv.
        # The original script used 'bin/python3'. We will try 'bin/python3' first
        # and fall back to 'bin/python' as it's more standard for venvs.
        py3_exe = os.path.join(venv_path, "bin", "python3")
        if os.path.exists(py3_exe):
            return py3_exe
        return os.path.join(venv_path, "bin", "python")

def ensure_venv(project_dir) -> str:
    """
    Ensures a virtual environment exists in the specified project_dir.
    If the venv directory ('venv') doesn't exist, it creates it using the
    currently running Python interpreter and installs dependencies from
    'requirements.txt' located in project_dir.

    Args:
        project_dir (str): The absolute path to the project directory where the 'venv'
                           folder and 'requirements.txt' are expected to be.

    Returns:
        str: The absolute path to the Python executable within the virtual environment.

    Raises:
        FileNotFoundError: If the Python executable in the venv is not found after setup.
        subprocess.CalledProcessError: If any command (venv creation, pip install) fails.
        Exception: For other unexpected errors during setup.
    """
    if not os.path.isabs(project_dir):
        project_dir = os.path.abspath(project_dir)

    venv_path = os.path.join(project_dir, "venv")
    requirements_path = os.path.join(project_dir, "requirements.txt")

    current_python_executable = sys.executable  # Python used to run this script
    created_venv_now = False

    if not os.path.isdir(venv_path):
        print(f"Virtual environment not found at '{venv_path}'.")
        print(f"Creating virtual environment using '{current_python_executable}'...")
        try:
            subprocess.run(
                [current_python_executable, "-m", "venv", venv_path],
                check=True,
                stdout=sys.stdout, # stream output
                stderr=sys.stderr
            )
            print(f"Virtual environment created at '{venv_path}'")
            created_venv_now = True
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}", file=sys.stderr)
            raise
        except FileNotFoundError:
            print(f"Error: The Python executable '{current_python_executable}' was not found.", file=sys.stderr)
            raise

    venv_python_exe = _get_venv_python_executable(venv_path)

    if not os.path.isfile(venv_python_exe):
        raise FileNotFoundError(
            f"Python executable not found in virtual environment at '{venv_path}'. "
            f"Expected at '{venv_python_exe}'. "
            "The venv might be corrupted or was not created successfully."
        )

    if created_venv_now:
        if os.path.exists(requirements_path):
            print(f"Installing dependencies from '{requirements_path}' into virtual environment...")
            try:
                subprocess.run(
                    [venv_python_exe, "-m", "pip", "install", "-r", requirements_path],
                    check=True,
                    stdout=sys.stdout, # stream output
                    stderr=sys.stderr
                )
                print("Dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing dependencies: {e}", file=sys.stderr)
                raise
            except FileNotFoundError:
                # Should be caught by the check above, but as a safeguard:
                print(f"Error: The venv Python executable '{venv_python_exe}' was not found during pip install.", file=sys.stderr)
                raise
        else:
            print(f"'{requirements_path}' not found. Skipping dependency installation.")
    #else:
        # print(f"Virtual environment already exists at '{venv_path}'. Skipping creation and dependency installation.")

    return venv_python_exe
