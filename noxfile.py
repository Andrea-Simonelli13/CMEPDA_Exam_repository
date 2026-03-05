import sys
from pathlib import Path
import shutil

import nox

sys.path.append(str(Path(__file__).parent / "src"))

from CMEPDA_Exam_repository import (
    CMEPDA_EXAM_REPOSITORY_DOCS,
    CMEPDA_EXAM_REPOSITORY_ROOT,
    CMEPDA_EXAM_REPOSITORY_SRC,
    CMEPDA_EXAM_REPOSITORY_TESTS,
)

# Folders containing source code that potentially needs linting.
SOURCE_DIRS = ("src", "tests")
 #inserire le altre cartelle che devono essere lintate

# Reuse existing virtualenvs by default.
nox.options.reuse_existing_virtualenvs = True

@nox.session(venv_backend="none")
def cleanup(session: nox.Session) -> None:
    """Cleanup temporary files.
    """
    # Remove all the __pycache__ folders.
    for folder_path in (CMEPDA_EXAM_REPOSITORY_ROOT, CMEPDA_EXAM_REPOSITORY_SRC, CMEPDA_EXAM_REPOSITORY_TESTS):
        _path = folder_path / "__pycache__"
        if _path.exists():
            shutil.rmtree(_path)
    # Cleanup the docs.
    _path = CMEPDA_EXAM_REPOSITORY_DOCS / "_build"
    if _path.exists():
            shutil.rmtree(_path)


@nox.session(venv_backend="none")
def docs(session: nox.Session) -> None:
    """Build the HTML docs.

    Note this is a nox session with no virtual environment, based on the assumption
    that it is not very interesting to build the documentation with different
    versions of Python or the associated environment, since the final thing will
    be created remotely anyway. (This also illustrates the use of the nox.session
    decorator called with arguments.)
    """
    source_dir = CMEPDA_EXAM_REPOSITORY_DOCS
    output_dir = CMEPDA_EXAM_REPOSITORY_DOCS / "_build" / "html"
    session.run("sphinx-build", "-b", "html", source_dir, output_dir, *session.posargs)


@nox.session
def ruff(session: nox.Session) -> None:
    """Run ruff.
    """
    session.install("ruff")
    #session.install(".[dev]")
    session.run("ruff", "check", *session.posargs)


@nox.session
def pylint(session: nox.Session) -> None:
    """Run pylint.
    """
    session.install("pylint")
    #session.install(".[dev]")
    session.run("pylint", *SOURCE_DIRS, *session.posargs)


@nox.session
def test(session: nox.Session) -> None:
    """Run the unit tests.
    """
    session.install("pytest")
    #session.install(".[dev]")
    session.run("pytest", *session.posargs)