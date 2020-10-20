from os import makedirs
from shutil import rmtree
from os.path import join, exists
from subprocess import run

from invoke import task

from tasks.util.env import PROJ_ROOT

_BUILD_DIR = join(PROJ_ROOT, "build", "cmake")
_BIN_DIR = join(_BUILD_DIR, "bin")
_INSTALL_PREFIX = join(_BUILD_DIR, "install")


@task
def cmake(ctx, clean=False):
    if clean and exists(_BUILD_DIR):
        rmtree(_BUILD_DIR)

    if not exists(_BUILD_DIR):
        makedirs(_BUILD_DIR)

    cmd = [
        "cmake",
        "-GNinja",
        "-DCMAKE_INSTALL_PREFIX={}".format(_INSTALL_PREFIX),
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_CXX_COMPILER=/usr/bin/clang++-10",
        "-DCMAKE_C_COMPILER=/usr/bin/clang-10",
        "../..",
    ]

    run(" ".join(cmd), shell=True, cwd=_BUILD_DIR)


@task
def cc(ctx, target):
    """
    Compile the given target
    """
    run(
        "cmake --build . --target {}".format(target),
        cwd=_BUILD_DIR,
        shell=True,
    )


@task
def install(ctx, target):
    """
    Install the given target
    """
    run(
        "ninja install {}".format(target),
        cwd=_BUILD_DIR,
        shell=True,
    )


@task
def r(ctx, target):
    """
    Run the given target
    """
    run(
        "./{}".format(target),
        cwd=_BIN_DIR,
        shell=True,
    )
