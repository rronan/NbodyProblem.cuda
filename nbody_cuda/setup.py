import contextlib
import os.path
import subprocess
import sys
from distutils import ccompiler, msvccompiler
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc

import numpy as np
from setuptools import Extension, setup

MOD_NAMES = ["nbody_cuda"]

compile_options = {
    "msvc": ["/Ox", "/EHsc"],
    "other": {
        "gcc": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
        "nvcc": [
            # "-arch=sm_30",
            "--ptxas-options=-v",
            "-c",
            "--compiler-options",
            "'-fPIC'",
        ],
    },
}
link_options = {"msvc": [], "other": []}


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # save references to the default compiler_so and _comple methods
    if hasattr(self, "compiler_so"):
        default_compiler_so = self.compiler_so
    else:
        # This was put in for Windows, but I'm running blind here...
        default_compiler_so = None
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu" and CUDA is not None:
            # use the cuda for .cu files
            if hasattr(self, "set_executable"):
                # This was put in for Windows, but I'm running blind here...
                self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# By subclassing build_extensions we have the actual compiler that will be used
# which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_options:
    def build_options(self):
        if hasattr(self.compiler, "initialize"):
            self.compiler.initialize()
        self.compiler.platform = sys.platform[:6]
        for e in self.extensions:
            e.extra_compile_args = compile_options.get(
                self.compiler.compiler_type, compile_options["other"]
            )
            e.extra_link_args = link_options.get(
                self.compiler.compiler_type, link_options["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def generate_cython(root, source):
    print("Cythonizing sources")
    p = subprocess.call(
        [sys.executable, os.path.join(root, "bin", "cythonize.py"), source],
        env=os.environ,
    )
    if p != 0:
        raise RuntimeError("Running cythonize failed")


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if "CUDA_HOME" in os.environ:
        home = os.environ["CUDA_HOME"]
        nvcc = os.path.join(home, "bin", "nvcc")
        if not os.path.exists(nvcc):
            nvcc = os.path.join(home, "bin", "nvcc.exe")
    elif "CUDA_PATH" in os.environ:
        home = os.environ["CUDA_PATH"]
        nvcc = os.path.join(home, "bin", "nvcc")
        if not os.path.exists(nvcc):
            nvcc = os.path.join(home, "bin", "nvcc.exe")
    elif os.path.exists("/usr/local/cuda/bin/nvcc"):
        home = "/usr/local/cuda"
        nvcc = os.path.join(home, "bin", "nvcc")
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            print(
                "Warning: The nvcc binary could not be located in your $PATH. "
                "For GPU capability, either add it to your path, or set $CUDA_HOME"
            )
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": os.path.join(home, "include"),
        "lib64": os.path.join(home, "lib64"),
    }
    if not os.path.exists(cudaconfig["lib64"]):
        cudaconfig["lib64"] = os.path.join(home, "lib")
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            print("Warning: The CUDA %s path could not be located in %s" % (k, v))
            return None
    return cudaconfig


CUDA = locate_cuda()


def is_source_release(path):
    return os.path.exists(os.path.join(path, "PKG-INFO"))


def clean(path):
    for name in MOD_NAMES:
        name = name.replace(".", "/")
        for ext in [".so", ".html", ".cpp", ".c"]:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        return clean(root)

    with chdir(root):
        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, "include"),
            np.get_include(),
        ]

        if (
            ccompiler.new_compiler().compiler_type == "msvc"
            and msvccompiler.get_build_version() == 9
        ):
            include_dirs.append(os.path.join(root, "include", "msvc9"))

        ext_modules = []
        if CUDA is None:
            pass
        else:
            with chdir(root):
                ext_modules.append(
                    Extension(
                        "gravity_gpu",
                        sources=["main.cu"],
                        library_dirs=[CUDA["lib64"]],
                        libraries=["cudart"],
                        language="c++",
                        runtime_library_dirs=[CUDA["lib64"]],
                        # this syntax is specific to this build system
                        # we're only going to use certain compiler args with nvcc and not with gcc
                        # the implementation of this trick is in customize_compiler() below
                        extra_compile_args=[
                            "-arch=sm_30",
                            "--ptxas-options=-v",
                            "-c",
                            "--compiler-options",
                            "'-fPIC'",
                        ],
                        include_dirs=include_dirs + [CUDA["include"]],
                    ),
                )

        setup(
            name="gravity_gpu",
            zip_safe=False,
            # packages=PACKAGES,
            # package_data={"": ["*.pyx", "*.pxd", "*.pxi", "*.cpp"]},
            description="",
            version="1.0",
            ext_modules=ext_modules,
            install_requires=[
                "numpy>=1.7.0",
            ],
            setup_requires=["wheel>=0.32.0,<0.33.0"],
            classifiers=[
                "Development Status :: 5 - Production/Stable",
                "Environment :: Console",
                "Intended Audience :: Developers",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Operating System :: POSIX :: Linux",
                "Operating System :: MacOS :: MacOS X",
                "Operating System :: Microsoft :: Windows",
                "Programming Language :: Cython",
                "Programming Language :: Python :: 2.7",
                "Programming Language :: Python :: 3.5",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Topic :: Scientific/Engineering",
            ],
            cmdclass={"build_ext": build_ext_subclass},
        )


setup_package()
