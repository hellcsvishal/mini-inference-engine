from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "engine",
        ["engine.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-mavx"],  # IMPORTANT
    ),
]

setup(
    name="engine",
    ext_modules=ext_modules,
)
