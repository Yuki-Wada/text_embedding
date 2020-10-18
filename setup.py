import itertools
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Extensions
ext_modules = []


c_extensions = {}
cpp_extensions = {
    'mltools.model.word2vec_impl.word2vec_impl_cython': ['mltools/model/word2vec_impl/word2vec_impl_cython.pyx'],
}

def make_c_ext(use_cython=False):
    for module, sources in c_extensions.items():
        if use_cython:
            sources = [source.replace('.c', '.pyx') for source in sources]
        yield Extension(module, sources=sources, language='c')

def make_cpp_ext(use_cython=False):
    extra_args = []
    system = platform.system()

    if system == 'Linux':
        extra_args.extend(['-std=c++11', '-I/mnt/i/Yuki/workspace/mltools/lib/eigen-3.3.7', '-mavx512f'])
    elif system == 'Darwin':
        extra_args.extend(['-stdlib=libc++', '-std=c++11', '-I/mnt/i/Yuki/workspace/mltools/lib/eigen-3.3.7', '-mavx512f'])

    for module, sources in cpp_extensions.items():
        if use_cython:
            sources = [source.replace('.cpp', '.pyx') for source in sources]
        yield Extension(
            module,
            sources=sources,
            language='c++',
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
            library_dirs=['model/word2vec_impl']
        )

ext_modules = list(itertools.chain(
    make_c_ext(use_cython=False),
    make_cpp_ext(use_cython=False)
))

# Custom Commands
class CustomBuildExt(build_ext):
    #
    # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    #
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # https://docs.python.org/2/library/__builtin__.html#module-__builtin__
        __builtins__.__NUMPY_SETUP__ = False

        import numpy as np #pylint: disable=import-outside-toplevel
        import Cython.Build #pylint: disable=import-outside-toplevel

        self.include_dirs.append(np.get_include())
        self.include_dirs.append('model/word2vec_impl')
        Cython.Build.cythonize(list(make_c_ext(use_cython=True)))
        Cython.Build.cythonize(list(make_cpp_ext(use_cython=True)))

cmdclass = {'build_ext': CustomBuildExt}

# Setup Function
setup(
    name='mltools',
    version='0.0',
    description='',

    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),

    author='Yuki Wada',
    author_email='yuki.w.228285@gmail.com',

    url='https://github.com/Yuki-Wada/nlp_model',
    license='',
    keywords='Word2Vec, w2v',

    zip_safe=False,
)
