import os
import itertools
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Extensions
ext_modules = []


c_extensions = {}
cpp_extensions = {
    'mltools.model.word2vec_impl': ['mltools/model/word2vec_impl.pyx']
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
        extra_args.append('-std=c++11')
    elif system == 'Darwin':
        extra_args.extend(['-stdlib=libc++', '-std=c++11'])

    for module, sources in cpp_extensions.items():
        if use_cython:
            sources = [source.replace('.cpp', '.pyx') for source in sources]
        yield Extension(
            module,
            sources=sources,
            language='c++',
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
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
        Cython.Build.cythonize(list(make_c_ext(use_cython=True)))
        Cython.Build.cythonize(list(make_cpp_ext(use_cython=True)))

cmdclass = {'build_ext': CustomBuildExt}

# Packages to install
install_requires = [
    'dill >= 0.3.1.1',
    'tqdm >= 4.41.1',
    'numpy >= 1.18.1',
    'gensim >= 3.8.1',
    'Cython == 0.29.15',
]

additional_requires = [
    'pandas >= 0.25.3',
    'scikit-learn >= 0.22.1',
]
install_requires += additional_requires

deep_q_requires = [
    'gym >= 0.16.0'
]
install_requires += deep_q_requires

torch_requires = [
    'torch == 1.3.1+cpu',
    'torchsummaryX == 1.3.0',
    'adabound == 0.0.5',
]
install_requires += torch_requires

if os.name == 'nt':
    install_requires.append('mecab-python-windows == 0.996.3')
else:
    install_requires.append('mecab-python')

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

    install_requires=install_requires,
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html'
    ],
)
