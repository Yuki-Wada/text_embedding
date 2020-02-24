import setuptools

setuptools.setup(
    name='nlp_model2',
    version='0.0',
    packages=setuptools.find_packages(),
    author='Yuki Wada',
    author_email='yuki.w.228285@gmail.com',
    description='',
    zip_safe=False,
    license='',
    keywords='',
    url='https://github.com/Yuki-Wada/nlp_model',
    install_requires=[
        'numpy',
        'pandas==0.25.3',
        'gensim==3.8.1',
        'mecab-python-windows',
        #'torch==1.4.0+cu92',
        'torchsummaryX',
        'cython'
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html'
    ]

)
