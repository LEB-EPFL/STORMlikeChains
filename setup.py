try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Polymer simulations for localization microscopy',
    'long_description': 'PolymerPy facilitates the generation and analysis of biopolymer datasets for use in single molecule localization microscopy work.'
    'author': 'Kyle M. Douglass',
    'url': 'https://github.com/kmdouglass/PolymerPy',
    'download_url': 'https://github.com/kmdouglass/PolymerPy',
    'author_email': 'kyle.douglass@epfl.ch',
    'version': '0.1.0a1',
    'install_requires': ['numpy',
                         'scipy',
                         'matplotlib',
                         'scikit-learn'],
    'packages': ['PolymerPy'],
    'scripts': [],
    'name': 'PolymerPy',
    'classifiers': ['Development Status :: 3 - Alpha',
                    'Programming Language :: Python :: 3',
                    'Intended Audience :: Science/Research',
                    'Topic :: Scientific/Engineering',]
    'keywords': 'polymer simulation localization superresolution microscopy',
    
}

setup(**config)
