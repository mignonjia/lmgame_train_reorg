from setuptools import setup, find_packages
import os
import sys

setup(
    name='lmgame',
    version='0',
    package_dir={'': '.'},
    packages=find_packages(include=['lmgame']),
    author='lmgame Team',
    author_email='',
    acknowledgements='',
    description='',
    install_requires=[], 
    package_data={'lmgame': ['*/*.md']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ]
)