import glob
from setuptools import setup

setup(
    name='vampire',
    version='0.0',
    description='🧛 Deep generative models for T cell receptor protein sequences 🧛',
    url='https://github.com/matsengrp/vampire',
    author='Matsen group',
    author_email='ematsen@gmail.com',
    packages=['vampire'],
    package_data={'vampire': ['data/*']},
    scripts=glob.glob('vampire/scripts/*.sh'),
    entry_points={'console_scripts': [
        'tcr-vae=vampire.tcr_vae:cli',
        'gene-name-conversion=vampire.gene_name_conversion:cli',
    ]})
