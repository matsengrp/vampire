import glob
from setuptools import setup
import versioneer

setup(
    name='vampire',
    description='🧛 Deep generative models for T cell receptor protein sequences 🧛',
    url='https://github.com/matsengrp/vampire',
    author='Matsen group',
    author_email='ematsen@gmail.com',
    packages=['vampire', 'vampire.models'],
    package_data={'vampire': ['data/*']},
    scripts=glob.glob('vampire/scripts/*.sh'),
    entry_points={
        'console_scripts': [
            'tcr-vae=vampire.tcr_vae:cli',
            'gene-name-conversion=vampire.gene_name_conversion:cli',
        ]
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
