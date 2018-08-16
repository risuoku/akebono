from setuptools import (
    setup,
    find_packages,
)
import importlib

_version = importlib.import_module('akebono').__version__
EXCLUDE_FROM_PACKAGES = []

setup (
    name='akebono',
    version=_version,
    author='risuoku',
    author_email='risuo.data@gmail.com',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'akebono = akebono.commands.run:main',
        ]
    },
    install_requires=[
        'pandas',
        'scikit-learn',
        'Jinja2',
        'scipy'
    ],
)
