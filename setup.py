"""Set-up routines for reanalysis DBNs study code."""

from setuptools import setup, find_packages

install_requires = [
    'arviz',
    'dask',
    'joblib',
    'loky',
    'netCDF4',
    'numpy',
    'pandas',
    'patsy',
    'pystan',
    'regionmask',
    'scikit-learn',
    'scipy',
    'statsmodels',
    'xarray',
]

setup_requires = [
    'pytest-runner',
]

tests_require = [
    'pytest',
    'pytest-cov',
    'pytest-flake8',
]

setup(
    name='reanalysis_dbns',
    version='0.0.1',
    author='Dylan Harries',
    author_email='Dylan.Harries@csiro.au',
    description='Code accompanying DBN study of reanalyses',
    long_description='',
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    zip_safe=False
)
