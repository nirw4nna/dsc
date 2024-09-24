#  Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from setuptools import setup, find_namespace_packages


if __name__ == '__main__':
    packages = find_namespace_packages(
        where='python', exclude=['tests', 'benchmarks']
    )
    package_dir = {'': 'python'}

    setup(
        name='dsc',
        version='0.1',
        author='Christian Gilli',
        author_email='christian.gilli@dspcraft.com',
        description='',
        url='https://github.com/dspcraft/dsc',
        packages=packages,
        package_dir=package_dir,
        install_requires=[
            'numpy',
        ],
        extras_require={
            'dev': [
                'matplotlib',
                'pytest',
                'tabulate'
            ]
        },
        python_requires='>=3.9'
    )