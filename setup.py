#  Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
from pathlib import Path


def _compile_cpp():
    subprocess.check_call(
        ['make', 'shared', 'DSC_FAST=1'],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

class BuildCmd(install):
    def run(self):
        _compile_cpp()
        install.run(self)

if __name__ == '__main__':
    with open(Path(__file__).parent / 'README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

    packages = find_packages('python')
    package_dir = {'': 'python'}
    package_data = {'dsc': ['*.so']}
    setup(
        name='dsc',
        version='0.1',
        author='Christian Gilli',
        author_email='christian.gilli@dspcraft.com',
        license='BSD-3-Clause',
        description='DSPCraft tensor processing library.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/dspcraft/dsc',
        packages=packages,
        package_dir=package_dir,
        install_requires=[
            'numpy',
            'psutil',
        ],
        extras_require={
            'dev': [
                'matplotlib',
                'pytest',
                'tabulate'
            ]
        },
        cmdclass={
            'install': BuildCmd,
        },
        include_package_data=True,
        package_data=package_data,
        python_requires='>=3.9'
    )