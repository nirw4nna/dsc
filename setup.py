#  Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
from subprocess import CalledProcessError
import os


class BuildCmd(install):
    @staticmethod
    def _compile():
        cwd = os.path.dirname(os.path.abspath(__file__))
        print(f'About to compile Cpp cwd={cwd}')
        print(f'files={os.listdir(cwd)}')
        try:
            subprocess.check_call(
                ['make shared DSC_FAST=1'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        except CalledProcessError:
            print('failed to compile cpp')

    def run(self):
        BuildCmd._compile()
        install.run(self)

if __name__ == '__main__':
    packages = find_packages('python')
    package_dir = {'': 'python'}
    package_data = {'': ['Makefile'], 'dsc': ['include/*', 'src/*']}
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
        cmdclass={
            'install': BuildCmd,
        },
        include_package_data=True,
        package_data=package_data,
        python_requires='>=3.9'
    )