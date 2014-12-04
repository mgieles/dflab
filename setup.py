#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = ['numpy', 'scipy>=0.13.0']

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='dflab',
    version='0.1.0',
    description='DF based stuff',
    long_description=readme + '\n\n' + history,
    author='Mark Gieles, Alice Zocchi',
    author_email='m.gieles@surrey.ac.uk, a.zocchi@surrey.ac.uk',
    url='https://github.com/mgieles/dflab',
    packages=[
        'dflab',
    ],
    package_dir={'dflab': 'dflab'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='dflab',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
