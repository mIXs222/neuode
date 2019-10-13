#!/usr/bin/env python

from distutils.core import setup
import os


setup(
    name='Neuode',
    version='1.0',
    description='Neural ODE Package',
    author='Supawit Chockchowwat',
    author_email='supawitch22@gmail.com',
    packages=[
        'neuode',
        'neuode.dynamics',
        'neuode.interface',
        'neuode.model',
        'neuode.util',
    ],
)