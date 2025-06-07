from setuptools import setup, find_packages
import os

# Read README content without failing if file doesn't exist
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except (FileNotFoundError, IOError):
    long_description = "Binary Options Trading Tools with advanced analysis"

setup(
    name='BinaryOptionsTools',
    version='1.0.0',
    author='Binary Options Trader',
    author_email='binaryoptions@example.com',
    description='Tools for Binary Options trading with advanced analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/example/BinaryOptionsTools',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'python-telegram-bot',
        'requests',
        'schedule',
        'ta',
        'scikit-learn',
        'torch',
        'pandas-ta',
        'tzlocal',
        'asyncio',
        'websockets',
        'python-dotenv'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)