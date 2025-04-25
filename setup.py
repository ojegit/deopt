from setuptools import setup, find_packages

setup(
    name='deopt',
    version='0.1.0',
    description='Differential Evolution Optimizer in Python',
    author='Your Name',
    author_email='your@email.com',
    packages=find_packages(),
    install_requires=['numpy>=2.1.3'],
    python_requires='>=3.12',
)