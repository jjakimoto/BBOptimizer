from setuptools import setup
from setuptools import find_packages

setup(name='bboptimizer',
      version='0.1.2',
      description='The python library of Black Box Optimization',
      url='https://github.com/jjakimoto/BBoptimizer',
      author='Tomoaki',
      author_email='f.j.akimoto@gmail.com',
      license='MIT',
      packages=find_packages(),
      py_modeuls=["bboptimizer"]
      )
