from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.1.2',
		     'h5py==2.9.0',
		     'sh==1.12.14']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
