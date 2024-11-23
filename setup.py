from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name='augflow',
    version='1.0.0',
    author='Omar Alnasier',
    author_email='omar@connectedmotion.io',
    description='A versatile augmentation library for computer vision projects.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ConnectedMotion/augflow',
    packages=find_packages(where='.', exclude=("tests",)),
    install_requires=[
        'numpy>=1.18.0',
        'opencv-python>=4.1.0',
        'shapely>=1.7.0',
        'matplotlib>=3.2.0',
        'tqdm>=4.50.0',
        'PyYAML>=5.3.0',  # Added comma here
        'pytest'           # Correctly separated
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
