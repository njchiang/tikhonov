from distutils.core import setup

setup(
    name='tikhonov',
    version='0.1.0',
    author='Jeff Chiang',
    author_email='jeff.njchiang@gmail.com',
    packages=['tikhonov'],
    url='https://github.com/njchiang/tikhonov.git',
    license='LICENSE',
    description='fMRI analysis wrappers for Monti Lab at UCLA.',
    install_requires=[
        "scikit-learn",
        "numpy",
        "scipy"
    ],
)