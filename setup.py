
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


def package_description():
    text = open('README.md', 'r').read()
    return text


setup(
    name='pyrlutils',
    version="0.0.2",
    description="Utility and Helpers for Reinformcement Learning",
    long_description=package_description(),
    long_description_content_type='text/markdown',
    classifiers=[
      "Topic :: Scientific/Engineering :: Mathematics",
      "Topic :: Software Development :: Libraries :: Python Modules",
      "Topic :: Software Development :: Version Control :: Git",
      "Programming Language :: Python :: 3.7",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
      "Intended Audience :: Science/Research",
      "Intended Audience :: Developers",
    ],
    keywords="machine learning, reinforcement leaning, artifiial intelligence",
    url="https://github.com/stephenhky/PyRLUtils",
    author="Kwan-Yuet Ho",
    author_email="stephenhky@yahoo.com.hk",
    license='MIT',
    packages=[
        'pyrlutils'
    ],
    install_requires=install_requirements(),
    tests_require=[
      'unittest'
    ],
    # scripts=[],
    include_package_data=True,
    test_suite="test",
    zip_safe=False
)
