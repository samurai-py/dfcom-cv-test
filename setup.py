from setuptools import setup
from setuptools_scm import get_version

setup(
    name="dfcom_cv",
    version=get_version(),
    license="MIT",
    include_package_data=True,
    package_data={},
)