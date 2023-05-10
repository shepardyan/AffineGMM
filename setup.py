from setuptools import setup, find_packages

setup(
    author="Yunqi Yan",
    python_requires='>=3.8',
    author_email="yyqdyx2009@live.com",
    description="Affine transform of (dense/sparse) GMMs",
    url="https://git.tsinghua.edu.cn/yyq22",
    name="AffineGMM",
    version="2.0",
    packages=find_packages(),
    install_requires=[],
    exclude_package_date={'': ['.gitignore'], '': ['dist'], '': 'build', '': 'utility.egg.info'},
)
