from setuptools import setup, find_packages
import os

long_description = ""
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="pingtai",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "plotly",
        "psutil",
        "scikit-learn",
        "tqdm",
        "tables",
        "xlsxwriter",
        "chardet",
        "appdirs",
        "nmrglue",
        "matplotlib",
        "seaborn",
        "joblib",
    ],
    
    entry_points={
        'console_scripts': [
            'nmr-processor=nmr_processor.__main__:main',
            'nmr-config=nmr_processor.setup_config:update_config_entry',
        ],
    },
    
    package_data={
        'nmr_processor': [
            'scripts/*.tcsh',
            'models/*.pkl'
        ],
    },
    
    author="Teng",
    description="生物信息分析平台 - 整合 NMR、代谢组学分析工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
)
