# Pingtai 生物信息分析平台

一个集成化的生物信息分析平台，包含 NMR 谱图处理、基因-代谢组学分析的工具。

## 安装
```bash
# 创建环境
mamba create -n pingtai python=3.11 \
    scikit-learn=1.6.1 \
    numpy pandas scipy plotly psutil tqdm \
    pytables xlsxwriter chardet appdirs joblib \
    matplotlib seaborn \
    -c conda-forge -c bioconda -y

# 激活并安装
conda activate pingtai
pip install nmrglue
pip install -e .
```

## 使用
```bash
nmr-processor \
    --mode all \
    --data_dir /path/to/data \
    --output_dir /path/to/results \
    --gcf_matrix /path/to/gcf_matrix.csv
```
## 作者

Teng

---


