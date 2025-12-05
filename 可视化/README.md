# GCF数据处理与可视化系统

## 📋 系统概述

本系统整合了antiSMASH、BIG-SCAPE和MIBiG数据库，提供从生物合成基因簇（BGC）到高质量科学论文级别网络图的完整分析流程。

### 🎯 主要功能

1. **自动准备BIG-SCAPE输入**：整合antiSMASH和MIBiG的gbk文件
2. **自动解析聚类结果**：从BIG-SCAPE输出中提取BGC-GCF映射关系
3. **可视化增强**：
   - 按功能类别分配颜色（NRPS、PKS、RiPPs等）
   - 按新颖度调整透明度（新颖GCF更不透明）
   - MIBiG匹配的GCF添加星星标记
   - 保持原有v4.4布局算法

### 📊 输出文件

- **strain_gcf_network.pdf**：高质量网络图（300dpi，矢量格式）
- **strain_gcf_network.png**：网络图预览（PNG格式）
- **network_statistics.txt**：详细统计报告
- **data/*.csv**：可视化数据文件

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 必需依赖
pip install pandas numpy matplotlib biopython networkx

# 科学计算依赖（推荐）
pip install scipy scikit-learn

# 其他工具
pip install psutil
```

### 2. 基础用法（推荐）

```bash
python run_pipeline.py \
  --antismash-dir /path/to/antismash_results/ \
  --mibig-dir /path/to/mibig_gbk/ \
  --output-dir /path/to/output/
```

系统会提示您手动运行BIG-SCAPE命令。

### 3. 自动模式

```bash
python run_pipeline.py \
  --antismash-dir /path/to/antismash_results/ \
  --mibig-dir /path/to/mibig_gbk/ \
  --output-dir /path/to/output/ \
  --auto-run-bigscape
```

---

## 📖 详细说明

### 输入数据要求

#### antiSMASH结果目录结构

```
antismash_results/
├── 003C31/
│   ├── ctg1.region001.gbk
│   ├── ctg1.region002.gbk
│   └── ...
├── 078C05/
│   ├── ctg1.region001.gbk
│   └── ...
└── ...
```

**注意**：
- 每个菌株一个文件夹，文件夹名为菌株名
- 只需要带`region`的gbk文件，不需要完整基因组文件

#### MIBiG数据库目录结构

```
mibig_gbk_4.0/
├── BGC0000001.gbk
├── BGC0000002.gbk
├── BGC0000003.gbk
└── ...
```

### 完整流程

#### 步骤1：准备BIG-SCAPE输入

系统会自动：
1. 扫描antiSMASH结果目录
2. 查找所有带region的gbk文件
3. 重命名文件（格式：`{strain_name}_{region_id}.gbk`）
4. 扫描MIBiG数据库
5. 重命名文件（格式：`MIBIG_{bgc_id}.gbk`）
6. 复制到统一目录

**输出示例**：
```
bigscape_input/
├── 003C31_region001.gbk
├── 003C31_region002.gbk
├── MIBIG_BGC0000001.gbk
├── MIBIG_BGC0000002.gbk
└── bgc_metadata.csv
```

#### 步骤2：运行BIG-SCAPE

**手动模式**：
系统会提示您运行以下命令：
```bash
cd ~/bigscape/BiG-SCAPE-1.1.5
python bigscape.py -i /path/to/bigscape_input/ --cutoffs 0.3 -o /path/to/bigscape_output/
```

**自动模式**：
系统会自动执行上述命令。

#### 步骤3：解析聚类结果

系统会自动：
1. 读取所有`*_clustering_c0.30.tsv`文件
2. 提取BGC-GCF映射关系
3. 计算新颖度分数（基于是否包含MIBiG BGC）
4. 生成可视化CSV文件

**新颖度分数计算**：
- 包含MIBiG BGC → `score = 0.2`（已知）
- 不包含MIBiG BGC → `score = 0.8`（新颖）

#### 步骤4：生成可视化

系统会：
1. 读取可视化CSV
2. 计算菌株相似性（基于GCF的Jaccard系数）
3. 对菌株进行聚类
4. 使用MDS计算菌株位置
5. 使用力导向布局计算GCF位置
6. 应用视觉属性：
   - **颜色**：按功能类别（NRPS=紫、PKS=蓝、RiPPs=橙等）
   - **大小**：按strain_count（出现在多少菌株中）
   - **透明度**：按新颖度（新颖=不透明，已知=透明）
   - **标记**：MIBiG匹配的GCF添加星星

---

## 🎨 视觉设计

### 功能类别配色方案

| 功能类别 | 颜色 | 十六进制码 |
|---------|------|-----------|
| NRPS | 紫色 | #9B59B6 |
| PKSI | 蓝色 | #3498DB |
| PKS-NRP_Hybrids | 青蓝 | #1ABC9C |
| PKSother | 深蓝 | #2980B9 |
| RiPPs | 橙色 | #E67E22 |
| Terpene | 绿色 | #27AE60 |
| Others | 灰色 | #95A5A6 |
| Hybrid | 粉色 | #E91E63 |

### 视觉属性映射

1. **菌株节点**
   - 大小：固定300（可调整）
   - 颜色：按聚类分组
   - 边框：黑色
   - 标签：白色字体

2. **GCF节点**
   - 大小：`20 + strain_count * 5`
   - 颜色：按功能类别
   - 透明度：新颖度×0.9
   - 标记：MIBiG匹配→白色星星

3. **边**
   - 颜色：浅灰 (#CCCCCC)
   - 透明度：0.3
   - 宽度：0.5

---

## 📁 输出文件详解

### 1. 网络图文件

- **strain_gcf_network.pdf**：300dpi PDF矢量图，适合论文投稿
- **strain_gcf_network.png**：300dpi PNG栅格图，适合PPT和预览

### 2. 数据文件

#### data/nodes_strain.csv
```csv
strain_id,cluster_id,gcf_count
003C31,1,57
078C05,1,59
```

#### data/nodes_gcf.csv
```csv
gcf_id,biosyn_class,strain_count,has_mibig,novelty_score
GCF_10,NRPS,5,True,0.2
GCF_15,PKSI,2,False,0.8
```

#### data/edges_strain_gcf.csv
```csv
strain_id,gcf_id
003C31,GCF_10
078C05,GCF_15
```

### 3. 统计文件

- **network_statistics.txt**：详细统计报告
- **data_statistics.txt**：数据统计摘要
- **strain_similarity_matrix.csv**：菌株相似性矩阵

---

## 🔧 高级用法

### 自定义图像大小

```bash
python run_pipeline.py \
  --antismash-dir /path/to/antismash_results/ \
  --mibig-dir /path/to/mibig_gbk/ \
  --output-dir /path/to/output/ \
  --figsize 16 12 \
  --dpi 600
```

### 指定目录

```bash
python run_pipeline.py \
  --antismash-dir /path/to/antismash_results/ \
  --mibig-dir /path/to/mibig_gbk/ \
  --bigscape-input-dir /path/to/custom_input/ \
  --bigscape-output-dir /path/to/custom_output/ \
  --output-dir /path/to/output/
```

### 详细输出

```bash
python run_pipeline.py \
  --antismash-dir /path/to/antismash_results/ \
  --mibig-dir /path/to/mibig_gbk/ \
  --output-dir /path/to/output/ \
  --verbose
```

---

## ❓ 常见问题

### Q1: 运行过程中提示"BigSCAPE目录不存在"

**A**: 请确保BiG-SCAPE已安装并位于 `~/bigscape/BiG-SCAPE-1.1.5/` 目录。

### Q2: antiSMASH结果找不到region文件

**A**: 请检查：
1. antiSMASH目录结构是否正确（每个菌株一个文件夹）
2. gbk文件名是否包含`region`（如`ctg1.region001.gbk`）
3. 路径中是否包含特殊字符（如中文）

### Q3: BigSCAPE运行时间过长

**A**: BigSCAPE运行时间取决于：
1. BGC数量（通常几分钟到几小时）
2. 数据复杂度
3. 服务器性能

可以使用 `--auto-run-bigscape` 在后台运行，并定期检查输出。

### Q4: 图像显示异常

**A**: 请检查：
1. Python版本（推荐3.7+）
2. matplotlib后端设置
3. 字体是否支持（推荐Arial）

---

## 📚 模块说明

### gcf_data.py

核心数据处理模块。

**主要函数**：
- `prepare_bigscape_input()`: 准备BigSCAPE输入
- `parse_bigscape_clustering()`: 解析聚类结果
- `export_visualization_csvs()`: 导出可视化CSV

### gcf_network_plot.py

可视化模块，基于strain_gcf_network_v4.py增强。

**主要类**：
- `GCFNetworkVisualizer`: 可视化主类

**增强功能**：
- 按功能类别分配颜色
- 按新颖度调整透明度
- MIBiG匹配星星标记

### run_pipeline.py

主运行脚本，整合所有功能。

**使用模式**：
- 手动模式（默认）：提示用户运行BigSCAPE
- 自动模式：`--auto-run-bigscape` 标志

---

## 📝 更新日志

### v1.0.0 (2025-12-01)

- ✅ 集成antiSMASH、BIG-SCAPE、MIBiG
- ✅ 自动准备BigSCAPE输入
- ✅ 自动解析聚类结果
- ✅ 可视化增强：颜色、透明度、星星标记
- ✅ 两种运行模式：手动/自动
- ✅ 命令行参数配置
- ✅ 完整文档

---

## 👤 作者

Teng

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- BiG-SCAPE: https://git.wageningenur.nl/medema-group/BiG-SCAPE
- antiSMASH: https://antismash.secondarymetabolites.org/
- MIBiG: https://mibig.secondarymetabolites.org/
