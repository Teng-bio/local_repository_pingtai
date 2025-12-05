#!/usr/bin/env python3
"""
GCF网络可视化模块
基于gcf_data.py输出的CSV生成高质量网络图
version: 1.2.0
date: 2025-12-03
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors
from matplotlib.colors import to_rgba

# 尝试导入依赖
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("⚠️ 未安装networkx")

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.manifold import MDS
    from sklearn.metrics import silhouette_score
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("⚠️ 未安装scipy/sklearn，部分功能不可用")

# 尝试导入生物信息学依赖
try:
    from Bio import SeqIO
    from Bio.Blast import NCBIXML
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.warning("⚠️ 未安装biopython")

try:
    import subprocess
    import tempfile
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False

# =============================================================================
# 配置常量
# =============================================================================

# 功能类别配色方案
BIOSYN_CLASS_COLORS = {
    'NRPS': '#9B59B6',           # 紫色
    'PKSI': '#3498DB',           # 蓝色
    'PKS-NRP_Hybrids': '#1ABC9C', # 青蓝
    'PKSother': '#2980B9',       # 深蓝
    'RiPPs': '#E67E22',          # 橙色
    'Terpene': '#27AE60',        # 绿色
    'Others': '#95A5A6',         # 灰色
    'Hybrid': '#E91E63',         # 粉色
    'Unknown': '#BDC3C7'         # 浅灰
}

# 科学文献标准字体设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# =============================================================================
# 16S rRNA计算类（从v4.py移植）
# =============================================================================

class PhylogenyCalculator:
    """计算菌株16S rRNA亲缘关系"""

    def __init__(self, barrnap_path="barrnap"):
        self.barrnap_path = barrnap_path
        self.temp_dir = tempfile.mkdtemp()
        logging.info(f"创建临时目录: {self.temp_dir}")

    def extract_16s_sequences(self, genome_dir):
        """从基因组文件提取16S rRNA序列"""
        if not BIOPYTHON_AVAILABLE:
            logging.error("需要安装BioPython: pip install biopython")
            return {}

        sixteen_s_sequences = {}
        genome_dir = Path(genome_dir)

        # 查找所有基因组文件
        genome_files = list(genome_dir.glob("*.fasta")) + \
                       list(genome_dir.glob("*.fa")) + \
                       list(genome_dir.glob("*.fna"))

        logging.info(f"找到 {len(genome_files)} 个基因组文件")

        for genome_file in genome_files:
            strain_id = genome_file.stem
            logging.info(f"处理菌株: {strain_id}")

            # 运行barrnap
            gff_file = os.path.join(self.temp_dir, f"{strain_id}.gff")
            fasta_file = os.path.join(self.temp_dir, f"{strain_id}_16s.fasta")

            cmd = [
                self.barrnap_path,
                "--kingdom", "bac",
                "--threads", "4",
                "--outseq", fasta_file,
                str(genome_file)
            ]

            try:
                with open(gff_file, 'w') as gff_out:
                    result = subprocess.run(
                        cmd, stdout=gff_out, stderr=subprocess.PIPE, text=True
                    )

                if result.returncode != 0:
                    logging.warning(f"barrnap失败 {strain_id}: {result.stderr}")
                    continue

                # 读取16S序列
                if os.path.exists(fasta_file):
                    sequences = []
                    for record in SeqIO.parse(fasta_file, "fasta"):
                        if "16S" in record.description:
                            sequences.append(str(record.seq))

                    if sequences:
                        # 选择最长的16S序列
                        longest_seq = max(sequences, key=len)
                        sixteen_s_sequences[strain_id] = longest_seq
                        logging.info(f"  提取16S rRNA: {len(longest_seq)} bp")
                    else:
                        logging.warning(f"  未找到16S rRNA序列")

            except FileNotFoundError:
                logging.error(f"barrnap未安装或不在PATH中，请安装: conda install -c bioconda barrnap")
                break
            except Exception as e:
                logging.error(f"处理 {strain_id} 时出错: {e}")

        return sixteen_s_sequences

    def calculate_similarity_matrix(self, sequences):
        """计算16S rRNA相似度矩阵（使用k-mer方法）"""
        if len(sequences) < 2:
            logging.warning("序列数量不足，无法计算相似度")
            return pd.DataFrame()

        strain_ids = list(sequences.keys())
        n = len(strain_ids)
        similarity_matrix = np.zeros((n, n))

        for i, strain1 in enumerate(strain_ids):
            for j, strain2 in enumerate(strain_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    seq1 = sequences[strain1]
                    seq2 = sequences[strain2]
                    similarity = self._calculate_kmer_similarity(seq1, seq2)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        df = pd.DataFrame(
            similarity_matrix,
            index=strain_ids,
            columns=strain_ids
        )

        return df

    def _calculate_kmer_similarity(self, seq1, seq2, k=6):
        """使用k-mer方法计算序列相似度（Jaccard系数）"""
        def get_kmers(seq, k):
            return set([seq[i:i+k] for i in range(len(seq)-k+1)])

        kmers1 = get_kmers(seq1, k)
        kmers2 = get_kmers(seq2, k)

        if not kmers1 or not kmers2:
            return 0.0

        intersection = len(kmers1 & kmers2)
        union = len(kmers1 | kmers2)

        return intersection / union if union > 0 else 0.0


# =============================================================================
# 核心类
# =============================================================================

class GCFNetworkVisualizer:
    """GCF网络可视化器"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.edge_color = '#CCCCCC'

    def load_data(
        self,
        nodes_strain_csv: Path,
        nodes_gcf_csv: Path,
        edges_csv: Path
    ) -> tuple:
        """加载可视化数据"""
        self.logger.info("加载数据...")

        # 读取CSV
        self.df_strains = pd.read_csv(nodes_strain_csv)
        self.df_gcfs = pd.read_csv(nodes_gcf_csv)
        self.df_edges = pd.read_csv(edges_csv)

        self.logger.info(f"  菌株数量: {len(self.df_strains)}")
        self.logger.info(f"  GCF数量: {len(self.df_gcfs)}")
        self.logger.info(f"  边数量: {len(self.df_edges)}")

        return self.df_strains, self.df_gcfs, self.df_edges

    def calculate_strain_similarity(self, genome_dir: Path = None) -> pd.DataFrame:
        """计算菌株相似性

        参数:
            genome_dir: 基因组FASTA目录（用于16S计算）
                       如果为None，尝试读取现有16s_similarity_matrix.csv
        """
        self.logger.info("计算菌株相似性...")

        # 如果指定了基因组目录，计算16S相似性
        if genome_dir:
            self.logger.info(f"  从基因组目录提取16S序列: {genome_dir}")

            if not BIOPYTHON_AVAILABLE:
                self.logger.error("  需要BioPython: pip install biopython")
                raise ImportError("需要安装BioPython才能提取16S序列")

            if not SUBPROCESS_AVAILABLE:
                self.logger.error("  需要barrnap工具")
                raise ImportError("需要安装barrnap: conda install -c bioconda barrnap")

            # 创建PhylogenyCalculator并提取16S序列
            phylo_calc = PhylogenyCalculator()
            sequences = phylo_calc.extract_16s_sequences(genome_dir)

            if not sequences:
                self.logger.error("  未能提取到任何16S序列")
                raise ValueError("无法从基因组文件提取16S序列")

            self.logger.info(f"  成功提取 {len(sequences)} 个16S序列")

            # 计算相似性矩阵
            df_sim = phylo_calc.calculate_similarity_matrix(sequences)

            # 保存到文件
            sim_file = self.output_dir / "16s_similarity_matrix.csv"
            df_sim.to_csv(sim_file)
            self.logger.info(f"  保存16S相似性矩阵: {sim_file}")

            return df_sim

        # 如果没有指定基因组目录，尝试读取现有的16S矩阵
        similarity_file = self.output_dir / "16s_similarity_matrix.csv"
        if similarity_file.exists():
            self.logger.info("  读取16S相似性矩阵...")
            df_sim = pd.read_csv(similarity_file, index_col=0)
            return df_sim

        # 如果没有16S矩阵，报错
        self.logger.error("  未提供基因组目录，且未找到16S相似性矩阵文件")
        self.logger.error("  请提供基因组FASTA文件目录，或确保存在16s_similarity_matrix.csv")
        raise FileNotFoundError(f"需要提供基因组目录或16S相似性矩阵: {similarity_file}")

    def cluster_strains(self, similarity_matrix: pd.DataFrame) -> dict:
        """对菌株进行聚类"""
        if not SCIPY_AVAILABLE:
            self.logger.warning("未安装scipy，跳过聚类")
            return {strain: 1 for strain in similarity_matrix.index}

        self.logger.info("对菌株进行聚类...")

        # 转换为距离矩阵
        distance_matrix = 1 - similarity_matrix.values
        np.fill_diagonal(distance_matrix, 0)

        # 层次聚类
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')

        # 自动确定最优聚类数（限制在合理范围，沿用v4.4逻辑）
        n_samples = len(similarity_matrix)
        max_k = min(5, n_samples - 1)  # 减少最大聚类数（从10改为5）

        if max_k < 2:
            clusters = [1] * n_samples
        else:
            try:
                # 优先选择2-3个聚类（更符合v4.4结果）
                # 使用轮廓系数选择最优k
                best_k = 2
                best_score = -1

                # 限制在2-5之间，更接近v4.4的2类结果
                for k in range(2, max_k + 1):
                    cluster_labels = fcluster(linkage_matrix, t=k, criterion='maxclust')
                    if len(set(cluster_labels)) > 1:
                        score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                        if score > best_score:
                            best_score = score
                            best_k = k

                # 执行聚类
                cluster_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust')
                self.logger.info(f"  最优聚类数: {best_k} (silhouette={best_score:.3f})")
            except Exception as e:
                self.logger.warning(f"聚类失败: {e}")
                # 简单降级：按数量均分
                cluster_labels = [i % 2 + 1 for i in range(n_samples)]
                self.logger.info(f"  使用默认2类聚类")

        # 创建聚类映射
        strain_clusters = dict(zip(similarity_matrix.index, cluster_labels))

        # 统计每个聚类的菌株数
        for cluster_id in sorted(set(cluster_labels)):
            count = sum(1 for c in cluster_labels if c == cluster_id)
            self.logger.info(f"  聚类 {cluster_id}: {count} 个菌株")

        return strain_clusters

    def calculate_strain_positions(
        self,
        similarity_matrix: pd.DataFrame,
        strain_clusters: dict
    ) -> dict:
        """
        计算菌株位置（采用模块化布局：基于聚类分离 + MDS局部优化）
        此方法替代了原先简单的全局MDS，确保菌株根据聚类形成清晰的岛屿结构。
        """
        if not SCIPY_AVAILABLE:
            # 备用方案：随机布局
            strains = list(similarity_matrix.index)
            np.random.seed(42)
            positions = {strain: np.array([np.random.rand()*10, np.random.rand()*10]) for strain in strains}
            self.logger.info("使用随机布局")
            return positions

        self.logger.info("计算菌株位置 (基于聚类的模块化布局)...")
        strains = list(similarity_matrix.index)
        
        # 1. 统计每个聚类的菌株
        clusters = defaultdict(list)
        for strain in strains:
            cluster_id = strain_clusters.get(strain, 1)
            clusters[cluster_id].append(strain)
        
        n_clusters = len(clusters)
        self.logger.info(f"  将菌株分布在 {n_clusters} 个聚类区域中")
        
        # 2. 为每个聚类分配一个中心位置（在圆周上均匀分布）
        cluster_centers = {}
        # 基础半径，决定了不同聚类之间的距离
        base_radius = 5.0  # 使用较大的半径确保分离
        
        for i, cluster_id in enumerate(sorted(clusters.keys())):
            angle = 2 * np.pi * i / n_clusters
            # 将中心设在 (5,5) 加上偏移
            center_x = 5.0 + base_radius * np.cos(angle)
            center_y = 5.0 + base_radius * np.sin(angle)
            cluster_centers[cluster_id] = np.array([center_x, center_y])
        
        # 3. 在每个聚类中心周围分布菌株
        strain_positions = {}
        
        for cluster_id, cluster_strains in clusters.items():
            center = cluster_centers[cluster_id]
            n_strains = len(cluster_strains)
            
            if n_strains == 1:
                # 只有一个菌株，直接放在中心
                strain_positions[cluster_strains[0]] = center
            else:
                # 多个菌株，使用聚类内部的MDS保持相对位置
                # 提取子矩阵
                cluster_indices = [strains.index(s) for s in cluster_strains]
                # 注意：iloc使用位置索引
                cluster_sim_matrix = similarity_matrix.iloc[cluster_indices, cluster_indices]
                
                # 转换为距离矩阵
                dist_matrix = 1 - cluster_sim_matrix.values
                np.fill_diagonal(dist_matrix, 0)
                
                try:
                    mds = MDS(n_components=2, dissimilarity='precomputed',
                              random_state=42, normalized_stress='auto')
                    positions_2d = mds.fit_transform(dist_matrix)
                    
                    # 归一化并缩放到合适大小
                    # 将MDS结果中心化
                    positions_2d = positions_2d - positions_2d.mean(axis=0)
                    
                    # 缩放因子：根据菌株数量调整，避免过于拥挤
                    # 聚类内部的扩散范围
                    scale = 1.5 * np.sqrt(n_strains) / 2.0
                    
                    # 归一化到 -1 到 1 之间然后乘以 scale
                    max_range = np.max(np.abs(positions_2d))
                    if max_range > 0:
                        positions_2d = (positions_2d / max_range) * scale
                    
                    # 平移到聚类中心
                    for i, strain in enumerate(cluster_strains):
                        strain_positions[strain] = center + positions_2d[i]
                        
                except Exception as e:
                    self.logger.warning(f"  聚类 {cluster_id} 内部MDS失败: {e}")
                    # 备用：在中心周围随机分布
                    for strain in cluster_strains:
                        offset = (np.random.rand(2) - 0.5) * 1.0
                        strain_positions[strain] = center + offset

        return strain_positions

    def assign_strain_colors(self, strain_clusters: dict) -> dict:
        """为菌株分配颜色（按聚类）"""
        self.logger.info("分配菌株颜色...")

        # 统计每个聚类
        clusters = defaultdict(list)
        for strain, cluster_id in strain_clusters.items():
            clusters[cluster_id].append(strain)

        # 为每个聚类生成颜色
        n_clusters = len(clusters)
        if n_clusters <= len(BIOSYN_CLASS_COLORS):
            # 使用预设颜色
            cluster_colors = list(BIOSYN_CLASS_COLORS.values())[:n_clusters]
        else:
            # 生成颜色
            cluster_colors = []
            golden_angle = 0.618033988749895
            for i in range(n_clusters):
                hue = (i * golden_angle) % 1.0
                rgb = matplotlib.colors.hsv_to_rgb(hue, 0.65, 0.80)
                cluster_colors.append(matplotlib.colors.to_hex(rgb))

        # 为每个聚类内的菌株分配颜色梯度
        strain_colors = {}
        for i, (cluster_id, strains) in enumerate(sorted(clusters.items())):
            base_color = cluster_colors[i]

            # 生成颜色梯度
            n_strains = len(strains)
            for j, strain in enumerate(sorted(strains)):
                # 亮度从深到浅
                rgb = matplotlib.colors.to_rgb(base_color)
                h, s, v = matplotlib.colors.rgb_to_hsv(rgb)  # 修正：传入单个参数
                v_new = v + 0.3 * (j / max(1, n_strains - 1))
                v_new = min(v_new, 0.95)
                rgb_new = matplotlib.colors.hsv_to_rgb((h, s * 0.8, v_new))  # 修正：传入元组
                color_hex = matplotlib.colors.to_hex(rgb_new)
                strain_colors[strain] = color_hex

        return strain_colors

    def assign_gcf_colors(self) -> dict:
        """为GCF分配颜色（按功能类别）"""
        self.logger.info("分配GCF颜色...")

        gcf_colors = {}
        for _, row in self.df_gcfs.iterrows():
            gcf_id = row['gcf_id']
            biosyn_class = row['biosyn_class']

            # 选择颜色
            if biosyn_class in BIOSYN_CLASS_COLORS:
                color = BIOSYN_CLASS_COLORS[biosyn_class]
            else:
                # 备用颜色
                color = BIOSYN_CLASS_COLORS['Unknown']

            gcf_colors[gcf_id] = color

        return gcf_colors

    def calculate_alphas(self) -> dict:
        """计算透明度（基于新颖度）"""
        self.logger.info("计算透明度...")

        alphas = {}
        for _, row in self.df_gcfs.iterrows():
            gcf_id = row['gcf_id']
            novelty_score = row['novelty_score']
            has_mibig = row['has_mibig']

            # 基础透明度
            base_alpha = 0.9

            # 根据新颖度调整
            if has_mibig:
                # 已知GCF，更透明
                novelty_factor = 0.5
            else:
                # 新颖GCF，更不透明
                novelty_factor = 1.0

            alpha = base_alpha * novelty_factor
            alphas[gcf_id] = alpha

        return alphas

    def layout_network(
        self,
        strain_positions: dict,
        strain_clusters: dict
    ) -> dict:
        """
        计算网络布局（使用隐形锚点策略实现自然的功能聚类）
        
        策略：
        1. 菌株位置固定（由cluster_strains计算好的位置决定）。
        2. 特有GCF（只属于一个菌株）：
           - 不直接连接菌株，而是连接到该菌株下的一个"隐形功能锚点"。
           - 隐形锚点与菌株有强连接。
           - 结果：同类功能的GCF会围绕在隐形锚点周围，形成自然的云团。
        3. 共享GCF（属于多个菌株）：
           - 直接连接所有相关菌株，不使用锚点。
           - 结果：它们会自然悬浮在菌株社区之间。
        """
        if not NETWORKX_AVAILABLE:
            self.logger.warning("未安装networkx，使用简单布局")
            # 简单布局备用方案
            positions = {}
            for _, row in self.df_gcfs.iterrows():
                gcf_id = row['gcf_id']
                strain_ids = row['strain_ids'].split(';')
                related_positions = [strain_positions.get(s, np.array([5.0, 5.0])) for s in strain_ids]
                center = np.mean(related_positions, axis=0)
                np.random.seed(hash(gcf_id) % (2**32))
                offset = (np.random.rand(2) - 0.5) * 0.5
                positions[gcf_id] = center + offset
            return positions

        self.logger.info("计算网络布局 (使用隐形锚点策略)...")

        # 构建NetworkX图
        G = nx.Graph()

        # 1. 添加固定位置的菌株节点
        for strain_id, pos in strain_positions.items():
            G.add_node(f"strain_{strain_id}", node_type='strain', pos=pos)

        # 准备GCF信息字典以便查询
        gcf_info = {}
        for _, row in self.df_gcfs.iterrows():
            gcf_info[row['gcf_id']] = {
                'strain_ids': row['strain_ids'].split(';'),
                'biosyn_class': row['biosyn_class']
            }

        # 2. 添加GCF节点和隐形锚点
        # 记录已创建的锚点，格式: (strain_id, biosyn_class) -> anchor_node_id
        anchors = {}

        for gcf_id, info in gcf_info.items():
            strain_ids = info['strain_ids']
            biosyn_class = info['biosyn_class']
            gcf_node_id = f"gcf_{gcf_id}"
            
            G.add_node(gcf_node_id, node_type='gcf')
            
            # 判断是特有GCF还是共享GCF
            if len(strain_ids) == 1:
                # === 特有GCF：连接到隐形锚点 ===
                strain_id = strain_ids[0]
                anchor_key = (strain_id, biosyn_class)
                anchor_node_id = f"anchor_{strain_id}_{biosyn_class}"
                
                # 如果锚点不存在，创建它
                if anchor_key not in anchors:
                    G.add_node(anchor_node_id, node_type='anchor')
                    # 锚点连接到菌株 (强连接)
                    G.add_edge(f"strain_{strain_id}", anchor_node_id, weight=5.0)
                    anchors[anchor_key] = anchor_node_id
                
                # GCF连接到锚点 (较强连接，使其聚集)
                G.add_edge(anchor_node_id, gcf_node_id, weight=3.0)
                
            else:
                # === 共享GCF：直接连接到所有相关菌株 ===
                for strain_id in strain_ids:
                    # 连接权重适中，让其自然悬浮在中间
                    G.add_edge(f"strain_{strain_id}", gcf_node_id, weight=1.0)

        # 3. 初始化位置
        initial_pos = {}
        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'strain':
                strain_id = node.replace('strain_', '')
                initial_pos[node] = strain_positions.get(strain_id, np.array([5.0, 5.0]))
            elif G.nodes[node]['node_type'] == 'anchor':
                # 锚点初始位置在菌株附近
                strain_id = node.split('_')[1]
                base_pos = strain_positions.get(strain_id, np.array([5.0, 5.0]))
                initial_pos[node] = base_pos + (np.random.rand(2) - 0.5) * 0.1
            else:
                # GCF初始位置：先放在中心附近，让力导向去推
                initial_pos[node] = np.array([5.0, 5.0]) + (np.random.rand(2) - 0.5)

        # 4. 运行力导向布局
        fixed_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'strain']
        
        try:
            # 增加迭代次数以确保收敛，使用较大的k值防止过度重叠
            pos = nx.spring_layout(
                G,
                pos=initial_pos,
                fixed=fixed_nodes,
                k=0.3,             # 斥力系数
                iterations=500,    # 增加迭代次数
                weight='weight',
                seed=42
            )
            
            # 只返回GCF的位置，忽略锚点和菌株(菌株位置已知)
            return {k.replace('gcf_', ''): v for k, v in pos.items() if k.startswith('gcf_')}
            
        except Exception as e:
            self.logger.warning(f"布局计算失败: {e}")
            # 简单回退策略
            positions = {}
            for gcf_id in gcf_info:
                positions[gcf_id] = np.array([5.0, 5.0])
            return positions

    def generate_plot(
        self,
        strain_positions: dict,
        strain_colors: dict,
        gcf_colors: dict,
        gcf_alphas: dict,
        gcf_positions: dict,
        strain_clusters: dict,
        figsize=(14, 14),
        dpi=300
    ):
        """生成网络图"""
        self.logger.info("="*80)
        self.logger.info("生成网络图")
        self.logger.info("="*80)

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

        # 统计信息
        n_strains = len(self.df_strains)
        n_gcfs = len(self.df_gcfs)

        # 计算节点大小
        strain_size = 300
        gcf_base_size = 30  # 稍微调大一点以便看清
        gcf_scale = 5

        # 1. 绘制边
        self.logger.info("绘制边...")
        for _, row in self.df_edges.iterrows():
            strain_id = row['strain_id']
            gcf_id = row['gcf_id']

            # 获取位置
            strain_pos = strain_positions.get(strain_id)
            gcf_pos = gcf_positions.get(gcf_id)

            if strain_pos is None or gcf_pos is None:
                continue

            ax.plot(
                [strain_pos[0], gcf_pos[0]],
                [strain_pos[1], gcf_pos[1]],
                color=self.edge_color,
                linewidth=0.5,
                alpha=0.3,
                zorder=1
            )

        # 2. 绘制GCF节点
        self.logger.info("绘制GCF节点...")
        
        # 将GCF按类型(特有/共享)分类绘制
        for _, row in self.df_gcfs.iterrows():
            gcf_id = row['gcf_id']
            strain_count = row['strain_count']
            has_mibig = row['has_mibig']
            strain_ids_list = row['strain_ids'].split(';')

            # 位置
            pos = gcf_positions.get(gcf_id)
            if pos is None:
                continue

            # 大小（基于strain_count）
            size = gcf_base_size + strain_count * gcf_scale

            # 颜色和透明度
            color = gcf_colors.get(gcf_id, '#808080')
            alpha = gcf_alphas.get(gcf_id, 0.9)

            # === 视觉区分共享 vs 特有 ===
            if len(strain_ids_list) > 1:
                # 共享GCF：加粗黑色边框，保持原有颜色
                linewidth = 1.5
                edgecolor = 'black'
                zorder = 3  # 放在更上层
            else:
                # 特有GCF：无边框或极细边框
                linewidth = 0.0
                edgecolor = 'none'
                zorder = 2

            # 绘制GCF节点
            ax.scatter(
                pos[0], pos[1],
                s=size,
                c=color,
                alpha=alpha,
                edgecolors=edgecolor,
                linewidths=linewidth,
                zorder=zorder
            )

            # 如果有MIBiG匹配，添加星星标记
            if has_mibig:
                ax.scatter(
                    pos[0], pos[1],
                    s=size * 0.3,
                    marker='*',
                    facecolor='white',
                    edgecolor='black',
                    linewidths=1,
                    zorder=zorder + 1
                )

        # 3. 绘制菌株节点
        self.logger.info("绘制菌株节点...")
        for _, row in self.df_strains.iterrows():
            strain_id = row['strain_id']
            
            pos = strain_positions.get(strain_id)
            if pos is None:
                continue

            color = strain_colors.get(strain_id, '#E64B35')

            ax.scatter(
                pos[0], pos[1],
                s=strain_size,
                c=color,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5,
                zorder=4
            )

            # 添加菌株标签
            ax.annotate(
                strain_id,
                (pos[0], pos[1]),
                fontsize=8,
                ha='center',
                va='center',
                fontweight='bold',
                color='white',
                zorder=5
            )

        # 创建图例
        self.logger.info("创建图例...")

        legend_elements = []

        # 菌株聚类图例
        strain_cluster_counts = defaultdict(int)
        for cluster_id in strain_clusters.values():
            strain_cluster_counts[cluster_id] += 1

        for cluster_id in sorted(strain_cluster_counts.keys()):
            count = strain_cluster_counts[cluster_id]
            # 使用第一个菌株的颜色作为聚类代表色
            strains_in_cluster = [s for s, c in strain_clusters.items() if c == cluster_id]
            if strains_in_cluster:
                color = strain_colors.get(strains_in_cluster[0], '#E64B35')
                legend_elements.append(
                    mpatches.Patch(
                        color=color,
                        label=f'Cluster {cluster_id} (n={count})'
                    )
                )

        # GCF功能类别图例
        biosyn_classes = sorted(self.df_gcfs['biosyn_class'].unique())
        for biosyn_class in biosyn_classes:
            color = BIOSYN_CLASS_COLORS.get(biosyn_class, BIOSYN_CLASS_COLORS['Unknown'])
            count = len(self.df_gcfs[self.df_gcfs['biosyn_class'] == biosyn_class])
            legend_elements.append(
                mpatches.Patch(
                    color=color,
                    label=f'{biosyn_class} (n={count})'
                )
            )
        
        # 节点类型图例 (新增)
        legend_elements.append(mpatches.Patch(color='none', label='')) # 空行分隔
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label='Shared GCF (Bold Border)',
                      markerfacecolor='gray', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label='Specific GCF (No Border)',
                      markerfacecolor='gray', markersize=8, markeredgecolor='none')
        )

        # MIBiG标记图例
        legend_elements.append(
            mpatches.Patch(
                color='white',
                label='★ Contains MIBiG BGC'
            )
        )

        # 添加图例
        legend = ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=9,
            framealpha=0.95,
            edgecolor='black',
            fancybox=False,
            frameon=True
        )
        legend.get_frame().set_linewidth(0.5)

        # 设置标题
        ax.set_title(
            'Strain-GCF Co-occurrence Network',
            fontsize=16,
            fontweight='bold',
            pad=20
        )

        # 添加统计信息
        ax.text(
            0.5, 1.02,
            f'({n_strains} strains, {n_gcfs} GCFs)',
            transform=ax.transAxes,
            fontsize=12,
            ha='center',
            va='bottom'
        )

        # 隐藏坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        pdf_file = self.output_dir / 'strain_gcf_network.pdf'
        png_file = self.output_dir / 'strain_gcf_network.png'

        plt.savefig(
            pdf_file,
            dpi=dpi,
            bbox_inches='tight',
            format='pdf',
            facecolor='white',
            edgecolor='none'
        )
        self.logger.info(f"保存PDF: {pdf_file}")

        plt.savefig(
            png_file,
            dpi=dpi,
            bbox_inches='tight',
            format='png',
            facecolor='white',
            edgecolor='none'
        )
        self.logger.info(f"保存PNG: {png_file}")

        plt.close()

        self.logger.info("✅ 网络图生成完成!")

    def generate_statistics(self):
        """生成统计报告"""
        self.logger.info("生成统计报告...")

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Strain-GCF网络统计报告")
        report_lines.append("="*80)
        report_lines.append("")

        # 基本统计
        total_strains = len(self.df_strains)
        total_gcfs = len(self.df_gcfs)
        total_edges = len(self.df_edges)

        report_lines.append("基本统计:")
        report_lines.append(f"  菌株数量: {total_strains}")
        report_lines.append(f"  GCF数量: {total_gcfs}")
        report_lines.append(f"  边数量: {total_edges}")
        report_lines.append(f"  平均每个菌株的GCF数: {total_edges / total_strains:.1f}")
        report_lines.append(f"  平均每个GCF的菌株数: {total_edges / total_gcfs:.1f}")
        report_lines.append("")

        # 功能类别分布
        class_counts = self.df_gcfs['biosyn_class'].value_counts()
        report_lines.append("GCF功能类别分布:")
        for biosyn_class, count in class_counts.items():
            pct = count / total_gcfs * 100
            report_lines.append(f"  {biosyn_class}: {count} ({pct:.1f}%)")
        report_lines.append("")

        # MIBiG匹配统计
        has_mibig_count = self.df_gcfs['has_mibig'].sum()
        mibig_pct = has_mibig_count / total_gcfs * 100
        report_lines.append("MIBiG匹配:")
        report_lines.append(f"  包含MIBiG BGC的GCF: {has_mibig_count} ({mibig_pct:.1f}%)")
        report_lines.append(f"  新颖GCF（不含MIBiG）: {total_gcfs - has_mibig_count} ({100 - mibig_pct:.1f}%)")
        report_lines.append("")

        # 新颖度分数分布
        novel_gcfs = self.df_gcfs[self.df_gcfs['novelty_score'] >= 0.8]
        known_gcfs = self.df_gcfs[self.df_gcfs['novelty_score'] <= 0.3]
        report_lines.append("新颖度分数:")
        report_lines.append(f"  高新颖度GCF (score≥0.8): {len(novel_gcfs)}")
        report_lines.append(f"  已知GCF (score≤0.3): {len(known_gcfs)}")
        report_lines.append("")

        # 保存报告
        report_file = self.output_dir / "network_statistics.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"  保存: {report_file}")

    def run(
        self,
        nodes_strain_csv: Path,
        nodes_gcf_csv: Path,
        edges_csv: Path,
        genome_dir: Path = None,
        figsize=(14, 14),
        dpi=300
    ):
        """运行完整可视化流程"""
        # 加载数据
        self.load_data(nodes_strain_csv, nodes_gcf_csv, edges_csv)

        # 计算菌株相似性和聚类（只使用16S）
        similarity_matrix = self.calculate_strain_similarity(genome_dir=genome_dir)
        strain_clusters = self.cluster_strains(similarity_matrix)

        # 计算菌株位置
        strain_positions = self.calculate_strain_positions(similarity_matrix, strain_clusters)

        # 分配视觉属性
        strain_colors = self.assign_strain_colors(strain_clusters)
        gcf_colors = self.assign_gcf_colors()
        gcf_alphas = self.calculate_alphas()

        # 计算GCF位置
        gcf_positions = self.layout_network(strain_positions, strain_clusters)

        # 生成图像
        self.generate_plot(
            strain_positions,
            strain_colors,
            gcf_colors,
            gcf_alphas,
            gcf_positions,
            strain_clusters,
            figsize,
            dpi
        )

        # 生成统计报告
        self.generate_statistics()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """测试主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GCF网络可视化模块测试"
    )

    parser.add_argument(
        "--nodes-strain",
        type=str,
        required=True,
        help="菌株节点CSV文件路径"
    )

    parser.add_argument(
        "--nodes-gcf",
        type=str,
        required=True,
        help="GCF节点CSV文件路径"
    )

    parser.add_argument(
        "--edges",
        type=str,
        required=True,
        help="边CSV文件路径"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[14, 14],
        help="图像大小（宽度，高度）"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="图像分辨率"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )

    args = parser.parse_args()

    # 设置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # 创建可视化器
        visualizer = GCFNetworkVisualizer(output_dir=Path(args.output_dir))

        # 运行可视化
        visualizer.run(
            nodes_strain_csv=Path(args.nodes_strain),
            nodes_gcf_csv=Path(args.nodes_gcf),
            edges_csv=Path(args.edges),
            figsize=tuple(args.figsize),
            dpi=args.dpi
        )

        logger.info("\n" + "="*80)
        logger.info("✅ 可视化完成!")
        logger.info("="*80)
        logger.info(f"输出文件:")
        logger.info(f"  - PDF: {args.output_dir}/strain_gcf_network.pdf")
        logger.info(f"  - PNG: {args.output_dir}/strain_gcf_network.png")
        logger.info(f"  - 统计: {args.output_dir}/network_statistics.txt")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())