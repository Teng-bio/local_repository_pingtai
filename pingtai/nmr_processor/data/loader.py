#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块 - 统一处理NMR和GCF数据的加载功能
🔧 修复版本：正确提取fraction信息和处理数据类型
"""

import os
import pandas as pd
import logging
import chardet
import re
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np
from ..utils.decorators import memory_monitor
from ..utils.memory_protector import (
    MemoryProtector, 
    with_memory_protection,
    memory_limit,
    MemoryLeakDetector,
    emergency_cleanup
)
from ..config import Config
from ..analysis.peak_classifier import PeakClassifierFilter

# 设置日志记录
def setup_logging():
    """设置日志系统"""
    try:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(Config.get_log_path()), exist_ok=True)
        
        # 然后配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(Config.get_log_path()),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        # 出错时回退到仅控制台日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        logging.warning(f"设置文件日志失败: {e}")


# 🔧 修复：从sample_id提取fraction信息 - 正确处理各种格式
def extract_fraction_info(sample_id):
    """
    从sample_id中提取fraction信息
    
    支持的格式：
    - QT1-Fr1, QT1_Fr1
    - SDU37-Fr2, SDU37_fr2
    - 09-6S3-Fr3
    - QT1_fr2_peaks (文件名格式)
    - /path/003C31/003C31-Fr1/10 (路径格式)
    
    参数:
        sample_id (str): 样本ID（可能来自文件名或路径）
        
    返回:
        tuple: (strain_id, fraction)，如果无法提取返回(sample_id, 'unknown')
    """
    # 🔧 改进的正则表达式：匹配 -Fr[1-5] 或 _fr[1-5]（不区分大小写），可以不在末尾
    # 使用(?:_peaks)?$ 来处理 _peaks 后缀（可选）
    fraction_pattern = r'[-_](Fr[1-5])(?:[-_]peaks)?(?:$|/)'
    match = re.search(fraction_pattern, sample_id, re.IGNORECASE)
    
    if match:
        fraction = match.group(1).capitalize()  # 统一为 Fr1, Fr2 等格式
        # 提取strain部分（fraction之前的所有内容）
        strain = sample_id[:match.start()]
        # 清理strain中的尾部路径分隔符
        strain = strain.rstrip('/_\\')
        return strain, fraction
    
    # 如果没有匹配到标准格式，尝试从路径中查找
    # 例如：/home/teng/nmr/nmr_data/09-6S3/09-6S3-Fr3/10
    if '/' in sample_id or '\\' in sample_id:
        path_parts = Path(sample_id).parts
        for part in reversed(path_parts):
            match = re.search(fraction_pattern, part, re.IGNORECASE)
            if match:
                fraction = match.group(1).capitalize()
                strain = part[:match.start()].rstrip('/_\\')
                # 如果提取的strain为空，使用前一个目录名
                if not strain and len(path_parts) >= 2:
                    strain_idx = list(path_parts).index(part) - 1
                    if strain_idx >= 0:
                        strain = path_parts[strain_idx]
                return strain, fraction
    
    # 如果还是无法匹配，返回原样
    logging.debug(f"无法从 '{sample_id}' 提取fraction信息，使用默认值")
    return sample_id, 'unknown'


@with_memory_protection(max_memory_percent=85)
def load_gcf_matrix(matrix_path=None):
    """加载GCF矩阵文件
    
    参数:
        matrix_path (str): GCF矩阵文件的路径
            
    返回:
        dict: 菌株到GCF列表的映射
    """
    # 在函数调用时设置日志，而不是在导入时
    setup_logging()

    if matrix_path is None:
        matrix_path = Config.get_gcf_matrix_path()
        
    gcf_data = {}
    try:
        # 检测编码
        with open(matrix_path, 'rb') as f:
            raw = f.read(10000)
            enc = chardet.detect(raw)['encoding']
        
        with open(matrix_path, 'r', encoding=enc, errors='replace') as f:
            reader = pd.read_csv(f)
            for _, row in reader.iterrows():
                strain = row.get('Strain', '').strip()
                if not strain:
                    continue
                
                gcf_names = row.get('gcf_names', '')
                if not gcf_names:
                    logging.warning(f"菌株 {strain} 没有GCF数据")
                    continue
                
                gcf_data[strain] = gcf_names.split(';')
        
        logging.info(f"已加载 {len(gcf_data)} 个菌株的GCF数据")
        
        # 数据质量检查
        if len(gcf_data) == 0:
            raise ValueError("没有加载到任何GCF数据，请检查文件格式")
        
        return gcf_data
    except Exception as e:
        logging.error(f"处理GCF矩阵文件失败: {str(e)}")
        logging.exception("详细错误信息")
        raise


@with_memory_protection(max_memory_percent=85)
def load_nmr_data(peak_dir=None):
    """加载NMR峰数据
    
    参数:
        peak_dir (str): 包含峰文件的目录
            
    返回:
        dict: 样本ID到{peaks, fraction, strain}的映射
        格式: {sample_id: {'peaks': [...], 'fraction': 'Fr1', 'strain': 'QT1'}}
    """
    # 在函数调用时设置日志，而不是在导入时
    setup_logging()
    
    if peak_dir is None:
        peak_dir = Config.get_nmr_data_path()
        
    peak_data = {}
    count = 0
    
    try:
        if not os.path.exists(peak_dir):
            raise FileNotFoundError(f"峰数据目录不存在: {peak_dir}")
        
        # 递归查找所有CSV文件
        csv_files = list(Path(peak_dir).glob("**/*_peaks.csv"))
        if not csv_files:
            csv_files = list(Path(peak_dir).glob("**/*.csv"))
        
        logging.info(f"找到 {len(csv_files)} 个CSV文件")

        peak_filter = None

        # 如果启用峰分类器过滤，初始化分类器
        if Config.USE_PEAK_CLASSIFIER:
            try:
                peak_filter = PeakClassifierFilter(
                    model_path=Config.get_peak_classifier_model_path(),
                    confidence_threshold=Config.PEAK_CONFIDENCE_THRESHOLD
                )
                logging.info("✓ 峰分类器已启用")
            except Exception as e:
                logging.warning(f"峰分类器加载失败，将不进行过滤: {e}")
                peak_filter = None
        
        for csv_file in csv_files:
            try:
                # 🔧 从文件名或路径提取菌株和fraction信息
                # 优先从文件名提取
                filename = csv_file.stem  # 不含扩展名
                strain_id, fraction = extract_fraction_info(filename)
                
                # 如果文件名提取失败，尝试从路径提取
                if fraction == 'unknown':
                    # 例如：/path/to/09-6S3/09-6S3-Fr3/10/nmrpipe/spectrum.csv
                    parent_dir = csv_file.parent
                    if parent_dir.name == "nmrpipe":
                        parent_dir = parent_dir.parent
                    if parent_dir.name == "10":
                        parent_dir = parent_dir.parent
                    
                    strain_id, fraction = extract_fraction_info(parent_dir.name)
                    
                    # 如果还是失败，尝试从完整路径提取
                    if fraction == 'unknown':
                        strain_id, fraction = extract_fraction_info(str(csv_file))
                
                # 构建唯一的sample_id
                sample_id = f"{strain_id}_{fraction}" if fraction != 'unknown' else strain_id
                
                # 读取CSV文件
                df = pd.read_csv(csv_file)

                # === 峰分类器过滤 ===
                if peak_filter is not None:
                    try:
                        # 检查是否有必需的列
                        required_cols = ['HEIGHT', 'CONFIDENCE', 'X_PPM']
                        if all(col in df.columns for col in required_cols):
                            original_count = len(df)
                            df, filter_stats = peak_filter.filter_peaks(df)
                            logging.debug(
                                f"文件 {csv_file.name}: "
                                f"过滤前 {original_count} 峰 -> "
                                f"过滤后 {len(df)} 峰 "
                                f"(过滤率: {filter_stats['filter_rate']:.1f}%)"
                            )
                        else:
                            logging.warning(f"文件 {csv_file.name} 缺少必需列，跳过峰过滤")
                    except Exception as e:
                        logging.warning(f"文件 {csv_file.name} 峰过滤失败: {e}")
                # === 峰分类器过滤结束 ===

                # 提取峰数据
                peaks = _extract_peaks_from_df(df)
                if peaks:
                    # 保存peaks和fraction信息
                    peak_data[sample_id] = {
                        'peaks': peaks,
                        'fraction': fraction,
                        'strain': strain_id
                    }
                    count += 1
                    
                    # 批处理进度
                    if count % 100 == 0:
                        logging.info(f"已处理 {count} 个峰文件")
            except Exception as e:
                logging.warning(f"处理文件 {csv_file} 时出错: {str(e)}")
        
        logging.info(f"已加载 {count} 个峰文件，共 {len(peak_data)} 个峰数据记录")
        
        # 🔧 数据质量统计
        fraction_counts = defaultdict(int)
        strain_counts = defaultdict(int)
        for sample_id, info in peak_data.items():
            fraction_counts[info['fraction']] += 1
            strain_counts[info['strain']] += 1
        
        logging.info(f"数据统计：")
        logging.info(f"  - 菌株数量: {len(strain_counts)}")
        logging.info(f"  - Fraction分布: {dict(fraction_counts)}")
        
        # 数据质量检查
        if len(peak_data) == 0:
            logging.warning("没有加载到任何峰数据，这可能会影响分析结果")
        
        return peak_data
    except Exception as e:
        logging.error(f"处理峰数据失败: {str(e)}")
        logging.exception("详细错误信息")
        if count > 0:
            logging.info(f"已成功处理 {count} 个峰文件，继续使用部分数据")
            return peak_data
        raise


def _extract_peaks_from_df(df):
    """从DataFrame中提取峰数据
    
    参数:
        df (pandas.DataFrame): 包含峰数据的DataFrame
        
    返回:
        list: 峰值列表（降序排列，从高ppm到低ppm）
    """
    peaks = []
    
    # 尝试找到包含ppm或X_PPM的列
    ppm_col = None
    for col_name in ['ppm', 'X_PPM', 'x_ppm', 'chemical_shift']:
        if col_name in df.columns:
            ppm_col = col_name
            break
    
    if ppm_col is None and 'X_AXIS' in df.columns:
        ppm_col = 'X_AXIS'
    
    if ppm_col is None and len(df.columns) > 0:
        # 尝试使用第一列作为ppm列
        ppm_col = df.columns[0]
    
    if ppm_col is not None:
        # 提取峰值 - 🔧 确保所有值都是数值类型
        for _, row in df.iterrows():
            try:
                peak_value = row[ppm_col]
                # 🔧 强制转换为浮点数，跳过无效值
                if pd.isna(peak_value):
                    continue
                peak = float(peak_value)
                if 0 <= peak <= 15:  # 合理的ppm范围
                    peaks.append(peak)
            except (ValueError, TypeError) as e:
                # 记录无法转换的值
                logging.debug(f"跳过无效峰值: {row[ppm_col]} (错误: {e})")
                continue
        
        # 聚类处理峰
        if peaks:
            peaks = _cluster_peaks(peaks)
            # 确保峰值降序排列（从高ppm到低ppm）
            peaks.sort(reverse=True)
    
    return peaks


def _cluster_peaks(raw_peaks):
    """聚类处理峰，将相近的峰值合并
    
    根据用户分析，容差匹配可能已经足够，但保留此函数用于：
    1. 处理Deep Picker可能拾取的重复峰
    2. 将0.0001范围内的微小差异统一
    
    参数:
        raw_peaks (list): 原始峰值列表
        
    返回:
        list: 聚类后的峰值列表
    """
    if not raw_peaks:
        return []
    
    try:
        X = np.array(raw_peaks).reshape(-1, 1)
        # 使用0.001的eps与容差匹配保持一致
        db = DBSCAN(eps=0.001, min_samples=1).fit(X)
        
        clusters = defaultdict(list)
        for i, label in enumerate(db.labels_):
            clusters[label].append(raw_peaks[i])
        
        # 每个簇取最小值作为代表
        return [round(min(cluster), 3) for cluster in clusters.values()]
    except Exception as e:
        logging.error(f"峰聚类处理失败: {str(e)}")
        # 失败时直接返回原始峰
        return [round(p, 3) for p in raw_peaks]