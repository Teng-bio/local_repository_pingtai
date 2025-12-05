#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础分析模块 - 提供NMR数据基础分析功能
🔧 修复版本：正确处理峰数据类型
"""

import os
import logging
import time
import pandas as pd
import json
from pathlib import Path
from ..config import Config
from ..utils.decorators import memory_monitor
from ..data.loader import load_nmr_data, load_gcf_matrix

@memory_monitor
def analyze_nmr_data(gcf_matrix_path=None, nmr_data_path=None, output_path=None):
    """基础NMR数据分析功能
    
    对每个菌株的NMR数据进行基本分析，生成简单报告。
    
    参数:
        gcf_matrix_path (str): GCF矩阵文件路径
        nmr_data_path (str): NMR数据目录
        output_path (str): 输出目录
    
    返回:
        str: 输出目录路径
    """
    # 使用配置值作为默认值
    if output_path is None:
        output_path = Config.get_output_path()
    if gcf_matrix_path is None:
        gcf_matrix_path = Config.get_gcf_matrix_path()
    if nmr_data_path is None:
        nmr_data_path = Config.get_nmr_data_path()
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 加载数据
    logging.info("加载GCF矩阵数据...")
    gcf_data = load_gcf_matrix(gcf_matrix_path)
    
    logging.info("加载NMR峰数据...")
    peak_data = load_nmr_data(nmr_data_path)
    
    # 🔧 转换为菌株级别的映射 - 正确处理字典格式
    strain_to_peaks = {}
    for sample_id, peak_info in peak_data.items():
        # peak_info 是字典: {'peaks': [...], 'fraction': 'Fr1', 'strain': 'QT1'}
        if isinstance(peak_info, dict):
            strain_id = peak_info.get('strain')
            if not strain_id:
                # 降级处理：从sample_id提取
                strain_id = sample_id.split('_')[0]
        else:
            # 兼容旧格式
            strain_id = sample_id.split('_')[0]
        
        if strain_id not in strain_to_peaks:
            strain_to_peaks[strain_id] = []
        
        # 保存完整的peak_info（包含fraction等信息）
        strain_to_peaks[strain_id].append(peak_info)
    
    # 基本分析
    results = {}
    for strain_id, peaklists in strain_to_peaks.items():
        # 合并所有样本的峰 - 🔧 确保所有峰值都是数值类型
        all_peaks = []
        for peak_info in peaklists:
            if isinstance(peak_info, dict):
                # 如果是字典格式（包含peaks, fraction等）
                peaks = peak_info.get('peaks', [])
            else:
                # 如果是列表格式
                peaks = peak_info
            
            # 🔧 过滤并转换为浮点数
            for peak in peaks:
                try:
                    peak_float = float(peak)
                    if 0 <= peak_float <= 15:  # 合理范围检查
                        all_peaks.append(peak_float)
                except (ValueError, TypeError):
                    logging.debug(f"跳过无效峰值: {peak}")
                    continue
        
        # 计算简单统计
        min_val = min(all_peaks) if all_peaks else None
        max_val = max(all_peaks) if all_peaks else None
        avg_val = sum(all_peaks) / len(all_peaks) if all_peaks else None
        
        # 检查是否存在对应的GCF数据
        has_gcf = strain_id in gcf_data
        gcf_count = len(gcf_data.get(strain_id, [])) if has_gcf else 0
        
        results[strain_id] = {
            'peak_count': len(all_peaks),
            'sample_count': len(peaklists),
            'min_ppm': min_val,
            'max_ppm': max_val,
            'avg_ppm': avg_val,
            'has_gcf': has_gcf,
            'gcf_count': gcf_count
        }
        
        # 保存单个菌株的峰数据
        strain_output_file = os.path.join(output_path, f"{strain_id}_peaks.csv")
        pd.DataFrame({'ppm': all_peaks}).to_csv(strain_output_file, index=False)
        logging.info(f"保存菌株 {strain_id} 的峰数据到 {strain_output_file}")
    
    # 保存总体分析结果
    summary_file = os.path.join(output_path, "analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"分析完成！结果保存到 {output_path}")
    return output_path