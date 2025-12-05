#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NMR tab file converter module.
Converts tab-delimited peak list files to CSV format for further processing.
"""

import os
import re
import csv
import pandas as pd
import logging
from pathlib import Path
import shutil
from .config import Config

def extract_simple_name(name):
    """
    从复杂的名称中提取简化版本
    例如：从"Du-SDU37-Fr3"提取"fr3"，或从"Du-SDU37-F2"提取"fr2"
    
    Parameters:
        name (str): 原始名称
        
    Returns:
        str: 简化后的名称
    """
    # 查找Fr或F后面的数字
    import re
    match = re.search(r'[Ff]r?(\d+)', name)
    if match:
        number = match.group(1)
        return f"fr{number}"
    else:
        return name

def tab_to_csv(tab_file_path, output_dir=None, strain_id=None, sample_id=None, use_simple_names=True):
    """
    Convert NMR tab file to CSV format
    
    Parameters:
        tab_file_path (str): Path to tab file
        output_dir (str): Directory to save CSV file (defaults to same directory as tab file)
        strain_id (str): Strain identifier (optional)
        sample_id (str): Sample identifier (optional)
        use_simple_names (bool): Whether to simplify strain and sample names
    
    Returns:
        str: Path to the created CSV file
    """
    tab_file_path = Path(tab_file_path)
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = tab_file_path.parent
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从文件路径中解析菌株和样本信息
    if strain_id is None or sample_id is None:
        # 尝试从路径中提取菌株和样本信息
        # 预期的路径: /主文件夹/菌株名称/样本名/10/nmrpipe/spectrum.tab
        parts = list(tab_file_path.parts)
        
        # 查找"nmrpipe"文件夹位置
        nmrpipe_idx = -1
        for i, part in enumerate(parts):
            if part == "nmrpipe":
                nmrpipe_idx = i
                break
        
        if nmrpipe_idx > 2:  # 确保有足够的路径深度
            # nmrpipe的上两级应该是"10"文件夹和样本名
            if parts[nmrpipe_idx-1] == "10" and nmrpipe_idx >= 3:
                sample_dir = parts[nmrpipe_idx-2]
                strain_dir = parts[nmrpipe_idx-3]
                
                if strain_id is None:
                    strain_id = strain_dir
                
                if sample_id is None:
                    sample_id = sample_dir
        
        # 如果还是没有找到，使用备用方法
        if strain_id is None:
            # 尝试找到父目录中的菌株名
            parent_dir = tab_file_path.parent
            while parent_dir.name and parent_dir.name != "/":
                if "SDU" in parent_dir.name or "Fr" in parent_dir.name:
                    strain_id = parent_dir.name
                    break
                parent_dir = parent_dir.parent
            
            if strain_id is None:
                strain_id = "unknown_strain"
        
        if sample_id is None:
            # 使用目录名作为样本ID
            parent_dir = tab_file_path.parent
            if parent_dir.name == "nmrpipe":
                parent_dir = parent_dir.parent
                if parent_dir.name == "10":
                    parent_dir = parent_dir.parent
            
            sample_id = parent_dir.name
    
    # 简化名称（如果启用）
    original_strain_id = strain_id
    original_sample_id = sample_id
    
    if use_simple_names:
        strain_id = extract_simple_name(strain_id)
        sample_id = extract_simple_name(sample_id)
    
    # 创建CSV文件名
    csv_filename = f"{strain_id}_{sample_id}_peaks.csv"
    csv_file_path = output_dir / csv_filename
    
    # 读取并解析tab文件
    try:
        with open(tab_file_path, 'r') as tab_file:
            lines = tab_file.readlines()
    except UnicodeDecodeError:
        # 如果默认编码失败，尝试使用不同的编码
        try:
            with open(tab_file_path, 'r', encoding='latin1') as tab_file:
                lines = tab_file.readlines()
        except:
            logging.error(f"无法读取文件: {tab_file_path}")
            raise
    
    # 提取头部信息
    header_line = None
    format_line = None
    data_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("VARS"):
            header_line = line
        elif line.startswith("FORMAT"):
            format_line = line
        elif line and not line.startswith("#"):
            data_lines.append(line)
    
    if not header_line:
        logging.warning(f"文件格式不标准: {tab_file_path}，无VARS头部")
        # 尝试从文件内容猜测格式
        if len(data_lines) > 0 and len(data_lines[0].split()) >= 5:
            # 假设这是spectrum.tab文件的标准格式，列名为INDEX, X_AXIS, X_PPM, XW, HEIGHT, CONFIDENCE
            header_line = "VARS INDEX X_AXIS X_PPM XW HEIGHT CONFIDENCE"
            logging.info(f"使用默认列名: {header_line}")
        else:
            raise ValueError(f"无效的tab文件格式: {tab_file_path}。无法找到VARS头部信息。")
    
    # 提取列名
    column_names = header_line.strip().split()[1:]  # 跳过"VARS"
    logging.debug(f"列名: {column_names}")
    
    # 解析数据
    data = []
    for line in data_lines:
        values = line.strip().split()
        # 确保数据与列数匹配
        if len(values) >= len(column_names):
            # 如果数据超出列数，只取与列数匹配的部分
            data.append(values[:len(column_names)])
        elif len(values) > 0:  # 忽略空行
            # 如果数据不足，填充空值
            values_filled = values + [''] * (len(column_names) - len(values))
            data.append(values_filled)
    
    if not data:
        logging.warning(f"文件不包含数据: {tab_file_path}")
        # 创建空DataFrame
        df = pd.DataFrame(columns=column_names)
    else:
        # 创建DataFrame
        df = pd.DataFrame(data, columns=column_names)
    
    # 转换数值列
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass  # 如果转换失败，保留为字符串
    
    # 🆕 按化学位移(X_PPM)降序排序 - Deep Picker输出是无序的，需要排序以便后续分析
    if 'X_PPM' in df.columns and len(df) > 0:
        # 降序排列：从高ppm到低ppm（NMR谱图的标准显示方式）
        df = df.sort_values('X_PPM', ascending=False).reset_index(drop=True)
        ppm_max = df['X_PPM'].max()
        ppm_min = df['X_PPM'].min()
        logging.info(f"已按X_PPM降序排序，ppm范围: {ppm_max:.4f} - {ppm_min:.4f} ({len(df)} 个峰)")
    
    # 添加元数据列
    df['strain_id'] = original_strain_id  # 保存原始菌株ID
    df['simple_strain_id'] = strain_id    # 保存简化的菌株ID
    df['sample_id'] = original_sample_id  # 保存原始样本ID
    df['simple_sample_id'] = sample_id    # 保存简化的样本ID
    df['source_file'] = str(tab_file_path)
    
    # Save as CSV
    df.to_csv(csv_file_path, index=False)
    
    logging.info(f"Converted tab file to CSV: {csv_file_path}")
    return str(csv_file_path)

def find_and_convert_tab_files(base_dir, output_base_dir=None, use_simple_names=True, in_place=True):
    """
    查找并转换所有tab文件为CSV格式
    
    Parameters:
        base_dir (str): 基础目录，用于搜索tab文件
        output_base_dir (str): 输出CSV文件的基础目录（如果不是原地转换）
        use_simple_names (bool): 是否使用简化的菌株和样本名称
        in_place (bool): 是否在原地保存CSV文件（放入与tab文件相同的目录）
    
    Returns:
        list: 创建的所有CSV文件的路径
    """
    base_dir = Path(base_dir)
    created_files = []
    found_files = 0
    
    # 如果指定了输出目录且不是原地转换，创建输出目录
    if output_base_dir is not None and not in_place:
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有tab文件
    logging.info(f"开始在{base_dir}中查找tab文件...")
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tab'):
                found_files += 1
                tab_file = Path(root) / file
                
                # 确定输出目录
                if in_place:
                    # 原地保存CSV - 直接放入tab文件所在目录
                    output_dir = tab_file.parent
                else:
                    # 在新位置创建CSV目录结构
                    # 预期的原始路径: /主文件夹/菌株名称/样本名/10/nmrpipe/spectrum.tab
                    # 目标结构: output_base_dir/菌株名称/csv/
                    
                    # 寻找菌株名和样本名
                    parts = list(tab_file.parts)
                    strain_id = None
                    sample_id = None
                    
                    # 查找"nmrpipe"文件夹位置
                    nmrpipe_idx = -1
                    for i, part in enumerate(parts):
                        if part == "nmrpipe":
                            nmrpipe_idx = i
                            break
                    
                    if nmrpipe_idx > 2:  # 确保有足够的路径深度
                        # nmrpipe的上两级应该是"10"文件夹和样本名
                        if parts[nmrpipe_idx-1] == "10" and nmrpipe_idx >= 3:
                            strain_id = parts[nmrpipe_idx-3]
                    
                    if strain_id is None:
                        # 使用备用方法
                        rel_path = tab_file.relative_to(base_dir) if tab_file.is_relative_to(base_dir) else Path(tab_file.name)
                        parts = list(rel_path.parts)
                        if len(parts) > 0:
                            strain_id = parts[0]  # 假设第一级目录是菌株
                        else:
                            strain_id = "unknown_strain"
                    
                    # 创建输出目录
                    if use_simple_names:
                        strain_id = extract_simple_name(strain_id)
                    
                    output_dir = output_base_dir / strain_id / "csv"
                
                # 确保输出目录存在
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 转换tab文件
                try:
                    csv_file = tab_to_csv(
                        tab_file,
                        output_dir=output_dir,
                        use_simple_names=use_simple_names
                    )
                    created_files.append(csv_file)
                    logging.info(f"已转换: {tab_file} -> {csv_file}")
                except Exception as e:
                    logging.error(f"转换{tab_file}时出错: {str(e)}")
                    logging.exception("详细错误:")
    
    logging.info(f"找到{found_files}个tab文件，成功转换{len(created_files)}个")
    return created_files

def organize_nmr_samples(base_dir, output_dir=None):
    """
    Organize NMR samples by strain, ensuring each strain has its set of samples
    with standardized naming.
    
    Parameters:
        base_dir (str): Base directory with raw NMR data
        output_dir (str): Output directory for organized data
    
    Returns:
        dict: Dictionary mapping strains to their samples
    """
    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir / "organized"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify NMR experiment types
    nmr_types = ['HSQC', 'COSY', 'TOCSY', 'NOESY', '1H', 'NMR', '13C']
    
    # Dictionary to track strain -> samples
    strain_samples = {}
    
    # Dictionary to track sample types
    sample_counts = {}
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tab'):
                tab_file = Path(root) / file
                
                # Get relative path to determine structure
                try:
                    rel_path = tab_file.relative_to(base_dir)
                    path_parts = list(rel_path.parts)
                except ValueError:
                    # If not relative, use full path
                    path_parts = list(tab_file.parts)
                
                # Default assignments
                strain_id = None
                sample_type = None
                
                # Try to extract strain and sample type from path or filename
                if len(path_parts) >= 2:
                    # Assume first part is strain and extract experiment type
                    strain_id = path_parts[0]
                    
                    # Look for experiment type in directory names or file name
                    for part in path_parts[1:] + [file]:
                        for nmr_type in nmr_types:
                            if nmr_type in part:
                                sample_type = nmr_type
                                break
                        if sample_type:
                            break
                else:
                    # Try to extract from filename
                    for nmr_type in nmr_types:
                        if nmr_type in file:
                            sample_type = nmr_type
                            break
                
                # If we couldn't identify, use defaults
                if strain_id is None:
                    strain_id = "unknown_strain"
                    
                if sample_type is None:
                    # Use the parent directory name as sample type
                    sample_type = Path(root).name
                    if sample_type == base_dir.name:
                        sample_type = "unknown_type"
                
                # Create unique sample ID based on type and count
                if strain_id not in sample_counts:
                    sample_counts[strain_id] = {}
                
                if sample_type not in sample_counts[strain_id]:
                    sample_counts[strain_id][sample_type] = 0
                
                sample_counts[strain_id][sample_type] += 1
                count = sample_counts[strain_id][sample_type]
                
                sample_id = f"{sample_type}" if count == 1 else f"{sample_type}_{count}"
                
                # Track samples for this strain
                if strain_id not in strain_samples:
                    strain_samples[strain_id] = []
                
                if sample_id not in strain_samples[strain_id]:
                    strain_samples[strain_id].append(sample_id)
                
                # Create organized directory structure
                org_strain_dir = output_dir / strain_id
                org_sample_dir = org_strain_dir / sample_id
                org_sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy and rename the tab file with standardized name
                dest_file = org_sample_dir / f"{strain_id}_{sample_id}.tab"
                shutil.copy2(tab_file, dest_file)
                
                # Convert to CSV
                csv_dir = org_sample_dir / "nmrpipe"
                try:
                    tab_to_csv(
                        str(dest_file),
                        output_dir=str(csv_dir),
                        strain_id=strain_id,
                        sample_id=sample_id
                    )
                except Exception as e:
                    logging.error(f"Error converting organized tab file {dest_file}: {str(e)}")
    
    return strain_samples

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test with a sample tab file
    import sys
    if len(sys.argv) > 1:
        tab_file = sys.argv[1]
        csv_file = tab_to_csv(tab_file)
        print(f"Converted {tab_file} to {csv_file}")
