#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NMR processing module - handles execution of NMR data processing scripts.
"""

import subprocess
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from .config import Config
from .converter import find_and_convert_tab_files

def setup_logging(log_dir=None):
    """设置日志系统"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"nmr_process_{timestamp}.log")
    
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def run_nmr_processing(script_path=None, data_dir=None, process_flag="2", convert_tabs=True):
    """运行NMR数据处理脚本并提供详细的日志和错误处理
    
    参数:
        script_path (str): NMR处理脚本的路径
        data_dir (str): 要处理的数据目录
        process_flag (str): 处理标志，控制处理行为
        convert_tabs (bool): 是否在处理后转换tab文件为CSV
        
    返回:
        tuple: (返回码, 日志文件路径, 转换的CSV文件列表)
    """
   # 使用配置值作为默认值
    if script_path is None:
        script_path = Config.SCRIPT_PATH
    if data_dir is None:
        data_dir = Config.NMR_DATA_PATH
    
    # 转换为绝对路径
    script_path = os.path.abspath(script_path)
    data_dir = os.path.abspath(data_dir)
    
    # 设置日志
    log_dir = os.path.join(data_dir, "logs")
    log_file = setup_logging(log_dir)
    
    logging.info(f"开始NMR数据处理")
    logging.info(f"脚本路径: {script_path}")
    logging.info(f"数据目录: {data_dir}")
    logging.info(f"处理标志: {process_flag}")
    
    # 确保脚本存在
    if not os.path.isfile(script_path):
        logging.error(f"脚本文件不存在: {script_path}")
        
        # 尝试从包内复制一份
        package_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = os.path.dirname(script_path)
        
        if not os.path.exists(script_dir):
            os.makedirs(script_dir, exist_ok=True)
            logging.info(f"创建脚本目录: {script_dir}")
        
        # 优先从package_dir/scripts复制
        template_path = os.path.join(package_dir, "scripts", "nmr_converter.tcsh")
        if os.path.exists(template_path):
            import shutil
            shutil.copy2(template_path, script_path)
            os.chmod(script_path, 0o755)  # 确保脚本可执行
            logging.info(f"已从模板复制脚本到: {script_path}")
        else:
            logging.error(f"无法找到模板脚本")
            return 1, log_file, []
    
    # 确保数据目录存在
    if not os.path.isdir(data_dir):
        logging.error(f"数据目录不存在: {data_dir}")
        return 1, log_file, []
    
    # 构建命令和环境变量
    env = os.environ.copy()
    env['DEEP_PICKER_PATH'] = Config.DEEP_PICKER_PATH
    
    cmd = ["tcsh", script_path, data_dir, str(process_flag)]
    cmd_str = " ".join(cmd)
    logging.info(f"执行命令: {cmd_str}")
    logging.info(f"使用Deep Picker路径: {Config.DEEP_PICKER_PATH}")
    
    csv_files = []
    try:
        # 使用Popen实时获取输出并传递环境变量
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env  # 传递环境变量
        )
        
        # 实时读取并记录输出
        for line in process.stdout:
            line = line.rstrip()
            logging.info(f"NMR> {line}")
            print(f"NMR> {line}")  # 在控制台实时显示输出
        
        # 等待进程完成并获取返回码
        return_code = process.wait()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logging.info(f"脚本执行完成，返回码: {return_code}")
        logging.info(f"总执行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        
        # 处理完成后，如果启用了tab转换，则查找并转换tab文件
        if convert_tabs and return_code == 0:
            logging.info("处理完成，开始转换tab文件为CSV格式...")
            try:
                csv_files = find_and_convert_tab_files(data_dir)
                logging.info(f"成功转换 {len(csv_files)} 个tab文件到CSV格式")
            except Exception as e:
                logging.error(f"转换tab文件过程中发生错误: {str(e)}")
                logging.exception("详细错误信息:")
        
        return return_code, log_file, csv_files
        
    except Exception as e:
        logging.error(f"执行过程中发生错误: {str(e)}")
        return 1, log_file, csv_files

# 如果直接运行这个脚本
if __name__ == "__main__":
    print("这是一个模块文件，请通过main.py运行程序")