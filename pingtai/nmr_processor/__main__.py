#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NMR-GCF分析软件包主入口。
提供命令行界面和主要工作流程。
✅ 已修复所有兼容性问题
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 导入配置模块（已整合所有配置管理功能）
from nmr_processor.config import Config
from nmr_processor.processor import run_nmr_processing
from nmr_processor.converter import find_and_convert_tab_files, organize_nmr_samples
from nmr_processor.analysis.base import analyze_nmr_data
from nmr_processor.analysis.gcf_peak_matcher import match_gcf_peaks
from nmr_processor.utils.memory_protector import (
    MemoryProtector, 
    with_memory_protection,
    memory_limit,
    MemoryLeakDetector,
    emergency_cleanup
)

def setup_logging(log_dir=None):
    """设置日志系统"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(Config.get_output_path(), "logs")
    
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

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="NMR-GCF分析软件包")
    
    # 主要模式选择
    parser.add_argument("--mode", type=str, 
                      choices=["all", "process", "convert", "analyze", "match", "config"],
                      default="all", 
                      help="执行模式: all=完整流程, process=仅NMR处理, convert=仅转换, "
                           "analyze=仅分析, match=仅GCF-峰匹配, config=配置更新")
    
    # 通用参数
    parser.add_argument("--base_dir", type=str, help="基础数据目录", default=None)
    parser.add_argument("--output_dir", type=str, help="输出目录", default=None)
    parser.add_argument("--log_dir", type=str, help="日志目录")
    
    # NMR处理参数
    parser.add_argument("--script", type=str, help="tcsh脚本的路径", default=None)
    parser.add_argument("--data_dir", type=str, help="数据目录", default=None)
    parser.add_argument("--process_flag", type=str, help="处理标志", default="2")
    parser.add_argument("--no-convert", action="store_true", 
                      help="禁用tab文件自动转换为CSV")
    parser.add_argument("--convert-only", action="store_true", 
                      help="仅转换tab文件为CSV，不执行NMR处理")
    parser.add_argument("--organize", action="store_true", 
                      help="组织NMR数据到标准化的文件夹结构")
    parser.add_argument("--in-place", action="store_true", 
                      help="在原位置保存CSV文件（默认：False）")
    
    # GCF-峰匹配参数
    parser.add_argument("--gcf_matrix", type=str, help="GCF矩阵文件路径", default=None)
    parser.add_argument("--workers", type=int, help="工作进程数", default=None)
    parser.add_argument("--batch_size", type=int, help="批处理大小", default=None)

    # 峰分类器参数
    parser.add_argument("--use-classifier", action="store_true", 
                      help="启用峰分类器过滤低置信度峰")
    parser.add_argument("--classifier-threshold", type=float, default=0.5, 
                      help="峰分类器置信度阈值（默认0.5）")
    parser.add_argument("--classifier-model", type=str, 
                      help="峰分类器模型文件路径")
    
    # 聚类和打分参数
    parser.add_argument("--clustering-tolerance", type=float, 
                      help="DBSCAN聚类容差（ppm），默认0.002")
    parser.add_argument("--perfect-match-score", type=int,
                      help="完美匹配得分，默认10")
    parser.add_argument("--orphan-peak-score", type=int,
                      help="孤立峰扣分，默认-10")
    
    return parser.parse_args()

def update_config_from_args(args):
    """从命令行参数更新配置"""
    # 更新基础路径
    if args.base_dir:
        Config.BASE_PATH = os.path.abspath(args.base_dir)
        logging.info(f"已设置基础路径: {Config.BASE_PATH}")
    
    # 更新输出路径
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        Config._custom_output_path = output_dir
        original_get_output_path = Config.get_output_path
        Config.get_output_path = classmethod(
            lambda cls: Config._custom_output_path if hasattr(Config, '_custom_output_path') 
            else original_get_output_path()
        )
        logging.info(f"已设置输出路径: {output_dir}")
    
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
        Config._custom_nmr_data_path = data_dir
        original_get_nmr_data_path = Config.get_nmr_data_path
        Config.get_nmr_data_path = classmethod(
            lambda cls: Config._custom_nmr_data_path if hasattr(Config, '_custom_nmr_data_path') 
            else original_get_nmr_data_path()
        )
        logging.info(f"已设置NMR数据路径: {data_dir}")
    
    if args.script:
        Config.SCRIPT_PATH = os.path.abspath(args.script)
        logging.info(f"已设置脚本路径: {Config.SCRIPT_PATH}")
    
    if args.gcf_matrix:
        gcf_matrix = os.path.abspath(args.gcf_matrix)
        Config._custom_gcf_matrix_path = gcf_matrix
        original_get_gcf_matrix_path = Config.get_gcf_matrix_path
        Config.get_gcf_matrix_path = classmethod(
            lambda cls: Config._custom_gcf_matrix_path if hasattr(Config, '_custom_gcf_matrix_path') 
            else original_get_gcf_matrix_path()
        )
        logging.info(f"已设置GCF矩阵路径: {gcf_matrix}")
    
    # 更新处理参数
    if args.workers:
        Config.NUM_WORKERS = args.workers
        logging.info(f"已设置工作进程数: {Config.NUM_WORKERS}")
    
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
        logging.info(f"已设置批处理大小: {Config.BATCH_SIZE}")
    
    # 更新峰分类器配置
    if args.use_classifier:
        Config.USE_PEAK_CLASSIFIER = True
        logging.info("已启用峰分类器")

    if args.classifier_threshold:
        Config.PEAK_CONFIDENCE_THRESHOLD = args.classifier_threshold
        logging.info(f"峰分类器置信度阈值: {Config.PEAK_CONFIDENCE_THRESHOLD}")

    if args.classifier_model:
        Config.PEAK_CLASSIFIER_MODEL_PATH = os.path.abspath(args.classifier_model)
        logging.info(f"峰分类器模型路径: {Config.PEAK_CLASSIFIER_MODEL_PATH}")
    
    # 更新聚类和打分配置
    if args.clustering_tolerance:
        Config.PEAK_CLUSTERING['tolerance'] = args.clustering_tolerance
        logging.info(f"DBSCAN聚类容差: {Config.PEAK_CLUSTERING['tolerance']}")
    
    if args.perfect_match_score:
        Config.SCORING_CONFIG['perfect_match'] = args.perfect_match_score
        logging.info(f"完美匹配得分: {Config.SCORING_CONFIG['perfect_match']}")
    
    if args.orphan_peak_score:
        Config.SCORING_CONFIG['orphan_peak'] = args.orphan_peak_score
        logging.info(f"孤立峰扣分: {Config.SCORING_CONFIG['orphan_peak']}")
    
    # 创建必要的目录
    Config.create_directories()

def main():
    """主函数 - 软件包入口点"""
    # 初始化全局内存保护器
    global_protector = MemoryProtector(max_memory_percent=90)
    
    # 检查初始内存状态
    status, _, info = global_protector.check_memory(force=True)
    if status != 'safe':
        logging.warning(
            f"⚠️ 启动时内存已较高: {info['system_used_percent']:.1f}%"
        )
    
    # 使用整合后的配置方法检查首次运行
    first_run = Config.first_run_check()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果是配置模式，直接进入配置更新
    if args.mode == "config":
        Config.update_config_entry()
        return 0
    
    # 使用整合后的配置方法加载用户配置
    user_config = Config.get_user_config()
    
    # 从用户配置更新设置
    for key, value in user_config.items():
        if hasattr(Config, key.upper()):
            setattr(Config, key.upper(), value)
        elif key == 'data_dir' and value:
            Config.BASE_PATH = os.path.dirname(value)
    
    # 从命令行参数更新配置
    update_config_from_args(args)
    
    # 在继续之前创建必要的目录
    Config.create_directories()
    
    # 设置日志
    log_dir = args.log_dir if args.log_dir else os.path.join(Config.get_output_path(), "logs")
    log_file = setup_logging(log_dir)
    
    print("=" * 60)
    print("        NMR-GCF分析软件包 (升级版)")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"基础目录: {Config.BASE_PATH}")
    print(f"输出目录: {Config.get_output_path()}")
    print(f"NMR数据目录: {Config.get_nmr_data_path()}")
    print(f"日志文件: {log_file}")
    print("\n新功能:")
    print(f"  - DBSCAN峰聚类: tolerance={Config.PEAK_CLUSTERING['tolerance']}")
    print(f"  - 灵活打分系统: 完美匹配={Config.SCORING_CONFIG['perfect_match']}, "
          f"孤立峰={Config.SCORING_CONFIG['orphan_peak']}")
    print(f"  - 峰分类器: {'启用' if Config.USE_PEAK_CLASSIFIER else '禁用'}")
    if Config.USE_PEAK_CLASSIFIER:
        print(f"    置信度阈值: {Config.PEAK_CONFIDENCE_THRESHOLD}")
    print("=" * 60)
    
    try:
        # 使用内存限制上下文
        with memory_limit(max_percent=90) as protector:
            # 根据选择的模式执行相应的功能
            if args.mode in ["all", "process"]:
                print("\n[1] 执行NMR数据处理...")
                # 检查内存
                status, _, info = protector.check_memory(force=True)
                if status == 'warning':
                    logging.warning("内存接近限制，可能需要降级处理")
                
                return_code, log_file, csv_files = run_nmr_processing(
                    script_path=Config.SCRIPT_PATH,
                    data_dir=Config.get_nmr_data_path(),
                    process_flag=args.process_flag,
                    convert_tabs=not args.no_convert
                )
                
                if return_code != 0:
                    print(f"NMR处理失败，返回码: {return_code}")
                    if args.mode == "process":
                        return return_code
                else:
                    print(f"NMR处理成功完成，生成 {len(csv_files)} 个CSV文件")
                protector.force_gc()
                
            
            if args.mode in ["all", "convert"] or args.convert_only:
                print("\n[2] 执行tab文件转换...")
                if args.organize:
                    print("组织并转换NMR数据...")
                    strain_samples = organize_nmr_samples(
                        Config.get_nmr_data_path(), 
                        Config.get_output_path()
                    )
                    print(f"数据组织完成，共处理了 {len(strain_samples)} 个菌株")
                else:
                    print("转换tab文件为CSV...")
                    csv_files = find_and_convert_tab_files(
                        Config.get_nmr_data_path(), 
                        Config.get_output_path(),
                        in_place=args.in_place
                    )
                    print(f"转换完成，共生成 {len(csv_files)} 个CSV文件")
                protector.force_gc()
            
            if args.mode in ["all", "analyze"]:
                print("\n[3] 执行NMR数据分析...")
                output_dir = analyze_nmr_data(
                    gcf_matrix_path=Config.get_gcf_matrix_path(),
                    nmr_data_path=Config.get_nmr_data_path(),
                    output_path=Config.get_output_path()
                )
                protector.force_gc()
                print(f"NMR数据分析完成，结果保存在: {output_dir}")
            
            if args.mode in ["all", "match"]:
                print("\n[4] 执行GCF-峰匹配分析 (DBSCAN聚类模式)...")
                # 分析前估算内存
                estimated_gb, is_safe = protector.estimate_memory_needed(
                    data_size=1024**3,  # 假设1GB数据
                    factor=5.0
                )
                if not is_safe:
                    logging.warning(
                        f"⚠️ 内存可能不足（需要~{estimated_gb:.1f}GB），"
                        "建议分批处理"
                    )
                    
                    response = input("是否继续？(y/n): ")
                    if response.lower() != 'y':
                        return 1
                
                print(f"使用聚类容差: {Config.PEAK_CLUSTERING['tolerance']} ppm")
                print(f"打分规则: 完美匹配={Config.SCORING_CONFIG['perfect_match']}, "
                      f"缺失峰={Config.SCORING_CONFIG['missing_peak']}, "
                      f"孤立峰={Config.SCORING_CONFIG['orphan_peak']}")
                
                output_dir = match_gcf_peaks(
                    gcf_matrix_path=Config.get_gcf_matrix_path(),
                    nmr_data_path=Config.get_nmr_data_path(),
                    output_path=Config.get_output_path()
                )
                protector.force_gc()
                print(f"GCF-峰匹配分析完成，结果保存在: {output_dir}")
                print("\n输出内容:")
                print("  - 按菌株的文件夹（每个菌株一个子目录）")
                print("  - {strain}_{fraction}_rank.csv（按总分排序的簇）")
                print("  - {strain}_summary.csv（菌株汇总报告）")
                print("  - SQLite数据库（gcf_peak_analysis.db）")
            
            print("\n所有任务完成!")
            # 输出最终内存报告
            final_info = global_protector.get_memory_info()
            print("\n" + "=" * 80)
            print("📊 最终内存报告")
            print("=" * 80)
            print(f"峰值内存: {final_info['peak_memory_gb']:.2f} GB")
            print(f"最终使用: {final_info['system_used_percent']:.1f}%")
            print("=" * 80)
            return 0
            
    except MemoryError as e:
        print(f"\n💥 内存不足: {e}")
        print("\n建议措施：")
        print("  1. 关闭其他程序释放内存")
        print("  2. 增加系统内存（物理内存或交换空间）")
        print("  3. 减小处理批次大小")
        print("  4. 使用 --batch_size 和 --workers 参数调整")
        return 1
    
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        # 清理
        emergency_cleanup()
        return 1
    
    except Exception as e:
        logging.error(f"执行过程中发生错误: {str(e)}")
        logging.exception("详细错误信息:")
        return 1

if __name__ == "__main__":
    sys.exit(main())