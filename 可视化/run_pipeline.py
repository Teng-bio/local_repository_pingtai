#!/usr/bin/env python3
"""
GCF数据处理与可视化完整流程
整合gcf_data.py和gcf_network_plot.py，提供一键式操作
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
import argparse
from typing import Optional

# 导入自定义模块
try:
    from gcf_data import (
        setup_logging,
        prepare_bigscape_input,
        parse_bigscape_clustering,
        export_visualization_csvs
    )
    from gcf_network_plot import GCFNetworkVisualizer
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在同一目录下运行此脚本")
    sys.exit(1)

# 尝试导入YAML（可选）
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# 配置加载函数
# =============================================================================

def load_config(config_file: Path) -> dict:
    """加载YAML配置文件"""
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def merge_config_and_args(config: dict, args) -> argparse.Namespace:
    """
    合并配置文件和命令行参数
    命令行参数优先于配置文件
    """
    # 从配置文件创建命名空间
    namespace = argparse.Namespace()

    # 合并配置项
    for key, value in config.items():
        if hasattr(namespace, key):
            continue  # 跳过已存在的属性
        setattr(namespace, key, value)

    # 合并命令行参数（命令行参数优先）
    for key, value in vars(args).items():
        if value is not None:
            setattr(namespace, key, value)

    return namespace


# =============================================================================
# 主流程类
# =============================================================================

class GCFAnalysisPipeline:
    """GCF分析完整流程"""

    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(args.verbose)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.bigscape_input_dir = Path(args.bigscape_input_dir) if args.bigscape_input_dir else None
        self.bigscape_output_dir = Path(args.bigscape_output_dir) if args.bigscape_output_dir else None

    def run(self):
        """运行完整流程"""
        try:
            # 步骤1：准备BigSCAPE输入
            self._step1_prepare_input()

            # 步骤2：运行BigSCAPE（可选）
            if self.args.auto_run_bigscape:
                self._step2_run_bigscape()
            else:
                self._step2_prompt_bigscape()

            # 步骤3：解析BigSCAPE输出
            self._step3_parse_output()

            # 步骤4：生成可视化
            self._step4_visualize()

            # 完成
            self._print_summary()

            return True

        except Exception as e:
            self.logger.error(f"\n❌ 流程失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _step1_prepare_input(self):
        """步骤1：准备BigSCAPE输入"""
        print("\n" + "="*80)
        print("步骤 1/4：准备BigSCAPE输入")
        print("="*80)

        if not self.args.antismash_dir:
            raise ValueError("必须指定antiSMASH目录 (--antismash-dir)")

        if not self.args.mibig_dir:
            raise ValueError("必须指定MIBiG目录 (--mibig-dir)")

        # 检查路径
        antismash_path = Path(self.args.antismash_dir)
        mibig_path = Path(self.args.mibig_dir)

        if not antismash_path.exists():
            raise FileNotFoundError(f"antiSMASH目录不存在: {antismash_path}")

        if not mibig_path.exists():
            raise FileNotFoundError(f"MIBiG目录不存在: {mibig_path}")

        # 确定输出目录
        if not self.bigscape_input_dir:
            self.bigscape_input_dir = self.data_dir / "bigscape_input"
            self.logger.info(f"使用默认BigSCAPE输入目录: {self.bigscape_input_dir}")

        # 准备输入
        bgc_records, _ = prepare_bigscape_input(
            antismash_dir=antismash_path,
            mibig_dir=mibig_path,
            output_dir=self.bigscape_input_dir,
            logger=self.logger
        )

        # 保存BGC记录路径
        self.bgc_metadata_path = self.bigscape_input_dir / "bgc_metadata.csv"
        self.input_dir = self.bigscape_input_dir

        print(f"\n✅ 步骤1完成：准备了 {len(bgc_records)} 个BGC文件")
        print(f"   输入目录: {self.input_dir}")

    def _step2_run_bigscape(self):
        """步骤2：自动运行BigSCAPE"""
        print("\n" + "="*80)
        print("步骤 2/4：运行BigSCAPE")
        print("="*80)

        if not self.bigscape_output_dir:
            self.bigscape_output_dir = self.data_dir / "bigscape_output"
            self.logger.info(f"使用默认BigSCAPE输出目录: {self.bigscape_output_dir}")

        # 构建命令
        cmd = [
            "python", "bigscape.py",
            "-i", str(self.input_dir),
            "--cutoffs", "0.3",
            "-o", str(self.bigscape_output_dir)
        ]

        print(f"\n运行命令: {' '.join(cmd)}")
        print(f"预计运行时间: 几分钟到几小时（取决于数据量）")
        print(f"输出目录: {self.bigscape_output_dir}")
        print("\n正在运行BigSCAPE...")

        # 运行命令
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            cwd="~/bigscape/BiG-SCAPE-1.1.5",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # 实时输出
        for line in process.stdout:
            print(line.rstrip())

        # 等待完成
        return_code = process.wait()
        elapsed_time = time.time() - start_time

        if return_code != 0:
            raise RuntimeError(f"BigSCAPE运行失败 (返回码: {return_code})")

        print(f"\n✅ 步骤2完成：BigSCAPE运行成功 ({elapsed_time/60:.1f} 分钟)")
        print(f"   输出目录: {self.bigscape_output_dir}")

    def _step2_prompt_bigscape(self):
        """步骤2：提示用户手动运行BigSCAPE"""
        print("\n" + "="*80)
        print("步骤 2/4：运行BigSCAPE")
        print("="*80)

        if not self.bigscape_output_dir:
            self.bigscape_output_dir = self.data_dir / "bigscape_output"
            self.logger.info(f"使用默认BigSCAPE输出目录: {self.bigscape_output_dir}")

        print("\n请手动运行以下命令：\n")
        print(f"  cd ~/bigscape/BiG-SCAPE-1.1.5")
        print(f"  python bigscape.py -i {self.input_dir} --cutoffs 0.3 -o {self.bigscape_output_dir}\n")

        print(f"预计运行时间: 几分钟到几小时（取决于数据量）")
        print(f"完成后请按回车键继续...")

        try:
            input("按回车键继续...")
        except KeyboardInterrupt:
            print("\n\n❌ 用户中断")
            sys.exit(1)

        print(f"\n✅ 步骤2完成：BigSCAPE运行完成")
        print(f"   输出目录: {self.bigscape_output_dir}")

    def _step3_parse_output(self):
        """步骤3：解析BigSCAPE输出"""
        print("\n" + "="*80)
        print("步骤 3/4：解析BigSCAPE输出")
        print("="*80)

        if not self.bigscape_output_dir or not self.bigscape_output_dir.exists():
            raise FileNotFoundError(f"BigSCAPE输出目录不存在: {self.bigscape_output_dir}")

        # 检查必要文件
        network_files_dir = self.bigscape_output_dir / "network_files"
        if not network_files_dir.exists():
            raise FileNotFoundError(f"BigSCAPE网络文件目录不存在: {network_files_dir}")

        # 解析聚类结果
        bgc_records, gcf_records = parse_bigscape_clustering(
            bs_output_dir=self.bigscape_output_dir,
            bgc_metadata_path=self.bgc_metadata_path,
            logger=self.logger
        )

        # 导出可视化CSV
        export_visualization_csvs(
            bgc_records=bgc_records,
            gcf_records=gcf_records,
            output_dir=self.data_dir,
            logger=self.logger
        )

        # 保存路径
        self.nodes_strain_csv = self.data_dir / "nodes_strain.csv"
        self.nodes_gcf_csv = self.data_dir / "nodes_gcf.csv"
        self.edges_csv = self.data_dir / "edges_strain_gcf.csv"

        print(f"\n✅ 步骤3完成：解析完成并导出CSV")
        print(f"   数据目录: {self.data_dir}")

    def _step4_visualize(self):
        """步骤4：生成可视化"""
        print("\n" + "="*80)
        print("步骤 4/4：生成可视化")
        print("="*80)

        # 检查文件
        for file_path in [self.nodes_strain_csv, self.nodes_gcf_csv, self.edges_csv]:
            if not file_path.exists():
                raise FileNotFoundError(f"必需文件不存在: {file_path}")

        # 创建可视化器
        visualizer = GCFNetworkVisualizer(output_dir=self.output_dir)

        # 检查是否有基因组目录（用于16S聚类）
        genome_dir = None
        if hasattr(self.args, 'genome_dir') and self.args.genome_dir:
            genome_path = Path(self.args.genome_dir)
            if genome_path.exists() and genome_path.is_dir():
                genome_dir = genome_path
                print(f"\n📋 检测到基因组目录: {genome_dir}")
                print("将使用16S rRNA聚类（沿用v4.4逻辑）")
            else:
                # 如果指定了genome_dir但路径不存在，报错
                print(f"\n❌ 基因组目录不存在: {self.args.genome_dir}")
                print("请检查路径是否正确，或将配置文件中的genome_dir设置为null")
                print("如果不需要16S聚类，删除example_config.yaml中的genome_dir行")
                raise FileNotFoundError(f"基因组目录不存在: {self.args.genome_dir}")
        else:
            print(f"\n📋 未提供基因组目录")
            print("将尝试读取现有的16s_similarity_matrix.csv文件")
            print("如需计算16S序列，请在配置文件中设置genome_dir")

        # 运行可视化
        visualizer.run(
            nodes_strain_csv=self.nodes_strain_csv,
            nodes_gcf_csv=self.nodes_gcf_csv,
            edges_csv=self.edges_csv,
            genome_dir=genome_dir,  # None表示不计算，直接读取现有矩阵
            figsize=tuple(self.args.figsize) if self.args.figsize else (14, 14),
            dpi=self.args.dpi
        )

        print(f"\n✅ 步骤4完成：可视化图像生成完成")

    def _print_summary(self):
        """打印总结"""
        print("\n" + "="*80)
        print("🎉 分析完成！")
        print("="*80)

        print(f"\n📂 输出目录: {self.output_dir}")
        print(f"\n📊 主要输出文件:")
        print(f"   - 网络图 (PDF): {self.output_dir / 'strain_gcf_network.pdf'}")
        print(f"   - 网络图 (PNG): {self.output_dir / 'strain_gcf_network.png'}")
        print(f"   - 数据统计: {self.output_dir / 'network_statistics.txt'}")
        print(f"   - 可视化数据:")
        print(f"     * 菌株节点: {self.data_dir / 'nodes_strain.csv'}")
        print(f"     * GCF节点: {self.data_dir / 'nodes_gcf.csv'}")
        print(f"     * 边数据: {self.data_dir / 'edges_strain_gcf.csv'}")
        print(f"   - 原始数据:")
        print(f"     * BGC记录: {self.bigscape_output_dir / 'bgc_records.csv'}")
        print(f"     * GCF记录: {self.bigscape_output_dir / 'gcf_records.csv'}")

        print(f"\n📈 数据统计:")
        import pandas as pd
        df_gcf = pd.read_csv(self.nodes_gcf_csv)
        df_strain = pd.read_csv(self.nodes_strain_csv)
        print(f"   - 菌株数量: {len(df_strain)}")
        print(f"   - GCF数量: {len(df_gcf)}")
        print(f"   - 含MIBiG的GCF: {df_gcf['has_mibig'].sum()}")
        print(f"   - 新颖GCF: {len(df_gcf) - df_gcf['has_mibig'].sum()}")

        print("="*80)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GCF数据处理与可视化完整流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  1. 基础用法（需要手动运行BigSCAPE）:
     python run_pipeline.py \\
       --antismash-dir /path/to/antismash_results/ \\
       --mibig-dir /path/to/mibig_gbk/ \\
       --output-dir /path/to/output/

     然后按照提示手动运行BigSCAPE命令

  2. 使用配置文件:
     python run_pipeline.py --config example_config.yaml
     # 或
     python run_pipeline.py --config /path/to/custom_config.yaml

  3. 自动运行BigSCAPE:
     python run_pipeline.py \\
       --antismash-dir /path/to/antismash_results/ \\
       --mibig-dir /path/to/mibig_gbk/ \\
       --output-dir /path/to/output/ \\
       --auto-run-bigscape

  4. 指定目录:
     python run_pipeline.py \\
       --antismash-dir /path/to/antismash_results/ \\
       --mibig-dir /path/to/mibig_gbk/ \\
       --bigscape-input-dir /path/to/bigscape_input/ \\
       --bigscape-output-dir /path/to/bigscape_output/ \\
       --output-dir /path/to/output/

输出文件:
  - strain_gcf_network.pdf: 网络图（PDF格式，高分辨率）
  - strain_gcf_network.png: 网络图（PNG格式，预览用）
  - network_statistics.txt: 网络统计报告
  - data/*.csv: 可视化用CSV文件
        """
    )

    # 必需参数（如果未使用配置文件）
    parser.add_argument(
        "--antismash-dir",
        type=str,
        default=None,
        help="antiSMASH结果目录路径"
    )

    parser.add_argument(
        "--mibig-dir",
        type=str,
        default=None,
        help="MIBiG数据库目录路径"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录路径"
    )

    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML配置文件路径（可选）"
    )

    # 可选参数
    parser.add_argument(
        "--bigscape-input-dir",
        type=str,
        default=None,
        help="BigSCAPE输入目录路径（可选，默认：output_dir/data/bigscape_input）"
    )

    parser.add_argument(
        "--bigscape-output-dir",
        type=str,
        default=None,
        help="BigSCAPE输出目录路径（可选，默认：output_dir/data/bigscape_output）"
    )

    # 运行模式
    parser.add_argument(
        "--auto-run-bigscape",
        action="store_true",
        help="自动运行BigSCAPE（可选，默认：提示用户手动运行）"
    )

    # 可视化参数
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=('WIDTH', 'HEIGHT'),
        help="图像大小（宽度，高度），单位英寸（默认：14 14）"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="图像分辨率（默认：300）"
    )

    # 其他
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )

    # 16S聚类
    parser.add_argument(
        "--genome-dir",
        type=str,
        default=None,
        help="基因组FASTA文件目录（用于16S聚类，沿用v4.4逻辑）"
    )

    args = parser.parse_args()

    # 加载配置文件（如果指定了）
    if args.config:
        if not YAML_AVAILABLE:
            print("❌ 未安装PyYAML，无法使用配置文件")
            print("请安装: pip install pyyaml")
            sys.exit(1)

        try:
            config = load_config(Path(args.config))
            args = merge_config_and_args(config, args)
            print(f"✅ 已加载配置文件: {args.config}")
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            sys.exit(1)

    # 验证必需参数
    if args.figsize and len(args.figsize) != 2:
        parser.error("--figsize 需要两个值：宽度和高度")

    # 检查必需的路径参数
    if not args.antismash_dir:
        print("❌ 错误: 必须指定 --antismash-dir")
        print("   或使用 --config 指定配置文件")
        sys.exit(1)

    if not args.mibig_dir:
        print("❌ 错误: 必须指定 --mibig-dir")
        print("   或使用 --config 指定配置文件")
        sys.exit(1)

    if not args.output_dir:
        print("❌ 错误: 必须指定 --output-dir")
        print("   或使用 --config 指定配置文件")
        sys.exit(1)

    # 创建并运行流程
    pipeline = GCFAnalysisPipeline(args)

    print("\n" + "="*80)
    print("GCF数据处理与可视化完整流程")
    print("="*80)
    print(f"antiSMASH目录: {args.antismash_dir}")
    print(f"MIBiG目录: {args.mibig_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行模式: {'自动运行BigSCAPE' if args.auto_run_bigscape else '手动运行BigSCAPE'}")
    print("="*80)

    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
