#!/usr/bin/env python3
"""
GCF数据处理模块
处理从antiSMASH和MIBiG到BigSCAPE输入、输出，再到可视化CSV的完整流程
"""

import os
import re
import sys
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import pandas as pd
from collections import defaultdict

# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class BGCRecord:
    """BGC记录 - 代表一个生物合成基因簇"""
    bgc_id: str
    source: str  # 'strain' 或 'mibig'
    strain: str or None
    region_id: str or None
    filepath: str
    biosyn_class: str or None
    gcf_id: str or None

    def to_dict(self):
        return {
            'bgc_id': self.bgc_id,
            'source': self.source,
            'strain': self.strain,
            'region_id': self.region_id,
            'filepath': self.filepath,
            'biosyn_class': self.biosyn_class,
            'gcf_id': self.gcf_id
        }


@dataclass
class GCFRecord:
    """GCF记录 - 代表一个基因簇家族"""
    gcf_id: str
    biosyn_class: str
    bgc_ids: List[str]
    strain_ids: Set[str]
    has_mibig: bool
    mibig_ids: List[str]
    novelty_score: float

    def to_dict(self):
        return {
            'gcf_id': self.gcf_id,
            'biosyn_class': self.biosyn_class,
            'bgc_count': len(self.bgc_ids),
            'strain_count': len(self.strain_ids),
            'strain_ids': sorted(list(self.strain_ids)),
            'has_mibig': self.has_mibig,
            'mibig_ids': self.mibig_ids,
            'novelty_score': self.novelty_score
        }


# =============================================================================
# 核心函数
# =============================================================================

def setup_logging(verbose=False):
    """设置日志系统"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_strain_and_region_from_path(file_path: Path, strain_name: str) -> Tuple[str, str]:
    """
    从文件路径解析菌株名和region ID

    示例：
        '/path/003C31/ctg1.region009.gbk' -> ('003C31', 'region009')
        '/path/078C05/contig2.region045.gbk' -> ('078C05', 'region045')
    """
    # 查找带region的文件
    region_pattern = r'region(\d+)'
    match = re.search(region_pattern, file_path.name, re.IGNORECASE)

    if match:
        region_num = match.group(1).zfill(3)  # 补零，如 9 -> 009
        region_id = f"region{region_num}"
    else:
        # 备用方案：从文件名中提取
        region_id = None

    return strain_name, region_id


def find_region_gbk_files(antismash_dir: Path) -> List[Tuple[str, Path]]:
    """
    扫描antiSMASH目录，查找所有带region的gbk文件

    返回：
        List of (strain_name, file_path) tuples
    """
    gbk_files = []

    # 查找所有子目录（每个菌株一个目录）
    for strain_dir in antismash_dir.iterdir():
        if not strain_dir.is_dir():
            continue

        strain_name = strain_dir.name

        # 查找带region的gbk文件
        for gbk_file in strain_dir.glob("*.gbk"):
            if 'region' in gbk_file.name.lower():
                gbk_files.append((strain_name, gbk_file))

    return gbk_files


def prepare_bigscape_input(
    antismash_dir: Path,
    mibig_dir: Path,
    output_dir: Path,
    logger: logging.Logger
) -> Tuple[List[BGCRecord], Path]:
    """
    准备BigSCAPE输入：重命名并整合antiSMASH和MIBiG的gbk文件

    参数:
        antismash_dir: antiSMASH结果根目录
        mibig_dir: MIBiG数据库根目录
        output_dir: 输出目录（BigSCAPE输入）

    返回:
        (List[BGCRecord], output_dir)
        - BGC记录列表
        - 输出目录路径
    """
    logger.info("="*80)
    logger.info("步骤1：准备BigSCAPE输入")
    logger.info("="*80)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    bgc_records = []

    # ----------------------------------------------------------------------
    # 1. 处理antiSMASH结果
    # ----------------------------------------------------------------------
    logger.info("\n[1/2] 扫描antiSMASH结果...")
    antismash_path = Path(antismash_dir)
    if not antismash_path.exists():
        logger.error(f"antiSMASH目录不存在: {antismash_path}")
        raise FileNotFoundError(f"antiSMASH目录不存在: {antismash_path}")

    gbk_files = find_region_gbk_files(antismash_path)
    logger.info(f"找到 {len(gbk_files)} 个带region的gbk文件")

    for strain_name, gbk_file in gbk_files:
        # 解析菌株名和region ID
        _, region_id = parse_strain_and_region_from_path(gbk_file, strain_name)

        if not region_id:
            logger.warning(f"跳过文件（无法解析region）: {gbk_file}")
            continue

        # 构建BGC ID
        bgc_id = f"{strain_name}_{region_id}"

        # 目标文件路径
        dest_file = output_dir / f"{bgc_id}.gbk"

        # 复制文件
        try:
            shutil.copy2(gbk_file, dest_file)
        except Exception as e:
            logger.error(f"复制文件失败 {gbk_file} -> {dest_file}: {e}")
            continue

        # 创建记录
        record = BGCRecord(
            bgc_id=bgc_id,
            source='strain',
            strain=strain_name,
            region_id=region_id,
            filepath=str(dest_file),
            biosyn_class=None,  # 将在解析BigSCAPE后填充
            gcf_id=None  # 将在解析BigSCAPE后填充
        )
        bgc_records.append(record)

        logger.debug(f"  ✓ {bgc_id}")

    logger.info(f"处理了 {len(bgc_records)} 个antiSMASH BGC")

    # ----------------------------------------------------------------------
    # 2. 处理MIBiG数据库
    # ----------------------------------------------------------------------
    logger.info("\n[2/2] 扫描MIBiG数据库...")
    mibig_path = Path(mibig_dir)
    if not mibig_path.exists():
        logger.error(f"MIBiG目录不存在: {mibig_path}")
        raise FileNotFoundError(f"MIBiG目录不存在: {mibig_path}")

    mibig_files = list(mibig_path.glob("BGC*.gbk"))
    logger.info(f"找到 {len(mibig_files)} 个MIBiG gbk文件")

    for mibig_file in mibig_files:
        # 从文件名提取BGC ID（如 BGC0000001.gbk -> BGC0000001）
        bgc_id = mibig_file.stem

        # 目标文件路径（添加前缀便于识别）
        dest_file = output_dir / f"MIBIG_{bgc_id}.gbk"

        # 复制文件
        try:
            shutil.copy2(mibig_file, dest_file)
        except Exception as e:
            logger.error(f"复制文件失败 {mibig_file} -> {dest_file}: {e}")
            continue

        # 创建记录
        record = BGCRecord(
            bgc_id=f"MIBIG_{bgc_id}",
            source='mibig',
            strain=None,
            region_id=None,
            filepath=str(dest_file),
            biosyn_class=None,
            gcf_id=None
        )
        bgc_records.append(record)

        logger.debug(f"  ✓ MIBIG_{bgc_id}")

    logger.info(f"处理了 {len(mibig_files)} 个MIBiG BGC")

    # 保存BGC记录（用于后续解析）
    metadata_file = output_dir / "bgc_metadata.csv"
    df = pd.DataFrame([r.to_dict() for r in bgc_records])
    df.to_csv(metadata_file, index=False)
    logger.info(f"\nBGC元数据已保存: {metadata_file}")

    logger.info(f"\n✅ 总计准备 {len(bgc_records)} 个BGC文件")
    logger.info(f"输出目录: {output_dir}")
    logger.info("\n请运行以下命令启动BigSCAPE：")
    logger.info(f"  cd ~/bigscape/BiG-SCAPE-1.1.5")
    logger.info(f"  python bigscape.py -i {output_dir} --cutoffs 0.3 -o ~/bigscape/bs_output_with_mibig")

    return bgc_records, output_dir


def parse_bigscape_clustering(
    bs_output_dir: Path,
    bgc_metadata_path: Path,
    logger: logging.Logger
) -> Tuple[List[BGCRecord], Dict[str, GCFRecord]]:
    """
    解析BigSCAPE聚类结果，构建BGC-GCF映射

    参数:
        bs_output_dir: BigSCAPE输出目录
        bgc_metadata_path: BGC元数据CSV路径

    返回:
        (List[BGCRecord], Dict[str, GCFRecord])
        - 更新后的BGC记录列表
        - GCF记录字典 {gcf_id: GCFRecord}
    """
    logger.info("="*80)
    logger.info("步骤2：解析BigSCAPE聚类结果")
    logger.info("="*80)

    # 读取BGC元数据
    df_metadata = pd.read_csv(bgc_metadata_path)
    logger.info(f"读取了 {len(df_metadata)} 个BGC记录")

    # 创建BGC记录字典
    bgc_records = {}
    for _, row in df_metadata.iterrows():
        record = BGCRecord(
            bgc_id=row['bgc_id'],
            source=row['source'],
            strain=row['strain'],
            region_id=row['region_id'],
            filepath=row['filepath'],
            biosyn_class=row['biosyn_class'] if pd.notna(row['biosyn_class']) else None,
            gcf_id=row['gcf_id'] if pd.notna(row['gcf_id']) else None
        )
        bgc_records[record.bgc_id] = record

    # ----------------------------------------------------------------------
    # 查找clustering文件
    # ----------------------------------------------------------------------
    network_files_dir = bs_output_dir / "network_files"
    if not network_files_dir.exists():
        logger.error(f"BigSCAPE输出目录不存在: {network_files_dir}")
        raise FileNotFoundError(f"BigSCAPE输出目录不存在: {network_files_dir}")

    # 找到最新的run目录
    run_dirs = [d for d in network_files_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        logger.error("未找到BigSCAPE运行目录")
        raise FileNotFoundError("未找到BigSCAPE运行目录")

    # 选择最新的运行目录
    latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"使用运行目录: {latest_run_dir.name}")

    # 查找所有功能类别文件夹
    class_dirs = [d for d in latest_run_dir.iterdir() if d.is_dir()]
    logger.info(f"找到 {len(class_dirs)} 个功能类别")

    # 存储所有BGC的GCF映射
    bgc_to_gcf = {}
    gcf_to_bgcs = defaultdict(list)

    # ----------------------------------------------------------------------
    # 解析每个类别的聚类文件
    # ----------------------------------------------------------------------
    for class_dir in class_dirs:
        biosyn_class = class_dir.name
        clustering_file = class_dir / f"{biosyn_class}_clustering_c0.30.tsv"

        if not clustering_file.exists():
            logger.warning(f"  跳过（文件不存在）: {clustering_file.name}")
            continue

        logger.info(f"\n解析 {biosyn_class}...")

        # 读取聚类文件
        df = pd.read_csv(clustering_file, sep='\t', dtype=str)

        # 支持多种列名格式（BigSCAPE输出和可能的变体）
        bgc_col = None
        family_col = None

        for col in df.columns:
            col_lower = col.lower().replace(' ', '').replace('#', '')
            if col_lower in ['bgc_name', 'bgcname']:
                bgc_col = col
            elif col_lower in ['familynumber', 'familynum', 'family_number']:
                family_col = col

        if bgc_col is None or family_col is None:
            logger.warning(f"  跳过（列名不匹配）: {clustering_file.name}")
            logger.warning(f"    找到列: {list(df.columns)}")
            logger.warning(f"    期望列: ['BGC_name', 'Family_number'] 或类似")
            continue

        # 处理每一行
        logger.info(f"  找到列: BGC='{bgc_col}', Family='{family_col}'")
        logger.info(f"  数据行数: {len(df)}")

        for _, row in df.iterrows():
            bgc_name = str(row[bgc_col]).strip()
            family_num = str(row[family_col]).strip()

            if not bgc_name or not family_num:
                continue

            gcf_id = f"GCF_{family_num}"

            # 更新BGC记录
            if bgc_name in bgc_records:
                bgc_records[bgc_name].gcf_id = gcf_id
                bgc_records[bgc_name].biosyn_class = biosyn_class
            else:
                # 可能是MIBiG BGC（以MIBIG_开头）
                logger.warning(f"  未找到BGC记录: {bgc_name}")

            # 建立映射
            bgc_to_gcf[bgc_name] = gcf_id
            gcf_to_bgcs[gcf_id].append(bgc_name)

        logger.info(f"  ✓ 处理了 {len(df)} 个BGC")

    logger.info(f"\n建立了 {len(gcf_to_bgcs)} 个GCF")

    # ----------------------------------------------------------------------
    # 构建GCF记录
    # ----------------------------------------------------------------------
    gcf_records = {}

    for gcf_id, bgc_list in gcf_to_bgcs.items():
        # 获取第一个BGC的生物合成类别（所有BGC应该都属于同一类别）
        first_bgc = bgc_list[0]
        biosyn_class = bgc_records[first_bgc].biosyn_class if first_bgc in bgc_records else "Unknown"

        # 收集菌株信息（只来自strain的BGC）
        strain_ids = set()
        mibig_ids = []
        has_mibig = False

        for bgc_id in bgc_list:
            if bgc_id in bgc_records:
                record = bgc_records[bgc_id]
                if record.source == 'strain' and record.strain:
                    strain_ids.add(record.strain)
                elif record.source == 'mibig':
                    has_mibig = True
                    mibig_ids.append(bgc_id)

        # 计算新颖度分数
        if has_mibig:
            # 包含MIBiG BGC -> 认为是已知的
            novelty_score = 0.2
        else:
            # 不包含MIBiG BGC -> 认为是新颖的
            novelty_score = 0.8

        # 创建GCF记录
        gcf_record = GCFRecord(
            gcf_id=gcf_id,
            biosyn_class=biosyn_class,
            bgc_ids=bgc_list,
            strain_ids=strain_ids,
            has_mibig=has_mibig,
            mibig_ids=mibig_ids,
            novelty_score=novelty_score
        )

        gcf_records[gcf_id] = gcf_record

    logger.info(f"\n✅ 构建了 {len(gcf_records)} 个GCF记录")

    # 保存结果
    bgc_df = pd.DataFrame([r.to_dict() for r in bgc_records.values()])
    bgc_df.to_csv(bs_output_dir / "bgc_records.csv", index=False)

    gcf_df = pd.DataFrame([r.to_dict() for r in gcf_records.values()])
    gcf_df.to_csv(bs_output_dir / "gcf_records.csv", index=False)

    logger.info(f"BGC记录已保存: {bs_output_dir / 'bgc_records.csv'}")
    logger.info(f"GCF记录已保存: {bs_output_dir / 'gcf_records.csv'}")

    return list(bgc_records.values()), gcf_records


def export_visualization_csvs(
    bgc_records: List[BGCRecord],
    gcf_records: Dict[str, GCFRecord],
    output_dir: Path,
    logger: logging.Logger
):
    """
    导出可视化所需的CSV文件

    输出：
        - nodes_strain.csv: 菌株节点信息
        - nodes_gcf.csv: GCF节点信息
        - edges_strain_gcf.csv: 菌株-GCF边信息
    """
    logger.info("="*80)
    logger.info("步骤3：导出可视化CSV")
    logger.info("="*80)

    # ----------------------------------------------------------------------
    # 1. 导出菌株节点
    # ----------------------------------------------------------------------
    logger.info("\n[1/3] 生成菌株节点数据...")

    # 统计每个菌株的GCF
    strain_gcf_count = defaultdict(int)
    strain_clusters = {}  # 将通过16S相似性计算，这里先用placeholder

    for bgc_record in bgc_records:
        if bgc_record.source == 'strain' and bgc_record.strain:
            strain_gcf_count[bgc_record.strain] += 1

    strain_nodes = []
    for strain_id, gcf_count in sorted(strain_gcf_count.items()):
        strain_nodes.append({
            'strain_id': strain_id,
            'cluster_id': 1,  # TODO: 从16S聚类计算
            'gcf_count': gcf_count
        })

    df_strains = pd.DataFrame(strain_nodes)
    strains_csv = output_dir / "nodes_strain.csv"
    df_strains.to_csv(strains_csv, index=False)
    logger.info(f"  ✓ 保存: {strains_csv}")
    logger.info(f"  菌株数量: {len(strain_nodes)}")

    # ----------------------------------------------------------------------
    # 2. 导出GCF节点
    # ----------------------------------------------------------------------
    logger.info("\n[2/3] 生成GCF节点数据...")

    gcf_nodes = []
    for gcf_id, gcf_record in sorted(gcf_records.items()):
        strain_count = len(gcf_record.strain_ids)
        bgc_count = len(gcf_record.bgc_ids)
        gcf_nodes.append({
            'gcf_id': gcf_id,
            'biosyn_class': gcf_record.biosyn_class,
            'strain_count': strain_count,
            'bgc_count': bgc_count,
            'strain_ids': ';'.join(sorted(gcf_record.strain_ids)),
            'has_mibig': gcf_record.has_mibig,
            'mibig_ids': ';'.join(gcf_record.mibig_ids) if gcf_record.mibig_ids else '',
            'novelty_score': gcf_record.novelty_score
        })

    df_gcfs = pd.DataFrame(gcf_nodes)
    gcfs_csv = output_dir / "nodes_gcf.csv"
    df_gcfs.to_csv(gcfs_csv, index=False)
    logger.info(f"  ✓ 保存: {gcfs_csv}")
    logger.info(f"  GCF数量: {len(gcf_nodes)}")

    # ----------------------------------------------------------------------
    # 3. 导出边
    # ----------------------------------------------------------------------
    logger.info("\n[3/3] 生成边数据...")

    edges = []
    for bgc_record in bgc_records:
        if bgc_record.source == 'strain' and bgc_record.strain and bgc_record.gcf_id:
            edges.append({
                'strain_id': bgc_record.strain,
                'gcf_id': bgc_record.gcf_id,
                'bgc_id': bgc_record.bgc_id
            })

    df_edges = pd.DataFrame(edges)
    edges_csv = output_dir / "edges_strain_gcf.csv"
    df_edges.to_csv(edges_csv, index=False)
    logger.info(f"  ✓ 保存: {edges_csv}")
    logger.info(f"  边数量: {len(edges)}")

    # ----------------------------------------------------------------------
    # 4. 生成统计报告
    # ----------------------------------------------------------------------
    logger.info("\n[4/3] 生成统计报告...")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("GCF数据统计报告")
    report_lines.append("="*80)
    report_lines.append("")

    # 基本统计
    total_strains = len(strain_nodes)
    total_gcfs = len(gcf_nodes)
    total_edges = len(edges)

    report_lines.append(f"菌株数量: {total_strains}")
    report_lines.append(f"GCF数量: {total_gcfs}")
    report_lines.append(f"边数量: {total_edges}")
    report_lines.append("")

    # GCF类型分布
    if total_gcfs > 0:
        class_counts = df_gcfs['biosyn_class'].value_counts()
        report_lines.append("GCF功能类别分布:")
        for biosyn_class, count in class_counts.items():
            pct = count / total_gcfs * 100
            report_lines.append(f"  {biosyn_class}: {count} ({pct:.1f}%)")
        report_lines.append("")

        # MIBiG匹配统计
        has_mibig_count = df_gcfs['has_mibig'].sum()
        mibig_pct = has_mibig_count / total_gcfs * 100
        report_lines.append(f"包含MIBiG BGC的GCF: {has_mibig_count} ({mibig_pct:.1f}%)")
        report_lines.append(f"新颖GCF（不含MIBiG）: {total_gcfs - has_mibig_count} ({100 - mibig_pct:.1f}%)")
        report_lines.append("")

        # 新颖度分数分布
        novel_gcfs = df_gcfs[df_gcfs['novelty_score'] >= 0.8]
        known_gcfs = df_gcfs[df_gcfs['novelty_score'] <= 0.3]
        report_lines.append(f"高新颖度GCF (score≥0.8): {len(novel_gcfs)}")
        report_lines.append(f"已知GCF (score≤0.3): {len(known_gcfs)}")
        report_lines.append("")
    else:
        report_lines.append("GCF功能类别分布: 无数据")
        report_lines.append("MIBiG匹配: 无数据")
        report_lines.append("新颖度分数: 无数据")
        report_lines.append("")

    # 保存报告
    report_file = output_dir / "data_statistics.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"  ✓ 统计报告: {report_file}")

    logger.info(f"\n✅ 可视化CSV导出完成!")
    logger.info(f"输出目录: {output_dir}")


# =============================================================================
# 主函数（用于测试）
# =============================================================================

def main():
    """测试主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GCF数据处理模块测试"
    )

    parser.add_argument(
        "--antismash-dir",
        type=str,
        required=True,
        help="antiSMASH结果目录"
    )

    parser.add_argument(
        "--mibig-dir",
        type=str,
        required=True,
        help="MIBiG数据库目录"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )

    parser.add_argument(
        "--bigscape-output",
        type=str,
        required=True,
        help="BigSCAPE输出目录"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging(args.verbose)

    try:
        # 步骤1：准备BigSCAPE输入
        bgc_records, input_dir = prepare_bigscape_input(
            antismash_dir=Path(args.antismash_dir),
            mibig_dir=Path(args.mibig_dir),
            output_dir=Path(args.output_dir),
            logger=logger
        )

        print("\n" + "="*80)
        print("准备完成！请运行BigSCAPE然后重新执行脚本解析结果。")
        print("="*80)
        print(f"运行命令:")
        print(f"  cd ~/bigscape/BiG-SCAPE-1.1.5")
        print(f"  python bigscape.py -i {input_dir} --cutoffs 0.3 -o {args.bigscape_output}")
        print()
        print(f"完成后运行:")
        print(f"  python {__file__} --antismash-dir {args.antismash_dir} --mibig-dir {args.mibig_dir} --output-dir {args.output_dir} --bigscape-output {args.bigscape_output}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
