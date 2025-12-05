#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GCF-峰匹配（升级版）- 统一使用 DBSCAN 聚类 + 灵活打分 + 按菌株报告。
数据库仅作为存储使用；旧系统的报表输出已禁用以节省IO和时间。
"""

import os
import logging
import sqlite3
from pathlib import Path
import pandas as pd

from ..config import Config
from ..utils.decorators import memory_monitor
from ..utils.memory_protector import memory_limit
from ..data.loader import load_gcf_matrix, load_nmr_data

# 新流程模块
from .peak_clusterer import CrossStrainPeakClusterer
from .flexible_scorer import FlexibleScorer
from .strain_reporter import StrainRankReporter


class SQLiteOutputManager:
    """SQLite数据库输出管理器 - 仅存储用途（不再生成旧报表）"""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.db_path = self.output_path / "gcf_peak_analysis.db"
        self.output_path.mkdir(exist_ok=True)
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS gcf_peak_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strain TEXT NOT NULL,
                gcf TEXT NOT NULL,
                fraction TEXT,
                chemical_shift REAL NOT NULL,
                score INTEGER NOT NULL,
                peak_intensity REAL,
                region_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strain, gcf, fraction, chemical_shift) ON CONFLICT REPLACE
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_strain_gcf ON gcf_peak_scores(strain, gcf)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_score ON gcf_peak_scores(score)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_chemical_shift ON gcf_peak_scores(chemical_shift)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_strain ON gcf_peak_scores(strain)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_fraction ON gcf_peak_scores(fraction)')
        conn.commit()
        conn.close()
        logging.info(f"SQLite数据库已初始化: {self.db_path}")

    def _get_region_index(self, chemical_shift: float) -> int:
        if 0.0 <= chemical_shift < 5.0:
            return 0
        elif 5.0 <= chemical_shift < 10.0:
            return 1
        elif 10.0 <= chemical_shift <= 15.0:
            return 2
        else:
            return -1

    def save_batch_records(self, records):
        if not records:
            return 0
        conn = sqlite3.connect(self.db_path)
        data = []
        for r in records:
            region_idx = self._get_region_index(r['bin'])
            data.append((
                r['strain'],
                r['gcf'],
                r.get('fraction', 'unknown'),
                float(r['bin']),
                int(r['score']),
                r.get('peak_intensity', None),
                region_idx
            ))
        conn.executemany('''
            INSERT OR REPLACE INTO gcf_peak_scores
            (strain, gcf, fraction, chemical_shift, score, peak_intensity, region_index)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        conn.close()
        logging.info(f"已保存 {len(data)} 条记录到SQLite数据库")
        return len(data)

    def generate_summary_reports(self):
        # 新流程：数据库仅作存储，不再生成旧系统风格的汇总输出
        logging.info("跳过旧系统汇总报告生成（仅保存记录，报告以新Rank为准）")
        return


class EnhancedGCFAnalyzer:
    """简化分析器：聚类→打分→新报告→数据库存储（仅存储）。"""

    @memory_monitor
    def __init__(self, gcf_data, peak_data):
        if not gcf_data:
            raise ValueError("GCF数据为空")
        if not peak_data:
            logging.warning("峰数据为空")
        self.gcf_data = gcf_data
        self.peak_data = peak_data
        self.sqlite_manager = SQLiteOutputManager(Config.get_output_path())

    def _save_to_database_clustered(self, all_scores):
        """写入SQLite；可配置是否存储baseline=1的记录以降低IO/空间。"""
        store_baseline = Config.DB_STORAGE.get('store_baseline', False)
        baseline = Config.SCORING_CONFIG.get('baseline', 1)
        records = []
        for strain, fractions in all_scores.items():
            for fraction, gcfs in fractions.items():
                for gcf, gcf_data in gcfs.items():
                    for rep_peak, score in gcf_data.get('clusters', {}).items():
                        if not store_baseline and score == baseline:
                            continue
                        records.append({
                            'strain': str(strain),
                            'gcf': str(gcf),
                            'fraction': str(fraction),
                            'bin': float(rep_peak),
                            'score': int(score)
                        })
        if records:
            self.sqlite_manager.save_batch_records(records)
        else:
            logging.info("无需要写入的记录（可能仅包含baseline，且已禁用存储）")

    def analyze_with_clustering(self, output_path):
        """DBSCAN聚类 → 灵活打分 → 新Rank报告 → 写入SQLite（仅存储）。"""
        tol = Config.PEAK_CLUSTERING.get('tolerance', 0.002)
        with memory_limit(max_percent=85):
            # 1) 跨菌株峰聚类
            clusterer = CrossStrainPeakClusterer(tolerance=tol)
            clustered_data = clusterer.cluster_peaks_by_fraction(self.peak_data)
            # 2) 打分
            scorer = FlexibleScorer(Config.SCORING_CONFIG)
            all_scores = scorer.score_clustered_peaks(clustered_data, self.gcf_data)
            # 3) 簇总分（用于排名）
            cluster_totals = scorer.calculate_cluster_total_scores(clustered_data, all_scores)
            # 4) 新报告
            reporter = StrainRankReporter(output_path)
            reporter.generate_all_reports(clustered_data, all_scores, cluster_totals)
            # 5) 数据库存储
            self._save_to_database_clustered(all_scores)
        return output_path

    @memory_monitor
    def generate_outputs(self):
        # 兼容旧接口：核心输出已在 analyze_with_clustering 完成
        logging.info("DBSCAN聚类与打分结果已写入数据库，并生成按菌株Rank与汇总报告")


def match_gcf_peaks(gcf_matrix_path=None, nmr_data_path=None, output_path=None):
    """执行 GCF-峰匹配（DBSCAN 聚类模式）；保持旧函数名兼容"""
    if output_path is None:
        output_path = Config.get_output_path()
    if gcf_matrix_path is None:
        gcf_matrix_path = Config.get_gcf_matrix_path()
    if nmr_data_path is None:
        nmr_data_path = Config.get_nmr_data_path()

    os.makedirs(output_path, exist_ok=True)

    logging.info("加载GCF矩阵数据...")
    gcf_data = load_gcf_matrix(gcf_matrix_path)

    logging.info("加载NMR峰数据...")
    peak_data = load_nmr_data(nmr_data_path)

    if not peak_data:
        logging.error("未加载到任何NMR峰数据，终止分析")
        return output_path

    logging.info("开始 GCF-峰匹配（DBSCAN 聚类）...")
    analyzer = EnhancedGCFAnalyzer(gcf_data, peak_data)
    analyzer.analyze_with_clustering(output_path)
    logging.info(f"分析完成，结果保存到: {output_path}")
    return output_path
