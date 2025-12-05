#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵活打分系统 - 支持可配置的扣分策略
用于跨菌株聚类后的 GCF×簇 得分计算与簇总分统计。
"""

import logging


class FlexibleScorer:
    """灵活的 GCF-峰打分系统。"""

    def __init__(self, score_config=None):
        """
        参数:
            score_config: 打分配置，默认:
                {
                    'perfect_match': 10,
                    'missing_peak': 0,
                    'orphan_peak': -10,
                    'baseline': 1
                }
        """
        self.config = score_config or {
            'perfect_match': 10,
            'missing_peak': 0,
            'orphan_peak': -10,
            'baseline': 1
        }
        logging.info(f"打分配置: {self.config}")

    def score_clustered_peaks(self, clustered_data, strain_gcf_dict):
        """
        基于聚类结果计算所有菌株的得分。

        参数:
            clustered_data: {'Fr1': {rep_peak: {'strains': [...]}}}
            strain_gcf_dict: {'strain': ['GCF1', 'GCF2', ...]}

        返回:
            {'strain': {'Fr1': {'GCF': {'clusters': {rep_peak: score}, 'total': int}}}}
        """
        logging.info("开始计算 GCF-峰得分")

        # 收集所有 GCF
        all_gcfs = sorted({g for gcfs in strain_gcf_dict.values() for g in gcfs})
        all_scores = {}

        for strain in strain_gcf_dict.keys():
            strain_gcfs = set(strain_gcf_dict.get(strain, []))
            strain_scores = {}
            for fraction, clusters in clustered_data.items():
                fr_scores = {}
                for gcf in all_gcfs:
                    has_gcf = (gcf in strain_gcfs)
                    cluster_scores = {}
                    total = 0
                    for rep_peak, info in clusters.items():
                        has_peak = (strain in info.get('strains', []))
                        if has_gcf and has_peak:
                            s = self.config['perfect_match']
                        elif has_gcf and not has_peak:
                            s = self.config['missing_peak']
                        elif (not has_gcf) and has_peak:
                            s = self.config['orphan_peak']
                        else:
                            s = self.config['baseline']
                        cluster_scores[rep_peak] = s
                        total += s
                    fr_scores[gcf] = {'clusters': cluster_scores, 'total': total}
                strain_scores[fraction] = fr_scores
            all_scores[strain] = strain_scores

        logging.info("得分计算完成")
        return all_scores

    def calculate_cluster_total_scores(self, clustered_data, all_strain_scores):
        """
        计算每个聚类簇的跨菌株总得分（用于排名）。

        返回:
            {'Fr1': {rep_peak: {'total_score': int, 'strain_scores': {'strain': score}}}}
        """
        totals = {}
        for fraction in clustered_data.keys():
            frac_totals = {}
            for rep_peak in clustered_data[fraction].keys():
                ss = {}
                # 对所有菌株，取该簇在其所有 GCF 上的最高分（保持与报告逻辑一致）
                for strain, frs in all_strain_scores.items():
                    if fraction not in frs:
                        continue
                    max_s = self.config['baseline']
                    for gcf_data in frs[fraction].values():
                        if rep_peak in gcf_data['clusters']:
                            v = gcf_data['clusters'][rep_peak]
                            if v > max_s:
                                max_s = v
                    ss[strain] = max_s
                frac_totals[rep_peak] = {
                    'total_score': sum(ss.values()),
                    'strain_scores': ss
                }
            totals[fraction] = frac_totals
        return totals

