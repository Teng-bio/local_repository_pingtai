#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
跨菌株峰聚类器 - DBSCAN 聚类整合
从验证过的新功能移植，提供按 Fraction 的跨菌株峰对齐能力。
"""

import logging
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN


class CrossStrainPeakClusterer:
    """
    跨菌株峰聚类器

    核心功能:
    1) 对同一 Fraction 的所有菌株真实峰做 DBSCAN 聚类
    2) 拆分过宽的簇，避免误合并
    3) 强制中心距离约束，提升簇质量
    4) 生成代表峰（用于后续打分与排名）
    """

    def __init__(self, tolerance=0.002):
        self.tolerance = tolerance
        self.clustering_eps = 2.0 * tolerance
        self.max_cluster_span = 2.0 * tolerance

    def cluster_peaks_by_fraction(self, peak_data):
        """对每个 Fraction 的峰进行跨菌株聚类。

        参数:
            peak_data: loader 返回的峰数据
                {
                    'sample_id': {
                        'peaks': [...],
                        'fraction': 'Fr1',
                        'strain': 'QT1'
                    }, ...
                }

        返回:
            {'Fr1': {rep_peak: {'members': [...], 'strains': [...], 'peak_count': N}}, ...}
        """
        logging.info("=" * 80)
        logging.info("开始跨菌株峰聚类 (DBSCAN)")
        logging.info("=" * 80)

        by_fraction = self._organize_by_fraction(peak_data)
        clustered = {}

        for fraction in sorted(by_fraction.keys()):
            peaks_with_strains = by_fraction[fraction]
            if not peaks_with_strains:
                continue
            clusters = self._cluster_fraction_peaks(peaks_with_strains)
            clustered[fraction] = clusters
            logging.info(f"  {fraction}: {len(peaks_with_strains)} -> {len(clusters)} 簇")

        logging.info("聚类完成")
        return clustered

    def _organize_by_fraction(self, peak_data):
        by_fraction = defaultdict(list)
        for sample_id, info in peak_data.items():
            strain = info.get('strain')
            fraction = info.get('fraction')
            peaks = info.get('peaks', [])
            for p in peaks:
                by_fraction[fraction].append({'peak': float(p), 'strain': strain})
        return by_fraction

    def _cluster_fraction_peaks(self, peaks_with_strains):
        if not peaks_with_strains:
            return {}

        peaks = np.array([p['peak'] for p in peaks_with_strains])
        X = peaks.reshape(-1, 1)
        labels = DBSCAN(eps=self.clustering_eps, min_samples=1).fit(X).labels_

        labels = self._split_wide_clusters(peaks, labels)
        labels = self._enforce_center_distance_constraint(peaks, labels)

        clusters = {}
        for cid in set(labels):
            mask = (labels == cid)
            idx = np.where(mask)[0]
            cpeaks = peaks[mask]
            rep = float(np.median(cpeaks))
            members = [peaks_with_strains[i] for i in idx]
            strains = sorted(set(m['strain'] for m in members))
            clusters[rep] = {
                'members': members,
                'strains': strains,
                'peak_count': len(members)
            }
        return clusters

    def _split_wide_clusters(self, peaks, labels):
        new_labels = labels.copy()
        next_id = int(labels.max()) + 1 if len(labels) else 0
        for cid in set(labels):
            mask = (labels == cid)
            cpeaks = peaks[mask]
            if len(cpeaks) > 1:
                span = float(cpeaks.max() - cpeaks.min())
                if span > self.max_cluster_span:
                    cidx = np.where(mask)[0]
                    order = np.argsort(cpeaks)
                    sidx = cidx[order]
                    speaks = cpeaks[order]
                    sub_groups = self._recursive_split(speaks)
                    for sub in sub_groups:
                        for val in sub:
                            for i, abs_idx in enumerate(sidx):
                                if abs(speaks[i] - val) < 1e-9:
                                    new_labels[abs_idx] = next_id
                                    break
                        next_id += 1
        return new_labels

    def _recursive_split(self, sorted_peaks):
        if len(sorted_peaks) == 0:
            return []
        span = float(sorted_peaks[-1] - sorted_peaks[0])
        if span <= self.max_cluster_span:
            return [sorted_peaks]
        groups = []
        start = 0
        for i in range(1, len(sorted_peaks)):
            if float(sorted_peaks[i] - sorted_peaks[start]) > self.max_cluster_span:
                groups.append(sorted_peaks[start:i])
                start = i
        if start < len(sorted_peaks):
            groups.append(sorted_peaks[start:])
        return groups

    def _enforce_center_distance_constraint(self, peaks, labels):
        new_labels = labels.copy()
        next_id = int(labels.max()) + 1 if len(labels) else 0
        for cid in set(labels):
            mask = (labels == cid)
            cpeaks = peaks[mask]
            if len(cpeaks) > 1:
                rep = np.median(cpeaks)
                max_dist = float(np.max(np.abs(cpeaks - rep)))
                if max_dist > self.tolerance:
                    cidx = np.where(mask)[0]
                    median = np.median(cpeaks)
                    order = np.argsort(np.abs(cpeaks - median))
                    sidx = cidx[order]
                    speaks = cpeaks[order]
                    cur = [speaks[0]]
                    cur_idx = [sidx[0]]
                    for i in range(1, len(speaks)):
                        tmp = cur + [speaks[i]]
                        tmed = np.median(tmp)
                        tmax = float(np.max(np.abs(np.array(tmp) - tmed)))
                        if tmax <= self.tolerance:
                            cur.append(speaks[i])
                            cur_idx.append(sidx[i])
                        else:
                            for idx in cur_idx:
                                new_labels[idx] = next_id
                            next_id += 1
                            cur = [speaks[i]]
                            cur_idx = [sidx[i]]
                    if cur_idx:
                        for idx in cur_idx:
                            new_labels[idx] = next_id
                        next_id += 1
        return new_labels

