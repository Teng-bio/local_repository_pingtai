#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
按菌株生成 Rank 报告（每个菌株一个文件夹）。
"""

import logging
from pathlib import Path
import pandas as pd


class StrainRankReporter:
    """按菌株生成 Rank 与汇总报告。"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_all_reports(self, clustered_data, all_scores, cluster_totals):
        logging.info("=" * 80)
        logging.info("生成按菌株 Rank 报告")
        logging.info("=" * 80)
        for strain in all_scores.keys():
            self._generate_strain_reports(
                strain, clustered_data, all_scores[strain], cluster_totals
            )
        logging.info("报告生成完成")

    def _generate_strain_reports(self, strain, clustered_data, strain_scores, cluster_totals):
        sdir = self.output_dir / strain
        sdir.mkdir(exist_ok=True)
        for fraction in strain_scores.keys():
            self._generate_fraction_rank(
                strain,
                fraction,
                clustered_data.get(fraction, {}),
                strain_scores[fraction],
                cluster_totals.get(fraction, {}),
                sdir / f"{strain}_{fraction}_rank.csv",
            )
        self._generate_strain_summary(strain, strain_scores, sdir / f"{strain}_summary.csv")

    def _generate_fraction_rank(self, strain, fraction, clusters, strain_fraction_scores, cluster_totals, output_file):
        rows = []
        # 按簇总分从高到低排序
        sorted_clusters = sorted(
            cluster_totals.items(), key=lambda x: x[1]['total_score'], reverse=True
        )
        for rank, (rep_peak, total_info) in enumerate(sorted_clusters, 1):
            strain_max = 1  # 基线分
            matched_gcfs = []
            for gcf, gcf_data in strain_fraction_scores.items():
                if rep_peak in gcf_data['clusters']:
                    sc = gcf_data['clusters'][rep_peak]
                    if sc > strain_max:
                        strain_max = sc
                    if sc == 10:
                        matched_gcfs.append(f"{gcf}(10)")
                    elif sc == -10:
                        matched_gcfs.append(f"{gcf}(-10)")
            details = ", ".join([f"{s}:{v}" for s, v in sorted(total_info['strain_scores'].items())])
            rows.append({
                'rank': rank,
                'cluster_peak': f"{rep_peak:.4f}",
                'total_score': total_info['total_score'],
                'strain_max_score': strain_max,
                'gcf_list': '; '.join(matched_gcfs) if matched_gcfs else 'None',
                'strain_details': details
            })
        pd.DataFrame(rows).to_csv(output_file, index=False)
        logging.info(f"保存: {output_file.name}")

    def _generate_strain_summary(self, strain, strain_scores, output_file):
        summary = []
        for fraction, gcfs in strain_scores.items():
            if gcfs:
                first_gcf = next(iter(gcfs))
                total_clusters = len(gcfs[first_gcf]['clusters'])
            else:
                total_clusters = 0
            high = 0
            for gcf_data in gcfs.values():
                # 一个 GCF 有任意簇得分>=10 则计入高分一次
                if any(s >= 10 for s in gcf_data['clusters'].values()):
                    high += 1
            summary.append({
                'fraction': fraction,
                'total_clusters': total_clusters,
                'high_score_clusters': high,
                'gcf_count': len(gcfs)
            })
        pd.DataFrame(summary).to_csv(output_file, index=False)
        logging.info(f"保存: {output_file.name}")

