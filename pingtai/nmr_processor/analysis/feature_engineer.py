#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMR峰特征工程模块
从训练脚本nmr_peak_classifier.py中提取，确保与训练时的特征完全一致
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    高级特征工程器
    实现文档中讨论的所有特征，包括空间特征
    
    注意：这个类必须与训练时使用的类完全一致，
    否则加载pickle模型时会出现兼容性问题
    """
    
    def __init__(self, radius_ppm=0.05):
        """
        参数:
            radius_ppm: PPM轴上的局部分析半径（默认0.05）
        """
        self.radius_ppm = radius_ppm
        self.feature_names = []
    
    def engineer_all_features(self, df, verbose=True):
        """
        生成所有特征
        
        参数:
            df: DataFrame，必须包含 HEIGHT, CONFIDENCE, X_PPM
            verbose: 是否打印进度
        
        返回:
            features_df: 特征DataFrame
        """
        if verbose:
            print("\n" + "="*70)
            print("开始特征工程")
            print("="*70)
        
        features = pd.DataFrame(index=df.index)
        
        # === 1. 基础特征 ===
        if verbose:
            print("\n1. 生成基础特征...")
        features['log_height'] = np.log10(df['HEIGHT'])
        features['confidence'] = df['CONFIDENCE']
        
        if 'X_PPM' in df.columns:
            features['ppm'] = df['X_PPM']
        
        # === 2. 样本内归一化特征 ===
        if 'source_file' in df.columns and verbose:
            print("2. 生成样本内归一化特征...")
            for source in df['source_file'].unique():
                mask = df['source_file'] == source
                sample_data = df.loc[mask, 'HEIGHT']
                
                if len(sample_data) > 1:
                    features.loc[mask, 'height_zscore'] = (
                        sample_data - sample_data.mean()
                    ) / (sample_data.std() + 1e-10)
                else:
                    features.loc[mask, 'height_zscore'] = 0
        else:
            # 全局归一化
            features['height_zscore'] = (df['HEIGHT'] - df['HEIGHT'].mean()) / (df['HEIGHT'].std() + 1e-10)
        
        # === 3. PPM区域特征 ===
        if 'X_PPM' in df.columns:
            if verbose:
                print("3. 生成PPM区域特征...")
            
            # 水峰区域标记
            features['is_water_region'] = (
                (df['X_PPM'] > 4.5) & (df['X_PPM'] < 5.0)
            ).astype(int)
            
            # PPM区域编码（根据化学位移区域）
            features['ppm_region'] = pd.cut(
                df['X_PPM'],
                bins=[0, 1.5, 4.0, 5.5, 7.5, 10.0, 15.0],
                labels=[1, 2, 3, 4, 5, 6]
            ).astype(float)
            features['ppm_region'] = features['ppm_region'].fillna(6)
        
        # === 4. 空间特征（关键！）===
        if 'X_PPM' in df.columns:
            if verbose:
                print("4. 生成空间特征（局部密度、排名、孤立度）...")
            
            spatial_features = self._calculate_spatial_features(df)
            for col in spatial_features.columns:
                features[col] = spatial_features[col]
        
        # === 5. 交互特征 ===
        if verbose:
            print("5. 生成交互特征...")
        features['height_x_conf'] = df['HEIGHT'] * df['CONFIDENCE']
        features['log_height_x_conf'] = features['log_height'] * features['confidence']
        
        # === 6. 相对强度特征 ===
        if verbose:
            print("6. 生成相对强度特征...")
        features['height_percentile'] = df['HEIGHT'].rank(pct=True)
        
        # 样本内百分位
        if 'source_file' in df.columns:
            for source in df['source_file'].unique():
                mask = df['source_file'] == source
                features.loc[mask, 'height_percentile_sample'] = \
                    df.loc[mask, 'HEIGHT'].rank(pct=True)
        else:
            features['height_percentile_sample'] = features['height_percentile']
        
        self.feature_names = features.columns.tolist()
        
        if verbose:
            print(f"\n✓ 特征工程完成")
            print(f"  生成特征数: {len(self.feature_names)}")
            print(f"  特征列表: {self.feature_names}")
        
        return features
    
    def _calculate_spatial_features(self, df):
        """
        计算空间特征（基于PPM位置的局部特征）
        这是区分真实峰和周围噪音峰的关键特征
        """
        n_peaks = len(df)
        
        # 初始化
        local_density = np.zeros(n_peaks)
        local_rank = np.zeros(n_peaks)
        isolation_score = np.zeros(n_peaks)
        height_to_local_max = np.zeros(n_peaks)
        
        # 按PPM排序以加速搜索
        df_sorted = df.sort_values('X_PPM').reset_index(drop=True)
        original_indices = df.index.tolist()
        
        for i in range(n_peaks):
            ppm = df_sorted.iloc[i]['X_PPM']
            height = df_sorted.iloc[i]['HEIGHT']
            
            # 找到局部邻域内的峰
            nearby_mask = np.abs(df_sorted['X_PPM'] - ppm) < self.radius_ppm
            nearby_peaks = df_sorted[nearby_mask]
            
            # 1. 局部密度
            local_density[i] = len(nearby_peaks)
            
            # 2. 局部排名（1=最强）
            if len(nearby_peaks) > 1:
                local_rank[i] = (nearby_peaks['HEIGHT'] > height).sum() + 1
            else:
                local_rank[i] = 1
            
            # 3. 孤立度（自己的强度 / 周围峰的平均强度）
            nearby_without_self = nearby_peaks[nearby_peaks.index != i]
            if len(nearby_without_self) > 0:
                avg_nearby = nearby_without_self['HEIGHT'].mean()
                isolation_score[i] = height / (avg_nearby + 1e-10)
            else:
                isolation_score[i] = 10.0  # 完全孤立
            
            # 4. 与局部最大值的比率
            local_max = nearby_peaks['HEIGHT'].max()
            height_to_local_max[i] = height / local_max
        
        # 构建特征DataFrame（保持原始索引顺序）
        spatial_df = pd.DataFrame({
            'local_density': local_density,
            'local_rank': local_rank,
            'isolation_score': isolation_score,
            'height_to_local_max': height_to_local_max
        }, index=df_sorted.index)
        
        # 恢复原始索引顺序
        spatial_df = spatial_df.reindex(original_indices)
        
        return spatial_df
    
    def get_feature_names(self):
        """获取特征名称列表"""
        return self.feature_names.copy()
    
    def __repr__(self):
        return f"FeatureEngineer(radius_ppm={self.radius_ppm}, n_features={len(self.feature_names)})"


# 确保这个类可以被pickle正确序列化和反序列化
__all__ = ['FeatureEngineer']
