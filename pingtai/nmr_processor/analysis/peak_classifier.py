#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
峰分类器模块 - 使用预训练模型过滤低置信度峰
修复版本：添加了FeatureEngineer类的兼容性支持
"""

import pandas as pd
import numpy as np
import logging
import joblib
import sys
from pathlib import Path

# 导入FeatureEngineer类
try:
    from .feature_engineer import FeatureEngineer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from feature_engineer import FeatureEngineer
    except ImportError:
        # 最后尝试从当前目录导入
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from feature_engineer import FeatureEngineer

# 为了兼容旧模型文件，将FeatureEngineer添加到__main__命名空间
# 这样pickle可以从__main__模块找到这个类
if hasattr(sys.modules.get('__main__', {}), '__dict__'):
    sys.modules['__main__'].__dict__['FeatureEngineer'] = FeatureEngineer

try:
    from ..config import Config
except ImportError:
    # 如果在独立运行模式，使用默认配置
    class Config:
        PEAK_CLASSIFIER_MODEL_PATH = "./nmr_peak_classifier.pkl"

class PeakClassifierFilter:
    """峰分类器过滤器 - 使用机器学习模型过滤噪音峰"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        初始化峰分类器
        
        参数:
            model_path: 模型文件路径（.pkl）
            confidence_threshold: 置信度阈值，低于此值的峰将被过滤
        """
        if model_path is None:
            model_path = Config.PEAK_CLASSIFIER_MODEL_PATH
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classifier = None
        self.load_model()
    
    def load_model(self):
        """
        加载预训练模型
        
        修复了FeatureEngineer类无法找到的问题
        """
        try:
            # 首先尝试正常加载
            model_data = joblib.load(self.model_path)
            
            # 验证模型数据结构
            if not isinstance(model_data, dict):
                raise ValueError("模型文件格式不正确：期望dict类型")
            
            if 'model' not in model_data or 'feature_engineer' not in model_data:
                raise ValueError("模型文件缺少必需的组件（model或feature_engineer）")
            
            self.classifier = {
                'model': model_data['model'],
                'feature_engineer': model_data['feature_engineer']
            }
            
            # 验证feature_engineer是正确的类型
            if not isinstance(self.classifier['feature_engineer'], FeatureEngineer):
                logging.warning(
                    f"feature_engineer类型不匹配: "
                    f"期望FeatureEngineer，得到{type(self.classifier['feature_engineer'])}"
                )
            
            logging.info(f"✓ 峰分类模型已加载: {self.model_path}")
            logging.info(f"  模型类型: {type(self.classifier['model']).__name__}")
            logging.info(f"  特征工程器: {self.classifier['feature_engineer']}")
            
        except FileNotFoundError:
            error_msg = f"模型文件不存在: {self.model_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        except Exception as e:
            error_msg = f"加载峰分类模型失败: {e}"
            logging.error(error_msg)
            logging.error(f"  模型路径: {self.model_path}")
            logging.error(f"  错误类型: {type(e).__name__}")
            
            # 提供更详细的调试信息
            if "Can't get attribute" in str(e):
                logging.error(
                    "这是一个pickle反序列化错误。可能的原因："
                    "\n  1. 模型训练时使用的类在当前环境中不可用"
                    "\n  2. 类定义的模块路径已更改"
                    "\n  3. 需要重新训练模型或更新代码"
                )
            raise
    
    def filter_peaks(self, peaks_df):
        """
        过滤低置信度峰
        
        参数:
            peaks_df: DataFrame，必须包含 HEIGHT, CONFIDENCE, X_PPM 列
        
        返回:
            filtered_df: 过滤后的DataFrame
            stats: 过滤统计信息
        """
        if self.classifier is None:
            logging.warning("模型未加载，返回原始数据")
            stats = {'total': len(peaks_df), 'kept': len(peaks_df), 'filtered': 0}
            return peaks_df, stats
        
        if len(peaks_df) == 0:
            return peaks_df, {'total': 0, 'kept': 0, 'filtered': 0}
        
        # 检查必需列
        required_cols = ['HEIGHT', 'X_PPM']
        missing_cols = [col for col in required_cols if col not in peaks_df.columns]
        if missing_cols:
            logging.warning(f"缺少必需列: {missing_cols}，返回原始数据")
            stats = {'total': len(peaks_df), 'kept': len(peaks_df), 'filtered': 0}
            return peaks_df, stats
        
        try:
            # 特征工程
            feature_engineer = self.classifier['feature_engineer']
            X = feature_engineer.engineer_all_features(peaks_df, verbose=False)
            
            # 处理缺失值
            X = X.fillna(X.median())
            
            # 如果还有NaN，用0填充
            X = X.fillna(0)
            
            # 预测
            model = self.classifier['model']
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
            
            # 添加预测结果到DataFrame
            peaks_df = peaks_df.copy()
            peaks_df['is_real_peak'] = predictions
            peaks_df['real_peak_probability'] = probabilities
            
            # 过滤低置信度峰
            filtered_df = peaks_df[probabilities >= self.confidence_threshold].copy()
            
            # 统计信息
            stats = {
                'total': len(peaks_df),
                'kept': len(filtered_df),
                'filtered': len(peaks_df) - len(filtered_df),
                'filter_rate': (len(peaks_df) - len(filtered_df)) / len(peaks_df) * 100 if len(peaks_df) > 0 else 0
            }
            
            logging.info(
                f"峰过滤完成: 总数={stats['total']}, "
                f"保留={stats['kept']}, 过滤={stats['filtered']} "
                f"({stats['filter_rate']:.1f}%)"
            )
            
            return filtered_df, stats
            
        except Exception as e:
            logging.error(f"峰过滤过程中出错: {e}")
            logging.exception("详细错误信息:")
            # 出错时返回原始数据
            stats = {'total': len(peaks_df), 'kept': len(peaks_df), 'filtered': 0}
            return peaks_df, stats
    
    def batch_filter_csv_files(self, csv_dir, output_dir=None, pattern="*_peaks.csv"):
        """
        批量过滤CSV文件中的峰
        
        参数:
            csv_dir: CSV文件目录
            output_dir: 输出目录（默认在原目录创建filtered子目录）
            pattern: 文件匹配模式
        
        返回:
            summary: 处理摘要
        """
        csv_dir = Path(csv_dir)
        
        if output_dir is None:
            output_dir = csv_dir / "filtered"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        csv_files = list(csv_dir.rglob(pattern))
        logging.info(f"找到 {len(csv_files)} 个CSV文件")
        
        summary = []
        
        for csv_file in csv_files:
            try:
                # 读取CSV
                df = pd.read_csv(csv_file)
                
                # 检查必需列
                required_cols = ['HEIGHT', 'X_PPM']
                if not all(col in df.columns for col in required_cols):
                    logging.warning(f"跳过文件 {csv_file.name}: 缺少必需列")
                    continue
                
                # 过滤峰
                filtered_df, stats = self.filter_peaks(df)
                
                # 保存结果
                output_file = output_dir / csv_file.name.replace('.csv', '_filtered.csv')
                filtered_df.to_csv(output_file, index=False)
                
                # 记录统计
                summary.append({
                    'file': csv_file.name,
                    'total_peaks': stats['total'],
                    'kept_peaks': stats['kept'],
                    'filtered_peaks': stats['filtered'],
                    'filter_rate': stats['filter_rate']
                })
                
                logging.info(f"处理完成: {csv_file.name}")
                
            except Exception as e:
                logging.error(f"处理文件 {csv_file.name} 失败: {e}")
        
        # 生成汇总报告
        if summary:
            summary_df = pd.DataFrame(summary)
            report_file = output_dir / "filtering_summary.csv"
            summary_df.to_csv(report_file, index=False)
            logging.info(f"✓ 过滤汇总报告已保存: {report_file}")
        
        return summary


# 测试代码
if __name__ == "__main__":
    # 简单测试
    print("测试峰分类器模块...")
    print(f"FeatureEngineer类已导入: {FeatureEngineer}")
    print(f"FeatureEngineer在__main__中: {'FeatureEngineer' in sys.modules['__main__'].__dict__}")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'HEIGHT': [100, 50, 200, 30],
        'X_PPM': [7.5, 3.2, 8.1, 1.5],
        'CONFIDENCE': [0.9, 0.7, 0.95, 0.6]
    })
    
    # 测试特征工程
    fe = FeatureEngineer()
    features = fe.engineer_all_features(test_data, verbose=True)
    print(f"\n生成的特征:\n{features.head()}")