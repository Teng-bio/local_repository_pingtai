#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置设置 - 用于NMR数据处理和GCF-峰匹配分析。
增强版本，支持Deep Picker参数优化
🆕 整合配置管理功能，包括首次运行检查、配置初始化等
"""
import os
import sys
import json
import logging
import shutil
import multiprocessing as mp
from pathlib import Path
import psutil
import appdirs

# 使用appdirs库获取平台无关的用户配置目录
APP_NAME = "pingtai-nmr-processor"
APP_AUTHOR = "pingtai"
USER_CONFIG_DIR = appdirs.user_config_dir(APP_NAME, APP_AUTHOR)
USER_DATA_DIR = appdirs.user_data_dir(APP_NAME, APP_AUTHOR)
USER_CONFIG_PATH = os.path.join(USER_CONFIG_DIR, "config.json")

# 确保配置目录存在
os.makedirs(USER_CONFIG_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)


# 需要用户提供的路径列表
REQUIRED_PATHS = [
    {
        "name": "deep_picker_path",
        "description": "Deep Picker 工具路径 (deep_picker_1d)",
        "default": "",
        "required": True,
        "check_exists": True
    },
    {
        "name": "data_dir",
        "description": "NMR数据根目录",
        "default": USER_DATA_DIR,
        "required": True,
        "check_exists": False,
        "create_if_missing": True
    }
]


class Config:
    """NMR数据处理和GCF-峰匹配的配置设置"""
    
    # 检查是否在安装的包内运行
    IS_INSTALLED = getattr(sys, 'frozen', False) or not __file__.endswith('.py')
    
    # 基础路径设置 - 会被环境变量或参数覆盖
    DEFAULT_BASE_PATH = str(Path(USER_DATA_DIR) / "nmr_data")
   
    
    # 获取脚本路径 - 优先使用包内路径
    if IS_INSTALLED:
        import importlib.resources
        try:
            # Python 3.9+
            with importlib.resources.files('pingtai.nmr_processor.scripts') as script_dir:
                DEFAULT_SCRIPT_PATH = str(script_dir / "nmr_converter.tcsh")
        except AttributeError:
            # 兼容 Python 3.7-3.8
            DEFAULT_SCRIPT_PATH = str(Path(__file__).parent / "scripts" / "nmr_converter.tcsh")
    else:
        DEFAULT_SCRIPT_PATH = str(Path(__file__).parent / "scripts" / "nmr_converter.tcsh")
    #获取峰分类器模型默认路径
    DEFAULT_MODEL_PATH = str(Path(__file__).parent / "models" / "nmr_peak_classifier.pkl")
    
    # 从环境变量获取路径（如果可用）
    BASE_PATH = os.environ.get("NMR_BASE_PATH", DEFAULT_BASE_PATH)
    SCRIPT_PATH = os.environ.get("NMR_SCRIPT_PATH", DEFAULT_SCRIPT_PATH)
    
    # 用户定义的路径 - 默认为空，强制用户进行配置
    DEEP_PICKER_PATH = ""

     # === 峰分类器配置 ===
    PEAK_CLASSIFIER_MODEL_PATH = os.environ.get("NMR_PEAK_CLASSIFIER_PATH", DEFAULT_MODEL_PATH)
    USE_PEAK_CLASSIFIER = True  # 是否启用峰分类器
    PEAK_CONFIDENCE_THRESHOLD = 0.4 # 峰置信度阈值
    
    # 分析参数
    NUM_WORKERS = max(1, min(mp.cpu_count() - 1, 4))  # 默认为4或更少

    # ===== 峰聚类（DBSCAN）配置 =====
    PEAK_CLUSTERING = {
        'tolerance': 0.002,   # 聚类容差（ppm）
    }

    # ===== 新增：灵活打分配置 =====
    SCORING_CONFIG = {
        'perfect_match': 10,   # 有GCF且有峰
        'missing_peak': 0,     # 有GCF但无峰
        'orphan_peak': -10,    # 无GCF但有峰
        'baseline': 1          # 都不存在
    }

    # ===== 新增：报告生成配置 =====
    REPORT_CONFIG = {
        'generate_strain_folders': True,   # 按菌株生成子文件夹
        'generate_rank_reports': True,     # 生成rank文件
        'generate_summary': True,          # 生成summary文件
    }

    # ===== 新增：数据库存储配置（仅存储用途） =====
    DB_STORAGE = {
        'store_baseline': False  # 是否存储baseline(1分)记录; False可大幅减少存储与IO
    }

    #  内存管理配置
    MEMORY_PROTECTION = {
        'enabled': True,                    # 是否启用内存保护
        'max_memory_percent': 85,           # 最大内存使用百分比
        'warning_threshold': 85,            # 警告阈值
        'critical_threshold': 95,           # 危险阈值
        'enable_auto_gc': True,             # 启用自动GC
        'enable_leak_detection': True,      # 启用内存泄漏检测
        'emergency_cleanup_enabled': True,  # 启用紧急清理
    }
    
    # 内存降级策略
    MEMORY_DEGRADATION = {
        'batch_size_factor': 0.5,      # 降级时批处理大小系数
        'worker_factor': 0.5,          # 降级时工作进程系数
        'force_gc_frequency': 5,       # 强制GC频率（批次）
    }
    
    # Deep Picker优化相关参数
    DEEP_PICKER_OPTIMIZATION = {
        # 默认的model选择阈值
        'ppp_model2_threshold': 12.0,  # PPP < 12使用model 2 (代谢物)
        'ppp_model1_threshold': 12.0,  # PPP >= 12使用model 1 (蛋白质)
        
        # SNR阈值用于参数调整
        'snr_high_threshold': 100.0,    # 高SNR阈值
        'snr_medium_threshold': 50.0,   # 中SNR阈值
        'snr_low_threshold': 20.0,      # 低SNR阈值
        
        # 各质量等级的默认参数
        'high_quality_params': {
            'scale': 6.5,
            'scale2': 3.8,
            'auto_ppp': 'no'
        },
        'medium_quality_params': {
            'scale': 5.8,
            'scale2': 3.5,
            'auto_ppp': 'no'
        },
        'low_quality_params': {
            'scale': 5.0,
            'scale2': 3.0,
            'auto_ppp': 'no'
        },
        'very_low_quality_params': {
            'scale': 4.5,
            'scale2': 2.8,
            'auto_ppp': 'yes'
        },
        
        # PPP过低时的保守参数
        'low_ppp_params': {
            'scale': 7.0,
            'scale2': 4.0,
            'auto_ppp': 'yes'
        },
        
        # 结果质量检查阈值
        'quality_check': {
            'max_total_peaks': 150,          # 总峰数上限
            'max_aromatic_peaks': 50,        # 芳香区域峰数上限
            'max_negative_peaks': 5,         # 负化学位移峰数上限
            'min_total_peaks': 10,           # 总峰数下限
            'aromatic_ratio_threshold': 0.4  # 芳香区域峰数比例上限
        },
        
        # Kaiser窗函数参数（Deep Picker推荐）
        'kaiser_window': {
            'off': 0.5,
            'end': 0.896,
            'pow': 3.684
        },
        
        # 默认代谢物线宽用于PPP计算
        'typical_metabolite_linewidth_hz': 1.0,
        
        # zero filling参数
        'recommended_zf_factor': 2,  # Deep Picker建议的zero filling倍数
    }
    
    # 分析路径 - 动态生成
    

    @classmethod
    def get_output_path(cls):
        return os.path.join(cls.BASE_PATH, "results")
    
    @classmethod
    def get_log_path(cls):
        return os.path.join(cls.get_output_path(), "analysis.log")
    
    @classmethod
    def get_gcf_matrix_path(cls):
        return os.path.join(cls.BASE_PATH, "strain_GCF_cluster", "result", "strain_gcf_matrix.csv") 
    
    @classmethod
    def get_nmr_data_path(cls):
        return os.path.join(cls.BASE_PATH, "nmr_data")
    
    # Deep Picker参数优化相关方法
    @classmethod
    def calculate_optimal_deep_picker_params(cls, ppp, snr, noise_level=None):
        """
        基于PPP和SNR计算最佳Deep Picker参数
        
        参数:
            ppp (float): Points Per Peak值
            snr (float): 信噪比
            noise_level (float, optional): 噪声水平
            
        返回:
            dict: 包含scale, scale2, model, auto_ppp的参数字典
        """
        opt = cls.DEEP_PICKER_OPTIMIZATION
        
        # 选择模型
        if ppp < opt['ppp_model2_threshold']:
            model = 2  # 代谢物模型
        else:
            model = 1  # 蛋白质模型
        
        # 特殊情况：PPP过低
        if ppp < 6.0:
            params = opt['low_ppp_params'].copy()
            params['model'] = model
            return params
        
        # 根据SNR选择参数
        if snr >= opt['snr_high_threshold']:
            params = opt['high_quality_params'].copy()
        elif snr >= opt['snr_medium_threshold']:
            params = opt['medium_quality_params'].copy()
        elif snr >= opt['snr_low_threshold']:
            params = opt['low_quality_params'].copy()
        else:
            params = opt['very_low_quality_params'].copy()
        
        params['model'] = model
        return params
    
    @classmethod
    def get_kaiser_window_params(cls):
        """获取Deep Picker推荐的Kaiser窗函数参数"""
        return cls.DEEP_PICKER_OPTIMIZATION['kaiser_window']
    
    @classmethod
    def check_deep_picker_result_quality(cls, total_peaks, aromatic_peaks, negative_peaks):
        """
        检查Deep Picker结果质量
        
        参数:
            total_peaks (int): 总峰数
            aromatic_peaks (int): 芳香区域峰数
            negative_peaks (int): 负化学位移峰数
            
        返回:
            tuple: (quality_score, issues, suggestions)
        """
        check = cls.DEEP_PICKER_OPTIMIZATION['quality_check']
        issues = []
        suggestions = []
        quality_score = 100
        
        # 检查总峰数
        if total_peaks > check['max_total_peaks']:
            issues.append(f"总峰数过多 ({total_peaks} > {check['max_total_peaks']})")
            suggestions.append("建议提高scale参数")
            quality_score -= min(30, (total_peaks - check['max_total_peaks']) * 0.2)
        elif total_peaks < check['min_total_peaks']:
            issues.append(f"总峰数过少 ({total_peaks} < {check['min_total_peaks']})")
            suggestions.append("建议降低scale参数")
            quality_score -= (check['min_total_peaks'] - total_peaks) * 3
        
        # 检查芳香区域峰数
        if aromatic_peaks > check['max_aromatic_peaks']:
            issues.append(f"芳香区域假峰过多 ({aromatic_peaks} > {check['max_aromatic_peaks']})")
            suggestions.append("建议使用更严格的参数")
            quality_score -= (aromatic_peaks - check['max_aromatic_peaks']) * 0.5
        
        # 检查芳香区域峰数比例
        if total_peaks > 0:
            aromatic_ratio = aromatic_peaks / total_peaks
            if aromatic_ratio > check['aromatic_ratio_threshold']:
                issues.append(f"芳香区域峰数比例过高 ({aromatic_ratio:.2%} > {check['aromatic_ratio_threshold']:.0%})")
                suggestions.append("可能存在大量假峰，建议检查谱图预处理")
                quality_score -= (aromatic_ratio - check['aromatic_ratio_threshold']) * 50
        
        # 检查负化学位移峰数
        if negative_peaks > check['max_negative_peaks']:
            issues.append(f"负化学位移峰过多 ({negative_peaks} > {check['max_negative_peaks']})")
            suggestions.append("建议检查基线校正和相位校正")
            quality_score -= negative_peaks * 2
        
        quality_score = max(0, min(100, quality_score))
        
        return quality_score, issues, suggestions
    
    @classmethod
    def get_strict_params(cls, base_scale, base_scale2):
        """
        获取更严格的参数（用于结果优化）
        
        参数:
            base_scale (float): 基础scale参数
            base_scale2 (float): 基础scale2参数
            
        返回:
            dict: 严格参数字典
        """
        return {
            'scale': base_scale + 2.0,
            'scale2': base_scale2 + 1.0,
            'auto_ppp': 'no'  # 严格模式不使用auto_ppp
        }
    
    # 动态属性访问器 - 使属性访问更具动态性
    @classmethod
    def get(cls, attr_name, default=None):
        """获取配置属性值，支持动态计算的属性"""
        getter_method = getattr(cls, f"get_{attr_name}", None)
        if getter_method and callable(getter_method):
            return getter_method()
        return getattr(cls, attr_name.upper(), default)
    
    # 其他参数保持不变
    SUB_REGIONS = [  # 子区间配置 (起始ppm, 结束ppm, 步长)
        (0.0, 5.0, 0.001),
        (5.0, 10.0, 0.001),
        (10.0, 15.0, 0.001)
    ]
    
    # GCF-峰匹配参数 
    BATCH_SIZE = max(1, min(20, mp.cpu_count() // 2))  # 根据CPU动态调整
    
    
    # 内存优化参数
    MAX_DUPLICATE_SET_SIZE = 10_000_000  # 查重集合最大大小
    INTERMEDIATE_SAVE_FREQUENCY = 50     # 每处理多少批次保存一次中间结果
    
    
    # 产生属性的动态访问器，避免直接访问OUTPUT_PATH等值
    @property
    def OUTPUT_PATH(self):
        return self.get_output_path()
    
    @property
    def LOG_PATH(self):
        return self.get_log_path()
    
    @property
    def GCF_MATRIX_PATH(self):
        return self.get_gcf_matrix_path()
    
    @property
    def NMR_DATA_PATH(self):
        return self.get_nmr_data_path()
    
    # ==========================================
    # 🆕 整合配置管理功能
    # ==========================================
    
    @classmethod
    def first_run_check(cls):
        """检查是否是首次运行，确保配置已创建"""
        if os.path.exists(USER_CONFIG_PATH):
            return False
        
        print("=" * 60)
        print("  首次运行 NMR 处理软件")
        print("  需要进行初始化设置")
        print("=" * 60)
        print(f"配置将保存在: {USER_CONFIG_PATH}")
        print(f"数据将默认存储在: {USER_DATA_DIR}")
        print("=" * 60)
        
        cls.create_user_config()
        return True
    
    @classmethod
    def get_user_config(cls):
        """获取用户配置，如果不存在则创建"""
        if os.path.exists(USER_CONFIG_PATH):
            try:
                with open(USER_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    
                # 检查配置是否有所有必要项
                missing_items = [item for item in REQUIRED_PATHS 
                                if item["required"] and item["name"] not in config]
                
                if missing_items:
                    print("配置文件缺少必要项，需要更新配置:")
                    for item in missing_items:
                        print(f" - {item['description']}")
                    return cls._update_config_with_items(config, missing_items)
                    
                return config
            except Exception as e:
                logging.warning(f"读取配置文件失败: {e}")
        
        # 配置文件不存在，创建新配置
        return cls.create_user_config()
    
    @classmethod
    def create_user_config(cls):
        """引导用户创建配置文件"""
        print("=" * 60)
        print("  欢迎使用NMR数据处理软件！")
        print("  首次运行需要设置一些基本配置项")
        print("=" * 60)
        
        config = {}
        
        for item in REQUIRED_PATHS:
            default_value = item["default"]
            
            while True:
                prompt = f"{item['description']}"
                if default_value:
                    prompt += f" [{default_value}]: "
                else:
                    prompt += ": "
                    
                value = input(prompt) or default_value
                
                # 检查路径是否存在
                if item.get("check_exists", False) and value:
                    path = Path(value)
                    if not path.exists():
                        print(f"警告: 路径不存在 '{value}'")
                        
                        if item.get("create_if_missing", False):
                            try:
                                os.makedirs(value, exist_ok=True)
                                print(f"已创建目录: {value}")
                            except Exception as e:
                                print(f"创建目录失败: {e}")
                                if item.get("required", False):
                                    continue
                        elif item.get("required", False):
                            print("这是必需路径，请提供正确路径。")
                            continue
                        else:
                            print("将使用这个不存在的路径，请确保后续创建正确。")
                
                config[item["name"]] = value
                break
        
        # 保存配置
        try:
            os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
            with open(USER_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"配置已保存到: {USER_CONFIG_PATH}")
            
            # 复制默认脚本到用户目录
            try:
                script_path = cls.SCRIPT_PATH
                user_script_path = os.path.join(USER_DATA_DIR, "scripts", "nmr_converter.tcsh")
                
                os.makedirs(os.path.dirname(user_script_path), exist_ok=True)
                
                if os.path.exists(script_path):
                    shutil.copy2(script_path, user_script_path)
                    print(f"已复制默认脚本到: {user_script_path}")
            except Exception as e:
                print(f"复制脚本时出错: {e}")
        except Exception as e:
            print(f"保存配置失败: {e}")
            print("将使用临时配置继续执行，但不会保存供后续使用。")
        
        return config
    
    @classmethod
    def _update_config_with_items(cls, config, items):
        """更新配置文件中指定的项目"""
        print("=" * 60)
        print("  更新配置项")
        print("=" * 60)
        
        for item in items:
            default_value = item["default"]
            
            prompt = f"{item['description']}"
            if default_value:
                prompt += f" [{default_value}]: "
            else:
                prompt += ": "
                
            value = input(prompt) or default_value
            
            # 检查路径是否存在
            if item.get("check_exists", False) and value:
                path = Path(value)
                if not path.exists():
                    print(f"警告: 路径不存在 '{value}'")
                    
                    if item.get("create_if_missing", False):
                        try:
                            os.makedirs(value, exist_ok=True)
                            print(f"已创建目录: {value}")
                        except Exception as e:
                            print(f"创建目录失败: {e}")
            
            config[item["name"]] = value
        
        # 保存更新后的配置
        try:
            with open(USER_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"配置已更新并保存到: {USER_CONFIG_PATH}")
        except Exception as e:
            print(f"保存配置失败: {e}")
        
        return config
    
    @classmethod
    def update_config_entry(cls):
        """更新配置文件中的特定条目"""
        if not os.path.exists(USER_CONFIG_PATH):
            print("配置文件不存在，将重新创建完整配置。")
            return cls.create_user_config()
        
        try:
            with open(USER_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception:
            print("读取现有配置失败，将重新创建配置。")
            return cls.create_user_config()
        
        print("当前配置项:")
        for i, (key, value) in enumerate(config.items()):
            print(f"{i+1}. {key}: {value}")
        
        # 添加创建新项的选项
        print(f"{len(config)+1}. 添加新配置项")
        print("0. 退出")
        
        try:
            choice = int(input("\n请选择要更新的项目编号 (0 为退出): "))
            if choice == 0:
                return config
            
            if choice == len(config) + 1:
                # 添加新配置项
                key = input("请输入新配置项的名称: ")
                value = input(f"请输入 {key} 的值: ")
                config[key] = value
            else:
                keys = list(config.keys())
                if 1 <= choice <= len(keys):
                    key = keys[choice-1]
                    value = input(f"请输入 {key} 的新值 [{config[key]}]: ") or config[key]
                    config[key] = value
                else:
                    print("无效的选择")
                    return config
            
            with open(USER_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"配置已更新并保存到: {USER_CONFIG_PATH}")
        except (ValueError, IndexError):
            print("输入错误，配置未更改")
        
        return config
    
    @classmethod
    def load_user_config(cls):
        """加载用户配置文件"""
        if os.path.exists(USER_CONFIG_PATH):
            try:
                with open(USER_CONFIG_PATH, 'r') as f:
                    user_config = json.load(f)
                    
                # 更新配置
                if 'deep_picker_path' in user_config:
                    cls.DEEP_PICKER_PATH = user_config['deep_picker_path']
                if 'data_dir' in user_config and user_config['data_dir']:
                    cls.BASE_PATH = os.path.dirname(user_config['data_dir'])
                
                # 加载Deep Picker优化参数（如果存在）
                if 'deep_picker_optimization' in user_config:
                    dp_config = user_config['deep_picker_optimization']
                    for key, value in dp_config.items():
                        if key in cls.DEEP_PICKER_OPTIMIZATION:
                            if isinstance(cls.DEEP_PICKER_OPTIMIZATION[key], dict):
                                cls.DEEP_PICKER_OPTIMIZATION[key].update(value)
                            else:
                                cls.DEEP_PICKER_OPTIMIZATION[key] = value
                
                # 加载峰聚类配置（如果存在）
                if 'peak_clustering' in user_config:
                    pc_config = user_config['peak_clustering']
                    for key, value in pc_config.items():
                        if key in cls.PEAK_CLUSTERING:
                            cls.PEAK_CLUSTERING[key] = value
                
                # 加载打分配置（如果存在）
                if 'scoring_config' in user_config:
                    sc_config = user_config['scoring_config']
                    for key, value in sc_config.items():
                        if key in cls.SCORING_CONFIG:
                            cls.SCORING_CONFIG[key] = value
                
                # 添加其他可能的配置项
                for key, value in user_config.items():
                    if hasattr(cls, key.upper()):
                        setattr(cls, key.upper(), value)
                    
                logging.info(f"已加载用户配置: {USER_CONFIG_PATH}")
                return True
            except Exception as e:
                logging.warning(f"加载用户配置失败: {e}")
        
        # 如果配置不存在，返回假表示需要引导用户创建配置
        return False
    
    @classmethod
    def save_user_config(cls):
        """保存用户配置到文件"""
        try:
            os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
            
            config_data = {
                'deep_picker_path': cls.DEEP_PICKER_PATH,
                'data_dir': cls.get_nmr_data_path(),
                'base_path': cls.BASE_PATH,
                'script_path': cls.SCRIPT_PATH,
                'deep_picker_optimization': cls.DEEP_PICKER_OPTIMIZATION,
                'peak_clustering': cls.PEAK_CLUSTERING,
                'scoring_config': cls.SCORING_CONFIG,
            }
            
            with open(USER_CONFIG_PATH, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logging.info(f"用户配置已保存: {USER_CONFIG_PATH}")
            return True
        except Exception as e:
            logging.error(f"保存用户配置失败: {e}")
            return False
    
    @classmethod
    def create_directories(cls):
        """如果目录不存在，则创建必要的目录"""
        os.makedirs(cls.get_output_path(), exist_ok=True)
        os.makedirs(os.path.dirname(USER_CONFIG_PATH), exist_ok=True)
        
        # 确保脚本目录存在
        scripts_dir = os.path.dirname(cls.SCRIPT_PATH)
        os.makedirs(scripts_dir, exist_ok=True)
    
    @classmethod
    def get_memory_status(cls):
        """获取当前内存状态"""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1024**3,
            "available": mem.available / 1024**3,
            "percent": mem.percent
        }
    
    @classmethod
    def adjust_batch_size(cls):
        """根据系统内存动态调整批处理大小"""
        mem_status = cls.get_memory_status()
        if mem_status["percent"] > 80:
            # 内存紧张，减小批处理大小
            cls.BATCH_SIZE = max(1, cls.BATCH_SIZE // 2)
            return True
        elif mem_status["percent"] < 50 and cls.BATCH_SIZE < 20:
            # 内存充足，增加批处理大小
            cls.BATCH_SIZE = min(20, cls.BATCH_SIZE + 2)
            return True
        return False
    
    @classmethod
    def get_peak_classifier_model_path(cls):
        """获取峰分类器模型路径"""
        return cls.PEAK_CLASSIFIER_MODEL_PATH


# 🆕 导出配置路径常量，方便其他模块使用
__all__ = ['Config', 'USER_CONFIG_PATH', 'USER_CONFIG_DIR', 'USER_DATA_DIR']