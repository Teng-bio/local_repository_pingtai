#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存保护模块 - 防止内存溢出和崩溃
提供多层次的内存保护机制
"""

import os
import sys
import gc
import logging
import psutil
import signal
import traceback
from functools import wraps
from contextlib import contextmanager
import time
import warnings


class MemoryProtector:
    """
    内存保护器 - 提供多层次内存保护
    
    功能：
    1. 实时监控内存使用
    2. 自动降级处理
    3. 内存泄漏检测
    4. OOM预警和阻止
    5. 优雅降级和恢复
    """
    
    # 内存阈值配置
    CRITICAL_THRESHOLD = 95   # 危险：立即停止
    WARNING_THRESHOLD = 85    # 警告：降级处理
    SAFE_THRESHOLD = 70       # 安全：正常运行
    
    # 检查频率
    CHECK_INTERVAL = 5        # 秒
    
    def __init__(self, max_memory_percent=85, enable_auto_gc=True):
        """
        初始化内存保护器
        
        参数:
            max_memory_percent: 最大内存使用百分比
            enable_auto_gc: 是否启用自动GC
        """
        self.max_memory_percent = max_memory_percent
        self.enable_auto_gc = enable_auto_gc
        
        # 获取系统信息
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.process = psutil.Process(os.getpid())
        
        # 监控状态
        self.last_check_time = 0
        self.warning_count = 0
        self.peak_memory = 0
        
        # 降级状态
        self.degraded_mode = False
        self.original_batch_size = None
        
        logging.info(f"内存保护器已初始化：")
        logging.info(f"  - 总内存: {self.total_memory:.2f} GB")
        logging.info(f"  - 最大使用率: {self.max_memory_percent}%")
        logging.info(f"  - 自动GC: {'启用' if self.enable_auto_gc else '禁用'}")
    
    def get_memory_info(self):
        """
        获取详细的内存信息
        
        返回:
            dict: 内存信息
        """
        # 系统内存
        sys_mem = psutil.virtual_memory()
        
        # 进程内存
        proc_mem = self.process.memory_info()
        
        info = {
            # 系统内存
            'system_total_gb': sys_mem.total / (1024**3),
            'system_available_gb': sys_mem.available / (1024**3),
            'system_used_percent': sys_mem.percent,
            
            # 进程内存
            'process_rss_gb': proc_mem.rss / (1024**3),  # 实际物理内存
            'process_vms_gb': proc_mem.vms / (1024**3),  # 虚拟内存
            
            # 额外信息
            'peak_memory_gb': self.peak_memory,
            'degraded_mode': self.degraded_mode
        }
        
        # 更新峰值
        if info['process_rss_gb'] > self.peak_memory:
            self.peak_memory = info['process_rss_gb']
        
        return info
    
    def check_memory(self, force=False):
        """
        检查内存使用情况
        
        参数:
            force: 是否强制检查（忽略时间间隔）
        
        返回:
            tuple: (status, action, info)
                status: 'safe', 'warning', 'critical'
                action: 建议的操作
                info: 内存信息
        """
        # 检查频率限制
        current_time = time.time()
        if not force and (current_time - self.last_check_time) < self.CHECK_INTERVAL:
            return 'safe', None, None
        
        self.last_check_time = current_time
        
        # 获取内存信息
        info = self.get_memory_info()
        mem_percent = info['system_used_percent']
        
        # 判断状态
        if mem_percent >= self.CRITICAL_THRESHOLD:
            status = 'critical'
            action = 'stop'
            self.warning_count += 1
            
            logging.critical(
                f"🚨 内存危险！使用率: {mem_percent:.1f}% "
                f"(进程: {info['process_rss_gb']:.2f} GB)"
            )
            
        elif mem_percent >= self.WARNING_THRESHOLD:
            status = 'warning'
            action = 'degrade'
            self.warning_count += 1
            
            logging.warning(
                f"⚠️ 内存警告！使用率: {mem_percent:.1f}% "
                f"(进程: {info['process_rss_gb']:.2f} GB)"
            )
            
        else:
            status = 'safe'
            action = None
            self.warning_count = 0
            
            if mem_percent > self.SAFE_THRESHOLD:
                logging.info(
                    f"ℹ️ 内存使用: {mem_percent:.1f}% "
                    f"(进程: {info['process_rss_gb']:.2f} GB)"
                )
        
        return status, action, info
    
    def force_gc(self):
        """强制垃圾回收"""
        before = self.get_memory_info()['process_rss_gb']
        
        # 多次GC以清理循环引用
        gc.collect()
        gc.collect()
        gc.collect()
        
        after = self.get_memory_info()['process_rss_gb']
        freed = before - after
        
        if freed > 0.1:  # 释放超过100MB
            logging.info(f"♻️ GC释放内存: {freed:.2f} GB")
        
        return freed
    
    def enter_degraded_mode(self, config=None):
        """
        进入降级模式
        
        降级措施：
        1. 减小批处理大小
        2. 减少工作进程
        3. 强制GC
        4. 清理缓存
        """
        if self.degraded_mode:
            return  # 已经在降级模式
        
        logging.warning("⬇️ 进入内存降级模式...")
        
        # 强制GC
        self.force_gc()
        
        # 降级配置
        if config is not None:
            # 保存原始配置
            self.original_batch_size = getattr(config, 'BATCH_SIZE', None)
            original_workers = getattr(config, 'NUM_WORKERS', None)
            
            # 降级
            if hasattr(config, 'BATCH_SIZE'):
                config.BATCH_SIZE = max(1, config.BATCH_SIZE // 2)
                logging.warning(f"  - 批处理大小: {self.original_batch_size} → {config.BATCH_SIZE}")
            
            if hasattr(config, 'NUM_WORKERS'):
                config.NUM_WORKERS = max(1, config.NUM_WORKERS // 2)
                logging.warning(f"  - 工作进程数: {original_workers} → {config.NUM_WORKERS}")
        
        self.degraded_mode = True
    
    def exit_degraded_mode(self, config=None):
        """退出降级模式"""
        if not self.degraded_mode:
            return
        
        logging.info("⬆️ 退出内存降级模式")
        
        # 恢复配置
        if config is not None and self.original_batch_size is not None:
            config.BATCH_SIZE = self.original_batch_size
            logging.info(f"  - 批处理大小已恢复: {config.BATCH_SIZE}")
        
        self.degraded_mode = False
        self.original_batch_size = None
    
    def handle_oom(self):
        """
        处理内存溢出情况
        
        返回:
            bool: True=可以继续, False=必须停止
        """
        logging.error("💥 检测到内存溢出风险！")
        
        # 尝试恢复
        logging.info("尝试释放内存...")
        
        # 1. 强制GC
        freed = self.force_gc()
        
        # 2. 再次检查
        status, _, info = self.check_memory(force=True)
        
        if status == 'critical':
            logging.error(f"❌ 无法恢复，内存仍然危险: {info['system_used_percent']:.1f}%")
            return False
        else:
            logging.info(f"✓ 内存已恢复到安全水平: {info['system_used_percent']:.1f}%")
            return True
    
    def estimate_memory_needed(self, data_size, factor=2.0):
        """
        估算处理数据所需的内存
        
        参数:
            data_size: 数据大小（字节）
            factor: 内存放大系数（默认2倍）
        
        返回:
            tuple: (estimated_gb, is_safe)
        """
        estimated_gb = (data_size * factor) / (1024**3)
        
        info = self.get_memory_info()
        available_gb = info['system_available_gb']
        
        is_safe = estimated_gb < (available_gb * 0.8)
        
        if not is_safe:
            logging.warning(
                f"⚠️ 内存可能不足：需要 ~{estimated_gb:.2f} GB, "
                f"可用 {available_gb:.2f} GB"
            )
        
        return estimated_gb, is_safe
    
    def monitor_loop(self, callback=None, interval=None):
        """
        内存监控循环（用于后台监控）
        
        参数:
            callback: 当状态变化时调用的函数
            interval: 检查间隔（秒）
        """
        if interval is None:
            interval = self.CHECK_INTERVAL
        
        try:
            while True:
                status, action, info = self.check_memory(force=True)
                
                if callback is not None:
                    callback(status, action, info)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logging.info("内存监控已停止")


# ========== 装饰器 ==========

def with_memory_protection(max_memory_percent=85):
    """
    内存保护装饰器
    
    用法:
        @with_memory_protection(max_memory_percent=85)
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            protector = MemoryProtector(max_memory_percent=max_memory_percent)
            
            # 执行前检查
            status, action, info = protector.check_memory(force=True)
            
            if status == 'critical':
                raise MemoryError(
                    f"内存使用过高 ({info['system_used_percent']:.1f}%), "
                    f"无法执行函数 {func.__name__}"
                )
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 执行后检查
                status, action, info = protector.check_memory(force=True)
                
                if status == 'warning':
                    logging.warning(
                        f"函数 {func.__name__} 执行后内存警告: "
                        f"{info['system_used_percent']:.1f}%"
                    )
                    protector.force_gc()
                
                return result
                
            except MemoryError as e:
                logging.error(f"内存溢出: {e}")
                protector.handle_oom()
                raise
            
        return wrapper
    return decorator


@contextmanager
def memory_limit(max_memory_gb=None, max_percent=None):
    """
    内存限制上下文管理器
    
    用法:
        with memory_limit(max_memory_gb=8.0):
            # 代码块
            pass
    """
    protector = MemoryProtector()
    
    if max_memory_gb is not None:
        max_percent = (max_memory_gb / protector.total_memory) * 100
    elif max_percent is None:
        max_percent = 85
    
    protector.max_memory_percent = max_percent
    
    # 进入前检查
    status, _, info = protector.check_memory(force=True)
    
    if status == 'critical':
        raise MemoryError(
            f"当前内存使用已超过限制: {info['system_used_percent']:.1f}%"
        )
    
    try:
        yield protector
        
    finally:
        # 退出后清理
        protector.force_gc()
        
        status, _, info = protector.check_memory(force=True)
        if status != 'safe':
            logging.warning(
                f"退出时内存仍然较高: {info['system_used_percent']:.1f}%"
            )


class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self, threshold_mb=100, window_size=10):
        """
        初始化检测器
        
        参数:
            threshold_mb: 泄漏阈值（MB）
            window_size: 检测窗口大小
        """
        self.threshold_mb = threshold_mb
        self.window_size = window_size
        self.memory_history = []
        self.process = psutil.Process(os.getpid())
    
    def check(self):
        """
        检查是否有内存泄漏
        
        返回:
            tuple: (is_leaking, leak_rate_mb_per_check)
        """
        current_mb = self.process.memory_info().rss / (1024**2)
        self.memory_history.append(current_mb)
        
        # 保持窗口大小
        if len(self.memory_history) > self.window_size:
            self.memory_history.pop(0)
        
        # 需要足够的数据点
        if len(self.memory_history) < self.window_size:
            return False, 0.0
        
        # 计算增长率
        first = self.memory_history[0]
        last = self.memory_history[-1]
        growth = last - first
        rate = growth / len(self.memory_history)
        
        # 判断是否泄漏
        is_leaking = growth > self.threshold_mb and rate > 5.0  # 每次检查增长>5MB
        
        if is_leaking:
            logging.warning(
                f"🔍 检测到潜在内存泄漏！"
                f"窗口增长: {growth:.1f} MB, "
                f"增长率: {rate:.1f} MB/检查"
            )
        
        return is_leaking, rate


# ========== 实用函数 ==========

def get_safe_batch_size(total_items, memory_per_item_mb, max_memory_percent=80):
    """
    计算安全的批处理大小
    
    参数:
        total_items: 总项目数
        memory_per_item_mb: 每个项目预估内存（MB）
        max_memory_percent: 最大内存使用百分比
    
    返回:
        int: 安全的批处理大小
    """
    available_mb = (psutil.virtual_memory().available / (1024**2)) * (max_memory_percent / 100)
    batch_size = int(available_mb / memory_per_item_mb)
    
    # 限制范围
    batch_size = max(1, min(batch_size, total_items))
    
    logging.info(
        f"计算安全批处理大小: {batch_size} "
        f"(可用内存: {available_mb:.0f} MB, "
        f"每项: {memory_per_item_mb:.1f} MB)"
    )
    
    return batch_size


def emergency_cleanup():
    """
    紧急内存清理
    
    用于内存危险时的最后手段
    """
    logging.warning("⚠️ 执行紧急内存清理...")
    
    before = psutil.virtual_memory().percent
    
    # 1. 强制GC
    gc.collect()
    gc.collect()
    gc.collect()
    
    # 2. 清理未引用对象
    gc.collect(generation=2)
    
    # 3. 尝试清理模块缓存（谨慎使用）
    import sys
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('_'):
            continue
        module = sys.modules.get(module_name)
        if hasattr(module, '__dict__'):
            for attr in list(module.__dict__.keys()):
                if attr.startswith('_cache'):
                    try:
                        delattr(module, attr)
                    except:
                        pass
    
    after = psutil.virtual_memory().percent
    freed = before - after
    
    logging.info(f"紧急清理完成，释放: {freed:.1f}%")
    
    return freed


# ========== 测试函数 ==========

def test_memory_protection():
    """测试内存保护功能"""
    print("=" * 80)
    print("内存保护模块测试")
    print("=" * 80)
    
    # 测试1：基本监控
    print("\n测试1：基本内存监控")
    protector = MemoryProtector(max_memory_percent=85)
    
    status, action, info = protector.check_memory(force=True)
    print(f"状态: {status}")
    print(f"系统内存: {info['system_used_percent']:.1f}%")
    print(f"进程内存: {info['process_rss_gb']:.2f} GB")
    
    # 测试2：内存限制上下文
    print("\n测试2：内存限制上下文")
    try:
        with memory_limit(max_percent=90) as p:
            print(f"进入内存保护区域，限制: {p.max_memory_percent}%")
            # 模拟一些操作
            data = [0] * 1000000
            print("操作完成")
    except MemoryError as e:
        print(f"内存限制触发: {e}")
    
    # 测试3：内存泄漏检测
    print("\n测试3：内存泄漏检测")
    detector = MemoryLeakDetector(threshold_mb=50, window_size=5)
    
    for i in range(10):
        is_leaking, rate = detector.check()
        print(f"检查 {i+1}: 泄漏={is_leaking}, 增长率={rate:.2f} MB/检查")
        time.sleep(0.1)
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_memory_protection()