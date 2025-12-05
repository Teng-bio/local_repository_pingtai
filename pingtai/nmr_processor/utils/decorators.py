#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
装饰器模块 - 提供通用的功能装饰器
"""

import os
import time
import logging
from functools import wraps

def memory_monitor(func):
    """内存监控装饰器，记录函数执行前后的内存使用情况"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        proc = psutil.Process(os.getpid())
        start_mem = proc.memory_info().rss / 1024**3
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        time_cost = time.time() - start_time
        end_mem = proc.memory_info().rss / 1024**3
        logging.info(
            f"{func.__name__} | 耗时: {time_cost:.1f}s | "
            f"内存变化: {end_mem - start_mem:+.2f}GB"
        )
        return result
    return wrapper