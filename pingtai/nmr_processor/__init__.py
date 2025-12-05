#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NMR 处理器包 - 提供核磁共振数据处理和分析功能
"""

from .processor import run_nmr_processing
from .converter import find_and_convert_tab_files, organize_nmr_samples
from .analysis.base import analyze_nmr_data
from .analysis.gcf_peak_matcher import match_gcf_peaks
from .setup_config import update_config_entry, get_user_config, first_run_check

__version__ = '0.1.0'
__all__ = [
    'run_nmr_processing', 
    'find_and_convert_tab_files',
    'organize_nmr_samples',
    'analyze_nmr_data',
    'match_gcf_peaks',
    'update_config_entry',
    'get_user_config',
    'first_run_check'
]