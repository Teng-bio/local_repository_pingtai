#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
参考峰自动检测脚本 - 用于集成到NMR处理流程
功能：检测ft1文件中最接近0 ppm的参考峰，输出REF校正值
用法：python ref_peak_detector.py <spectrum.ft1>
输出：REF参数值（浮点数，精确到6位小数）
"""

import sys
import numpy as np

try:
    import nmrglue as ng
    from scipy.signal import find_peaks
except ImportError:
    print("ERROR: Required packages not installed (nmrglue, scipy)", file=sys.stderr)
    sys.exit(1)


def find_reference_peak(ft1_path, search_range=(-0.5, 1.5)):
    """
    在指定化学位移范围内检测最接近0 ppm的参考峰
    
    参数:
        ft1_path (str): spectrum.ft1文件路径
        search_range (tuple): 搜索范围 (起始ppm, 结束ppm)
    
    返回:
        float: REF校正值（将参考峰移到0 ppm所需的偏移量）
        None: 检测失败
    """
    try:
        # 读取ft1文件
        dic, data = ng.pipe.read(ft1_path)
        uc = ng.pipe.make_uc(dic, data)
        ppm = uc.ppm_scale()
        real_data = np.real(data)
        
        # 限制搜索区域
        mask = (ppm >= search_range[0]) & (ppm <= search_range[1])
        region_ppm = ppm[mask]
        region_data = real_data[mask]
        
        if len(region_data) == 0:
            return None
        
        # 峰检测：寻找所有显著峰
        max_intensity = np.max(region_data)
        peaks, properties = find_peaks(
            region_data,
            height=max_intensity * 0.05,      # 高度>最大值的5%
            prominence=max_intensity * 0.02,   # 显著性>最大值的2%
            distance=10                        # 最小峰间距
        )
        
        if len(peaks) == 0:
            # 如果没有检测到峰，使用最大值点作为参考
            max_idx = np.argmax(region_data)
            ref_ppm = region_ppm[max_idx]
        else:
            # 从所有检测到的峰中选择最接近0 ppm的
            peak_ppms = [region_ppm[idx] for idx in peaks]
            peak_distances = [abs(p) for p in peak_ppms]
            min_distance_idx = np.argmin(peak_distances)
            ref_ppm = peak_ppms[min_distance_idx]
        
        # 返回REF校正值（负号：将ref_ppm移到0需要加上-ref_ppm）
        ref_correction = -ref_ppm
        
        return ref_correction
    
    except Exception as e:
        print(f"ERROR: Failed to process file - {e}", file=sys.stderr)
        return None


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python ref_peak_detector.py <spectrum.ft1>", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python ref_peak_detector.py spectrum.ft1", file=sys.stderr)
        sys.exit(1)
    
    ft1_file = sys.argv[1]
    
    # 检测参考峰
    ref_value = find_reference_peak(ft1_file)
    
    if ref_value is None:
        sys.exit(1)
    
    # 输出到stdout（用于tcsh捕获）
    print(f"{ref_value:.6f}")
    
    # 同时写入临时文件（备用方案，提高可靠性）
    try:
        with open("/tmp/nmr_ref_value.txt", "w") as f:
            f.write(f"{ref_value:.6f}\n")
    except:
        pass  # 忽略写入失败


if __name__ == "__main__":
    main()