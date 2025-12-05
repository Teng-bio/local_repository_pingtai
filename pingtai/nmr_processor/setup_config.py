#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置初始化模块 - 处理用户配置，解决路径问题
"""

import os
import sys
import json
import logging
from pathlib import Path
import appdirs
import shutil


# 从配置类导入配置路径
from nmr_processor.config import USER_CONFIG_PATH, USER_CONFIG_DIR, USER_DATA_DIR

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

def first_run_check():
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
    
    create_user_config()
    return True

def get_user_config():
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
                return update_config_with_items(config, missing_items)
                
            return config
        except Exception as e:
            logging.warning(f"读取配置文件失败: {e}")
    
    # 配置文件不存在，创建新配置
    return create_user_config()

def create_user_config():
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
            
            # 检查命令是否可执行
            if "check_command" in item and value:
                cmd = item["check_command"]
                try:
                    import subprocess
                    subprocess.run(cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    print(f"警告: 命令检查失败 '{cmd}'")
                    if item.get("required", False):
                        print("相关命令可能无法执行，请检查环境变量或安装。")
                        continue
            
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
            from .config import Config
            script_path = Config.SCRIPT_PATH
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

def update_config_with_items(config, items):
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

def update_config_entry():
    """更新配置文件中的特定条目"""
    if not os.path.exists(USER_CONFIG_PATH):
        print("配置文件不存在，将重新创建完整配置。")
        return create_user_config()
    
    try:
        with open(USER_CONFIG_PATH, 'r') as f:
            config = json.load(f)
    except Exception:
        print("读取现有配置失败，将重新创建配置。")
        return create_user_config()
    
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

def main():
    """配置工具主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NMR处理软件配置工具")
    parser.add_argument("--update", action="store_true", help="更新现有配置")
    parser.add_argument("--init", action="store_true", help="初始化新配置")
    parser.add_argument("--show", action="store_true", help="显示当前配置")
    
    args = parser.parse_args()
    
    if args.show:
        if os.path.exists(USER_CONFIG_PATH):
            try:
                with open(USER_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                print("当前配置:")
                for key, value in config.items():
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"读取配置失败: {e}")
        else:
            print("配置文件不存在")
    elif args.init:
        create_user_config()
    elif args.update:
        update_config_entry()
    else:
        # 如果没有指定参数，检查是否是首次运行
        if not os.path.exists(USER_CONFIG_PATH):
            first_run_check()
        else:
            update_config_entry()

if __name__ == "__main__":
    main()