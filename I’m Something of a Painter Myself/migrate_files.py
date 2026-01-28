#!/usr/bin/env python3
"""
脚本：将文件夹X中的所有文件迁移到目标文件夹。
支持复制或移动，递归处理子文件夹中的文件。
"""

import os
import shutil
import argparse

def migrate_files(source_dir, dest_dir, move=False, recursive=True):
    """
    迁移文件从source_dir到dest_dir。
    
    :param source_dir: 源文件夹路径
    :param dest_dir: 目标文件夹路径
    :param move: 如果True，则移动文件；否则复制
    :param recursive: 如果True，递归处理子文件夹
    """
    if not os.path.exists(source_dir):
        print(f"源文件夹不存在: {source_dir}")
        return
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"创建目标文件夹: {dest_dir}")
    
    migrated_count = 0
    
    if recursive:
        # 递归遍历所有文件
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                src_path = os.path.join(root, file)
                # 保持相对路径结构
                rel_path = os.path.relpath(root, source_dir)
                dest_subdir = os.path.join(dest_dir, rel_path) if rel_path != '.' else dest_dir
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir)
                dest_path = os.path.join(dest_subdir, file)
                
                try:
                    if move:
                        shutil.move(src_path, dest_path)
                    else:
                        shutil.copy2(src_path, dest_path)  # copy2保留元数据
                    migrated_count += 1
                except Exception as e:
                    print(f"迁移失败 {src_path}: {e}")
    else:
        # 只处理源文件夹根目录的文件
        for item in os.listdir(source_dir):
            src_path = os.path.join(source_dir, item)
            if os.path.isfile(src_path):
                dest_path = os.path.join(dest_dir, item)
                try:
                    if move:
                        shutil.move(src_path, dest_path)
                    else:
                        shutil.copy2(src_path, dest_path)
                    migrated_count += 1
                except Exception as e:
                    print(f"迁移失败 {src_path}: {e}")
    
    action = "移动" if move else "复制"
    print(f"完成！{action}了 {migrated_count} 个文件到 {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件夹中的文件迁移到另一个文件夹")
    parser.add_argument("source", help="源文件夹路径")
    parser.add_argument("dest", help="目标文件夹路径")
    parser.add_argument("--move", action="store_true", help="移动文件而不是复制")
    parser.add_argument("--no-recursive", action="store_true", help="不递归处理子文件夹，只迁移根目录文件")
    
    args = parser.parse_args()
    
    recursive = not args.no_recursive
    migrate_files(args.source, args.dest, move=args.move, recursive=recursive)