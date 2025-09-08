#!/usr/bin/env python3
"""
Script to delete files with specific naming pattern within an index range.
Uses tyro for command line interface.
"""

import os
import tyro
from pathlib import Path
from typing import Optional


def delete_files_in_range(
    directory: str,
    start_index: int,
    end_index: int,
    dry_run: bool = False
) -> None:
    """
    Delete files in the format {06d}_keypoints.json and {06d}_mano.json
    within the specified index range.
    
    Args:
        directory: Path to the directory containing the files
        start_index: Starting index (inclusive)
        end_index: Ending index (inclusive)
        dry_run: If True, only print what would be deleted without actually deleting
    """
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    if not directory_path.is_dir():
        print(f"Error: {directory} is not a directory")
        return
    
    if start_index > end_index:
        print(f"Error: Start index ({start_index}) cannot be greater than end index ({end_index})")
        return
    
    deleted_count = 0
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Deleting files from index {start_index:06d} to {end_index:06d}")
    print(f"Directory: {directory_path.absolute()}")
    
    for index in range(start_index, end_index + 1):
        # Format the index with 6 digits and zero padding
        index_str = f"{index:06d}"
        
        # Check for both file types
        keypoints_file = directory_path / f"{index_str}_keypoints.json"
        mano_file = directory_path / f"{index_str}_mano.json"
        
        # Delete keypoints file if it exists
        if keypoints_file.exists():
            if dry_run:
                print(f"Would delete: {keypoints_file}")
            else:
                keypoints_file.unlink()
                print(f"Deleted: {keypoints_file}")
            deleted_count += 1
        
        # Delete mano file if it exists
        if mano_file.exists():
            if dry_run:
                print(f"Would delete: {mano_file}")
            else:
                mano_file.unlink()
                print(f"Deleted: {mano_file}")
            deleted_count += 1
    
    if dry_run:
        print(f"\nDry run complete. Would delete {deleted_count} files.")
    else:
        print(f"\nDeletion complete. Deleted {deleted_count} files.")


if __name__ == "__main__":
    tyro.cli(delete_files_in_range)
