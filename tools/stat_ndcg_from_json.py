#!/usr/bin/env python3
"""
Script to directly calculate NDCG statistics from existing test_reflection_results.json files.
This is useful when enhanced_analyze skips all users but we still want to see the statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def load_user_records(file_path: Path) -> Optional[Dict]:
    """Load test_records and validation_records from JSON files, matching enhanced_analyze.py logic."""
    try:
        if not file_path.exists():
            return None
            
        user_id = file_path.parent.name
        user_records = {
            'user_id': user_id,
            'test_records': [],
            'validation_records': []
        }
        
        # Load test_reflection_results.json
        test_file = file_path
        if test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                if 'test_records' in test_data and len(test_data['test_records']) > 0:
                    # Add user_id to each record for tracking (like enhanced_analyze.py)
                    for record in test_data['test_records']:
                        record['user_id'] = user_id
                    user_records['test_records'] = test_data['test_records']
        
        # Load valid_reflection_results.json if it exists
        valid_file = file_path.parent / 'valid_reflection_results.json'
        if valid_file.exists():
            with open(valid_file, 'r', encoding='utf-8') as f:
                valid_data = json.load(f)
                if 'validation_records' in valid_data and len(valid_data['validation_records']) > 0:
                    # Add user_id to each record for tracking (like enhanced_analyze.py)
                    for record in valid_data['validation_records']:
                        record['user_id'] = user_id
                    user_records['validation_records'] = valid_data['validation_records']
        
        return user_records if (user_records['test_records'] or user_records['validation_records']) else None
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return None


def calculate_global_statistics(dataset_path: Path) -> Dict:
    """Calculate global NDCG statistics from all users' JSON files, matching enhanced_analyze.py logic exactly."""
    print(f"📂 Scanning directory: {dataset_path}")
    
    # Collect all records (matching _collect_global_statistics in enhanced_analyze.py)
    global_stats = {
        "validation_stats": {"all_records": []},
        "test_stats": {"all_records": []},
        "user_count": 0,
        "successful_users": 0,
        "total_test_users": 0
    }
    
    total_dirs = 0
    processed = 0
    
    # Collect all user records
    for user_dir in sorted(dataset_path.iterdir()):
        if not user_dir.is_dir():
            continue
        
        total_dirs += 1
        json_file = user_dir / 'test_reflection_results.json'
        
        if not json_file.exists():
            continue
        
        processed += 1
        user_records = load_user_records(json_file)
        
        if user_records:
            has_test_records = False
            
            # Load validation records (matching enhanced_analyze.py)
            if user_records['validation_records']:
                global_stats["validation_stats"]["all_records"].extend(user_records['validation_records'])
            
            # Load test records (matching enhanced_analyze.py)
            if user_records['test_records'] and len(user_records['test_records']) > 0:
                global_stats["test_stats"]["all_records"].extend(user_records['test_records'])
                has_test_records = True
            
            if has_test_records:
                global_stats["successful_users"] += 1
                global_stats["total_test_users"] += 1
        
        global_stats["user_count"] += 1
    
    print(f"  - Total directories: {total_dirs}")
    print(f"  - Found JSON files: {processed}")
    print(f"  - Users with test records: {global_stats['successful_users']}")
    
    # Calculate cumulative statistics (matching _calculate_global_reflection_statistics in enhanced_analyze.py)
    def calculate_cumulative_stats(records_list, total_users, phase_name):
        """Calculate cumulative statistics from records, matching enhanced_analyze.py exactly."""
        # Group records by round
        by_round = {}
        for record in records_list:
            round_num = record.get('round', 0)
            if round_num not in by_round:
                by_round[round_num] = []
            by_round[round_num].append(record)
        
        # Get all round numbers and sort
        all_rounds = sorted(by_round.keys())
        
        # Track cumulative state (matching enhanced_analyze.py)
        cumulative_ndcg_1 = {}  # user_id -> latest ndcg value
        cumulative_ndcg_5 = {}
        cumulative_ndcg_10 = {}
        cumulative_ndcg_20 = {}
        cumulative_hit_1 = set()  # users who have hit
        cumulative_hit_5 = set()
        cumulative_hit_10 = set()
        cumulative_hit_20 = set()
        
        result = {}
        
        # Calculate cumulative statistics for each round
        for target_round in all_rounds:
            # Process records for this round
            if target_round in by_round:
                for record in by_round[target_round]:
                    user_id = record.get('user_id', 'unknown')
                    
                    # Update NDCG (always use latest value for each user)
                    cumulative_ndcg_1[user_id] = record.get('ndcg_at_1', 0)
                    cumulative_ndcg_5[user_id] = record.get('ndcg_at_5', 0)
                    cumulative_ndcg_10[user_id] = record.get('ndcg_at_10', 0)
                    cumulative_ndcg_20[user_id] = record.get('ndcg_at_20', 0)
                    
                    # Update hit status (once hit, always counted as hit)
                    if record.get('hit_rate_at_5', 0) > 0:
                        cumulative_hit_5.add(user_id)
                    if record.get('hit_rate_at_10', 0) > 0:
                        cumulative_hit_10.add(user_id)
                    if record.get('hit_rate_at_20', 0) > 0:
                        cumulative_hit_20.add(user_id)
            
            # Calculate statistics
            hit_users_5 = len(cumulative_hit_5)
            hit_users_10 = len(cumulative_hit_10)
            hit_users_20 = len(cumulative_hit_20)
            
            # Calculate NDCG average across ALL users (users without records count as 0)
            avg_ndcg_5 = sum(cumulative_ndcg_5.values()) / total_users if total_users > 0 else 0
            avg_ndcg_10 = sum(cumulative_ndcg_10.values()) / total_users if total_users > 0 else 0
            avg_ndcg_20 = sum(cumulative_ndcg_20.values()) / total_users if total_users > 0 else 0
            
            result[f'round_{target_round}'] = {
                'avg_ndcg_at_5': avg_ndcg_5,
                'avg_ndcg_at_10': avg_ndcg_10,
                'avg_ndcg_at_20': avg_ndcg_20,
                'hit_rate_at_5': hit_users_5 / total_users if total_users > 0 else 0,
                'hit_rate_at_10': hit_users_10 / total_users if total_users > 0 else 0,
                'hit_rate_at_20': hit_users_20 / total_users if total_users > 0 else 0,
                'hit_users_5': hit_users_5,
                'hit_users_10': hit_users_10,
                'hit_users_20': hit_users_20,
                'total_users': total_users,
                'users_with_records': len(cumulative_ndcg_5)
            }
        
        return result
    
    # Total users should be the total number of users processed (matching enhanced_analyze.py)
    total_users = global_stats.get("user_count", global_stats.get("total_test_users", global_stats.get("successful_users", 0)))
    if total_users == 0:
        total_users = 1  # Avoid division by zero
    
    validation_global = calculate_cumulative_stats(
        global_stats["validation_stats"]["all_records"], 
        total_users, 
        "validation"
    )
    test_global = calculate_cumulative_stats(
        global_stats["test_stats"]["all_records"], 
        total_users, 
        "test"
    )
    
    return {
        'dataset': dataset_path.name,
        'total_users': total_dirs,
        'processed_users': processed,
        'successful_users': global_stats['successful_users'],
        'validation_global': validation_global,
        'test_global': test_global
    }


def print_statistics(stats: Dict):
    """Print statistics in a readable format."""
    print("\n" + "=" * 70)
    print(f"📊 全局反思统计结果 (基于 {stats['successful_users']} 个用户)")
    print("=" * 70)
    
    # Print validation statistics
    if stats['validation_global']:
        print("\n🔍 全局验证阶段统计 (累积):")
        for round_key in sorted(stats['validation_global'].keys(), key=lambda x: int(x.replace('round_', ''))):
            round_data = stats['validation_global'][round_key]
            round_num = round_key.replace('round_', '')
            print(f"\n  Round {round_num}:")
            print(f"    NDCG@5  = {round_data['avg_ndcg_at_5']:.4f}")
            print(f"    NDCG@10 = {round_data['avg_ndcg_at_10']:.4f}")
            print(f"    NDCG@20 = {round_data['avg_ndcg_at_20']:.4f}")
            print(f"    HR@5    = {round_data['hit_rate_at_5']:.4f} ({round_data['hit_users_5']}/{round_data['total_users']})")
            print(f"    HR@10   = {round_data['hit_rate_at_10']:.4f} ({round_data['hit_users_10']}/{round_data['total_users']})")
            print(f"    HR@20   = {round_data['hit_rate_at_20']:.4f} ({round_data['hit_users_20']}/{round_data['total_users']})")
            print(f"    Users with records: {round_data['users_with_records']}")
    
    # Print test statistics
    if stats['test_global']:
        print("\n✏️  全局测试阶段统计 (累积):")
        for round_key in sorted(stats['test_global'].keys(), key=lambda x: int(x.replace('round_', ''))):
            round_data = stats['test_global'][round_key]
            round_num = round_key.replace('round_', '')
            print(f"\n  Round {round_num}:")
            print(f"    NDCG@5  = {round_data['avg_ndcg_at_5']:.4f}")
            print(f"    NDCG@10 = {round_data['avg_ndcg_at_10']:.4f}")
            print(f"    NDCG@20 = {round_data['avg_ndcg_at_20']:.4f}")
            print(f"    HR@5    = {round_data['hit_rate_at_5']:.4f} ({round_data['hit_users_5']}/{round_data['total_users']})")
            print(f"    HR@10   = {round_data['hit_rate_at_10']:.4f} ({round_data['hit_users_10']}/{round_data['total_users']})")
            print(f"    HR@20   = {round_data['hit_rate_at_20']:.4f} ({round_data['hit_users_20']}/{round_data['total_users']})")
            print(f"    Users with records: {round_data['users_with_records']}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate NDCG statistics from existing JSON files')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name (bilibili, bilibili_backup, etc.)')
    parser.add_argument('--path', type=str, default=None, help='Direct path to dataset directory')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.path:
        dataset_path = Path(args.path)
    elif args.dataset:
        dataset_path = Path('download') / args.dataset
    else:
        # Try to get from environment or default
        import os
        dataset = os.getenv('DATASET', 'bilibili')
        dataset_path = Path('download') / dataset
    
    if not dataset_path.exists():
        print(f"❌ Directory not found: {dataset_path}")
        return
    
    print("=" * 70)
    print("📊 NDCG Statistics Calculator from JSON Files")
    print("=" * 70)
    
    # Calculate statistics
    stats = calculate_global_statistics(dataset_path)
    
    # Print statistics
    print_statistics(stats)
    
    # Save to file
    output_file = args.output or dataset_path / 'global_reflection_statistics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Statistics saved to: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()

