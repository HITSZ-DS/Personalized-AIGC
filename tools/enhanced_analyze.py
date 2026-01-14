import os
import json
import pickle
import random
import subprocess
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor
from agents.video_analyst import VideoAnalyst
from agents.image_analyst import ImageAnalyst
from agents.text_analyst import TextAnalyst
from agents.user_profile_generator import UserProfileGenerator
from tools.analyze import _convert_user_id, get_chronological_order, _collect_all_items, _process_single_item_task, format_analysis_to_prompt


class EnhancedAnalyzer:
    def __init__(self):
        self.dataset = os.getenv("DATASET", "bilibili")
        self.video_analyst = VideoAnalyst(max_workers=2)
        self.image_analyst = ImageAnalyst(max_workers=2)
        self.text_analyst = TextAnalyst(max_workers=2)
        self.user_profile_generator = UserProfileGenerator()

        # Test control parameter: True = test each round, False = test only final round
        self.test_each_round = os.getenv("TEST_EACH_ROUND", "true").lower() == "true"

        # Load mapping files
        self.item_map = self._load_item_map()
        self.user_map = self._load_user_map()

        # Load test and validation matrices
        self.test_matrix = self._load_matrix("test_matrix.pkl")
        self.valid_matrix = self._load_matrix("valid_matrix.pkl")

        # Cache for item mapping (douban_mapping.json or bilibili_mapping.json)
        self._cached_item_mapping = None
        self._load_cached_item_mapping()

        # Thread pool for API calls
        self.api_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="API-")

    def __del__(self):
        """Cleanup method to shutdown thread pool"""
        if hasattr(self, 'api_thread_pool'):
            self.api_thread_pool.shutdown(wait=False)

    def _load_item_map(self) -> Dict[str, str]:
        """Load item mapping from index to real item ID"""
        try:
            item_map_path = f"SSLRec/datasets/general_cf/{self.dataset}/item_map.json"
            with open(item_map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading item map: {e}")
            return {}

    def _load_user_map(self) -> Dict[str, str]:
        """Load user mapping from index to real user ID"""
        try:
            user_map_path = f"SSLRec/datasets/general_cf/{self.dataset}/user_map.json"
            with open(user_map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading user map: {e}")
            return {}

    def _load_matrix(self, filename: str):
        """Load pickle matrix file"""
        try:
            matrix_path = f"SSLRec/datasets/general_cf/{self.dataset}/{filename}"
            with open(matrix_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {filename}: {e}")
            return None

    def get_user_test_items(self, user_internal_id: int) -> List[str]:
        """Get test items for a user (convert from internal ID to real item IDs)"""
        if self.test_matrix is None or user_internal_id >= self.test_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in test set
        test_item_indices = self.test_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        test_items = []
        for item_idx in test_item_indices:
            real_item_id = self.item_map.get(str(item_idx))
            if real_item_id:
                test_items.append(real_item_id)

        return test_items

    def get_user_validation_items(self, user_internal_id: int) -> List[str]:
        """Get validation items for a user (convert from internal ID to real item IDs)"""
        if self.valid_matrix is None or user_internal_id >= self.valid_matrix.shape[0]:
            return []

        # Get item indices that this user interacted with in validation set
        valid_item_indices = self.valid_matrix[user_internal_id].nonzero()[1]

        # Convert to real item IDs
        valid_items = []
        for item_idx in valid_item_indices:
            real_item_id = self.item_map.get(str(item_idx))
            if real_item_id:
                valid_items.append(real_item_id)

        return valid_items

    def check_recommendation_hits(self, user_internal_id: int, recommended_items: List[str]) -> Dict:
        """Check if recommended items hit test items"""
        test_items = self.get_user_test_items(user_internal_id)
        print(f"[INFO] 测试视频: {test_items}")
        hits = []
        for rec_item in recommended_items:
            if rec_item in test_items:
                hits.append(rec_item)

        hit_rate = len(hits) / len(recommended_items) if recommended_items else 0

        return {
            "test_items": test_items,
            "recommended_items": recommended_items,
            "hits": hits,
            "hit_count": len(hits),
            "total_recommendations": len(recommended_items),
            "hit_rate": hit_rate
        }

    def get_user_interacted_items(self, user_real_id: str) -> Set[str]:
        """Get all items that user has interacted with from dataset CSV files"""
        interacted_items = set()

        user_dataset_path = f"dataset/{self.dataset}/{user_real_id}"
        if not os.path.exists(user_dataset_path):
            return interacted_items

        # Dataset-specific column mapping
        if self.dataset == 'bilibili':
            item_id_col = 'bvid'
        elif self.dataset == 'douban':
            item_id_col = 'doubanID'  # Use doubanID as unique identifier for douban
        elif self.dataset == 'redbook':
            item_id_col = 'redbookID'  # Use redbookID as unique identifier for redbook
        elif self.dataset == 'hupu':
            item_id_col = 'hupuID'  # Use hupuID as unique identifier for hupu
        else:
            item_id_col = 'note_id'  # Default fallback

        # Read all CSV files
        for file in os.listdir(user_dataset_path):
            if file.endswith('.csv'):
                try:
                    csv_path = os.path.join(user_dataset_path, file)
                    df = pd.read_csv(csv_path)
                    if item_id_col in df.columns:
                        interacted_items.update(df[item_id_col].tolist())
                except Exception as e:
                    print(f"⚠️  Error reading {file}: {e}")

        return interacted_items

    def get_video_title(self, bvid: str, max_retries: int = 3) -> Optional[str]:
        """Get video title using yt-dlp without downloading the video"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = random.uniform(2, 5)  # Shorter delay for title fetching
                    time.sleep(delay)

                url = f"https://www.bilibili.com/video/{bvid}"

                # Use yt-dlp to get video info only
                cmd = [
                    "yt-dlp",
                    "--get-title",
                    "--no-warnings",
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "--referer", "https://www.bilibili.com/",
                    url
                ]

                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )

                if process.returncode == 0:
                    title = process.stdout.strip()
                    if title:
                        return title
                else:
                    error_msg = process.stderr
                    # Check if it's a permanent error
                    if any(keyword in error_msg.lower() for keyword in ['private', 'deleted', 'not available', 'geo-blocked']):
                        print(f"⚠️  Cannot get title for {bvid}: {error_msg}")
                        return None

            except Exception as e:
                print(f"⚠️  Error getting title for {bvid}: {e}")

        print(f"⚠️  Failed to get title for {bvid} after {max_retries} attempts")
        return None



    def _load_cached_item_mapping(self):
        """Load and cache item mapping during initialization"""
        # Configuration for dataset mapping files
        dataset_mapping_config = {
            'douban': {'file': 'douban_mapping.json', 'name': 'Douban'},
            'bilibili': {'file': 'bilibili_mapping.json', 'name': 'Bilibili'},
            'redbook': {'file': 'redbook_mapping.json', 'name': 'Redbook'},
            'hupu': {'file': 'hupu_mapping.json', 'name': 'Hupu'}
        }

        try:
            config = dataset_mapping_config.get(self.dataset)
            if not config:
                print(f"⚠️  No mapping file configured for dataset: {self.dataset}")
                self._cached_item_mapping = {}
                return

            mapping_path = config['file']
            mapping_name = config['name']

            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self._cached_item_mapping = json.load(f)
                print(f"✅ Loaded {len(self._cached_item_mapping)} items from {mapping_name} mapping")
            else:
                print(f"⚠️  {mapping_name} mapping file not found: {mapping_path}")
                self._cached_item_mapping = {}
        except Exception as e:
            print(f"⚠️  Error loading {config['name'] if config else 'unknown'} mapping: {e}")
            self._cached_item_mapping = {}

    def _get_cached_item_mapping(self) -> Dict[str, str]:
        """Get cached item mapping without reloading"""
        return self._cached_item_mapping if self._cached_item_mapping is not None else {}

    def _get_hit_items_titles(self, hit_items: List[str]) -> List[str]:
        """Convert hit item IDs to their corresponding titles"""
        item_mapping = self._get_cached_item_mapping()
        hit_titles = []
        for item_id in hit_items:
            title = self._extract_title_from_mapping(item_id, item_mapping, f"Item_{item_id}")
            hit_titles.append(title)
        return hit_titles

    def _calculate_reflection_statistics(self, valid_records: List[Dict], test_records: List[Dict]) -> Dict:
        """Calculate average NDCG and Hit statistics for each reflection round"""
        stats = {
            "validation_stats": {},
            "test_stats": {},
            "overall_stats": {}
        }

        # Calculate validation statistics by round
        validation_by_round = {}
        for record in valid_records:
            round_num = record.get('round', 0)
            if round_num not in validation_by_round:
                validation_by_round[round_num] = []
            validation_by_round[round_num].append(record)

        for round_num, records in validation_by_round.items():
            ndcg_1_values = [r.get('ndcg_at_1', 0) for r in records]
            ndcg_5_values = [r.get('ndcg_at_5', 0) for r in records]
            ndcg_10_values = [r.get('ndcg_at_10', 0) for r in records]
            ndcg_20_values = [r.get('ndcg_at_20', 0) for r in records]
            hit_rate_1_values = [r.get('hit_rate_at_1', 0) for r in records]
            hit_rate_5_values = [r.get('hit_rate_at_5', 0) for r in records]
            hit_rate_10_values = [r.get('hit_rate_at_10', 0) for r in records]
            hit_rate_20_values = [r.get('hit_rate_at_20', 0) for r in records]
            hit_values = [1 if r.get('hit_success', False) else 0 for r in records]

            stats["validation_stats"][f"round_{round_num}"] = {
                "avg_ndcg_at_1": sum(ndcg_1_values) / len(ndcg_1_values) if ndcg_1_values else 0,
                "avg_ndcg_at_5": sum(ndcg_5_values) / len(ndcg_5_values) if ndcg_5_values else 0,
                "avg_ndcg_at_10": sum(ndcg_10_values) / len(ndcg_10_values) if ndcg_10_values else 0,
                "avg_ndcg_at_20": sum(ndcg_20_values) / len(ndcg_20_values) if ndcg_20_values else 0,
                "avg_hit_rate_at_1": sum(hit_rate_1_values) / len(hit_rate_1_values) if hit_rate_1_values else 0,
                "avg_hit_rate_at_5": sum(hit_rate_5_values) / len(hit_rate_5_values) if hit_rate_5_values else 0,
                "avg_hit_rate_at_10": sum(hit_rate_10_values) / len(hit_rate_10_values) if hit_rate_10_values else 0,
                "avg_hit_rate_at_20": sum(hit_rate_20_values) / len(hit_rate_20_values) if hit_rate_20_values else 0,
                "hit_rate": sum(hit_values) / len(hit_values) if hit_values else 0,
                "total_attempts": len(records)
            }

        # Calculate test statistics by round
        test_by_round = {}
        for record in test_records:
            round_num = record.get('round', 0)
            if round_num not in test_by_round:
                test_by_round[round_num] = []
            test_by_round[round_num].append(record)

        for round_num, records in test_by_round.items():
            ndcg_1_values = [r.get('ndcg_at_1', 0) for r in records]
            ndcg_5_values = [r.get('ndcg_at_5', 0) for r in records]
            ndcg_10_values = [r.get('ndcg_at_10', 0) for r in records]
            ndcg_20_values = [r.get('ndcg_at_20', 0) for r in records]
            hit_rate_1_values = [r.get('hit_rate_at_1', 0) for r in records]
            hit_rate_5_values = [r.get('hit_rate_at_5', 0) for r in records]
            hit_rate_10_values = [r.get('hit_rate_at_10', 0) for r in records]
            hit_rate_20_values = [r.get('hit_rate_at_20', 0) for r in records]
            hit_values = [1 if r.get('hit_success', False) else 0 for r in records]

            stats["test_stats"][f"round_{round_num}"] = {
                "avg_ndcg_at_1": sum(ndcg_1_values) / len(ndcg_1_values) if ndcg_1_values else 0,
                "avg_ndcg_at_5": sum(ndcg_5_values) / len(ndcg_5_values) if ndcg_5_values else 0,
                "avg_ndcg_at_10": sum(ndcg_10_values) / len(ndcg_10_values) if ndcg_10_values else 0,
                "avg_ndcg_at_20": sum(ndcg_20_values) / len(ndcg_20_values) if ndcg_20_values else 0,
                "avg_hit_rate_at_1": sum(hit_rate_1_values) / len(hit_rate_1_values) if hit_rate_1_values else 0,
                "avg_hit_rate_at_5": sum(hit_rate_5_values) / len(hit_rate_5_values) if hit_rate_5_values else 0,
                "avg_hit_rate_at_10": sum(hit_rate_10_values) / len(hit_rate_10_values) if hit_rate_10_values else 0,
                "avg_hit_rate_at_20": sum(hit_rate_20_values) / len(hit_rate_20_values) if hit_rate_20_values else 0,
                "hit_rate": sum(hit_values) / len(hit_values) if hit_values else 0,
                "total_attempts": len(records)
            }

        # Calculate overall statistics
        all_valid_ndcg_1 = [r.get('ndcg_at_1', 0) for r in valid_records]
        all_valid_ndcg_5 = [r.get('ndcg_at_5', 0) for r in valid_records]
        all_valid_ndcg_10 = [r.get('ndcg_at_10', 0) for r in valid_records]
        all_valid_ndcg_20 = [r.get('ndcg_at_20', 0) for r in valid_records]
        all_valid_hit_rate_1 = [r.get('hit_rate_at_1', 0) for r in valid_records]
        all_valid_hit_rate_5 = [r.get('hit_rate_at_5', 0) for r in valid_records]
        all_valid_hit_rate_10 = [r.get('hit_rate_at_10', 0) for r in valid_records]
        all_valid_hit_rate_20 = [r.get('hit_rate_at_20', 0) for r in valid_records]
        all_valid_hits = [1 if r.get('hit_success', False) else 0 for r in valid_records]

        all_test_ndcg_1 = [r.get('ndcg_at_1', 0) for r in test_records]
        all_test_ndcg_5 = [r.get('ndcg_at_5', 0) for r in test_records]
        all_test_ndcg_10 = [r.get('ndcg_at_10', 0) for r in test_records]
        all_test_ndcg_20 = [r.get('ndcg_at_20', 0) for r in test_records]
        all_test_hit_rate_1 = [r.get('hit_rate_at_1', 0) for r in test_records]
        all_test_hit_rate_5 = [r.get('hit_rate_at_5', 0) for r in test_records]
        all_test_hit_rate_10 = [r.get('hit_rate_at_10', 0) for r in test_records]
        all_test_hit_rate_20 = [r.get('hit_rate_at_20', 0) for r in test_records]
        all_test_hits = [1 if r.get('hit_success', False) else 0 for r in test_records]

        stats["overall_stats"] = {
            "validation": {
                "avg_ndcg_at_1": sum(all_valid_ndcg_1) / len(all_valid_ndcg_1) if all_valid_ndcg_1 else 0,
                "avg_ndcg_at_5": sum(all_valid_ndcg_5) / len(all_valid_ndcg_5) if all_valid_ndcg_5 else 0,
                "avg_ndcg_at_10": sum(all_valid_ndcg_10) / len(all_valid_ndcg_10) if all_valid_ndcg_10 else 0,
                "avg_ndcg_at_20": sum(all_valid_ndcg_20) / len(all_valid_ndcg_20) if all_valid_ndcg_20 else 0,
                "avg_hit_rate_at_1": sum(all_valid_hit_rate_1) / len(all_valid_hit_rate_1) if all_valid_hit_rate_1 else 0,
                "avg_hit_rate_at_5": sum(all_valid_hit_rate_5) / len(all_valid_hit_rate_5) if all_valid_hit_rate_5 else 0,
                "avg_hit_rate_at_10": sum(all_valid_hit_rate_10) / len(all_valid_hit_rate_10) if all_valid_hit_rate_10 else 0,
                "avg_hit_rate_at_20": sum(all_valid_hit_rate_20) / len(all_valid_hit_rate_20) if all_valid_hit_rate_20 else 0,
                "hit_rate": sum(all_valid_hits) / len(all_valid_hits) if all_valid_hits else 0,
                "total_attempts": len(valid_records)
            },
            "test": {
                "avg_ndcg_at_1": sum(all_test_ndcg_1) / len(all_test_ndcg_1) if all_test_ndcg_1 else 0,
                "avg_ndcg_at_5": sum(all_test_ndcg_5) / len(all_test_ndcg_5) if all_test_ndcg_5 else 0,
                "avg_ndcg_at_10": sum(all_test_ndcg_10) / len(all_test_ndcg_10) if all_test_ndcg_10 else 0,
                "avg_ndcg_at_20": sum(all_test_ndcg_20) / len(all_test_ndcg_20) if all_test_ndcg_20 else 0,
                "avg_hit_rate_at_1": sum(all_test_hit_rate_1) / len(all_test_hit_rate_1) if all_test_hit_rate_1 else 0,
                "avg_hit_rate_at_5": sum(all_test_hit_rate_5) / len(all_test_hit_rate_5) if all_test_hit_rate_5 else 0,
                "avg_hit_rate_at_10": sum(all_test_hit_rate_10) / len(all_test_hit_rate_10) if all_test_hit_rate_10 else 0,
                "avg_hit_rate_at_20": sum(all_test_hit_rate_20) / len(all_test_hit_rate_20) if all_test_hit_rate_20 else 0,
                "hit_rate": sum(all_test_hits) / len(all_test_hits) if all_test_hits else 0,
                "total_attempts": len(test_records)
            }
        }

        return stats

    def _extract_title_from_mapping(self, item_id: str, item_mapping: Dict, fallback: str = None) -> str:
        """Extract title from mapping based on dataset type"""
        if not item_mapping or item_id not in item_mapping:
            return fallback or f"Item_{item_id}"

        mapping_value = item_mapping[item_id]

        # For redbook, hupu, douban: value is a dict with 'title' field
        if self.dataset in ['bilibili','redbook', 'hupu', 'douban']:
            if isinstance(mapping_value, dict) and 'title' in mapping_value:
                return mapping_value['title']
            else:
                return fallback or f"Item_{item_id}"

        # Default fallback
        return fallback or f"Item_{item_id}"

    def _extract_title_and_tags_from_mapping(self, item_id: str, item_mapping: Dict, fallback: str = None) -> tuple:
        """Extract title and tags from mapping based on dataset type"""
        if not item_mapping or item_id not in item_mapping:
            return (fallback or f"Item_{item_id}", "")

        mapping_value = item_mapping[item_id]

        # For all datasets: value is a dict with 'title' and 'tags' fields
        if isinstance(mapping_value, dict):
            title = mapping_value.get('title', fallback or f"Item_{item_id}")
            tags = mapping_value.get('tags', '')
            # Truncate tags to 30 characters
            if tags and len(tags) > 30:
                tags = tags[:30]
            return (title, tags)
        else:
            return (fallback or f"Item_{item_id}", "")

    def _calculate_ndcg_at_k(self, predicted_items: List[str], target_items: List[str], k: int = None) -> float:
        """Calculate NDCG@K metric"""
        if not target_items or not predicted_items:
            return 0.0

        if k is None:
            k = len(predicted_items)

        predicted_k = predicted_items[:k]
        target_set = set(target_items)

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            if item in target_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(len(target_items), k)):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_hit_rate_at_k(self, predicted_items: List[str], target_items: List[str], k: int = None) -> float:
        """Calculate Hit Rate@K metric (Recall@K)"""
        if not target_items or not predicted_items:
            return 0.0

        if k is None:
            k = len(predicted_items)

        predicted_k = predicted_items[:k]
        target_set = set(target_items)

        # Count hits in top-k predictions
        hits = sum(1 for item in predicted_k if item in target_set)

        # Hit Rate@K = hits / min(k, |target_items|)
        return hits / min(k, len(target_items)) if min(k, len(target_items)) > 0 else 0.0

    def _calculate_test_metrics(self, predicted_items: List[str], target_items: List[str]) -> Dict:
        """Calculate comprehensive test metrics including NDCG and Hit Rate at different K values"""
        if not predicted_items or not target_items:
            return {
                "hit_count": 0,
                "hit_success": False,
                "ndcg_at_1": 0.0,
                "ndcg_at_5": 0.0,
                "ndcg_at_10": 0.0,
                "ndcg_at_20": 0.0,
                "hit_rate_at_1": 0.0,
                "hit_rate_at_5": 0.0,
                "hit_rate_at_10": 0.0,
                "hit_rate_at_20": 0.0
            }

        hit_items = [item for item in predicted_items if item in target_items]
        hit_count = len(hit_items)
        hit_success = hit_count > 0

        # Calculate NDCG at different K values
        ndcg_at_1 = self._calculate_ndcg_at_k(predicted_items, target_items, 1)
        ndcg_at_5 = self._calculate_ndcg_at_k(predicted_items, target_items, 5)
        ndcg_at_10 = self._calculate_ndcg_at_k(predicted_items, target_items, 10)
        ndcg_at_20 = self._calculate_ndcg_at_k(predicted_items, target_items, 20)

        # Calculate Hit Rate at different K values
        hit_rate_at_1 = self._calculate_hit_rate_at_k(predicted_items, target_items, 1)
        hit_rate_at_5 = self._calculate_hit_rate_at_k(predicted_items, target_items, 5)
        hit_rate_at_10 = self._calculate_hit_rate_at_k(predicted_items, target_items, 10)
        hit_rate_at_20 = self._calculate_hit_rate_at_k(predicted_items, target_items, 20)

        return {
            "hit_count": hit_count,
            "hit_success": hit_success,
            "ndcg_at_1": ndcg_at_1,
            "ndcg_at_5": ndcg_at_5,
            "ndcg_at_10": ndcg_at_10,
            "ndcg_at_20": ndcg_at_20,
            "hit_rate_at_1": hit_rate_at_1,
            "hit_rate_at_5": hit_rate_at_5,
            "hit_rate_at_10": hit_rate_at_10,
            "hit_rate_at_20": hit_rate_at_20
        }

    def get_random_items_with_titles(self, user_real_id: str, recommended_items: List[str], count: int = 99, user_internal_id: Optional[int] = None) -> List[Dict[str, str]]:
        """Get random items from dataset with their titles using mapping files"""
        # Get cached item mapping
        item_mapping = self._get_cached_item_mapping()

        if not item_mapping:
            print("⚠️  No item mapping available, falling back to API/CSV method")
            if self.dataset == 'douban':
                return self._get_random_items_with_titles_douban(user_real_id, recommended_items, count, user_internal_id)
            else:
                return self._get_random_items_with_titles_api(user_real_id, recommended_items, count, user_internal_id)

        # For douban, use the mapping file approach similar to bilibili
        if self.dataset == 'douban':
            return self._get_random_items_with_titles_from_mapping(user_real_id, recommended_items, count, item_mapping, 'doubanID', user_internal_id)
        elif self.dataset == 'redbook':
            return self._get_random_items_with_titles_from_mapping(user_real_id, recommended_items, count, item_mapping, 'redbookID', user_internal_id)
        elif self.dataset == 'hupu':
            return self._get_random_items_with_titles_from_mapping(user_real_id, recommended_items, count, item_mapping, 'hupuID', user_internal_id)
        else:
            return self._get_random_items_with_titles_from_mapping(user_real_id, recommended_items, count, item_mapping, 'bvid', user_internal_id)

    def _get_random_items_with_titles_from_mapping(self, user_real_id: str, recommended_items: List[str], count: int, item_mapping: Dict[str, str], item_key: str, user_internal_id: Optional[int] = None) -> List[Dict[str, str]]:
        """Generic function to get random items with titles from mapping file"""
        # Get all items user has interacted with
        interacted_items = self.get_user_interacted_items(user_real_id)

        # Get test and validation items if user_internal_id is provided
        test_items = set()
        validation_items = set()
        if user_internal_id is not None:
            test_items = set(self.get_user_test_items(user_internal_id))
            validation_items = set(self.get_user_validation_items(user_internal_id))

        # Combine all items to exclude: historical + recommended + test + validation
        exclude_items = interacted_items.union(set(recommended_items)).union(test_items).union(validation_items)

        print(f"🚫 Excluding {len(exclude_items)} items: {len(interacted_items)} historical + {len(recommended_items)} recommended + {len(test_items)} test + {len(validation_items)} validation")

        # Get available items from the mapping that are not excluded
        available_items = [item_id for item_id in item_mapping.keys() if item_id not in exclude_items]

        print(f"🎲 Found {len(available_items)} available items in {self.dataset} mapping (excluding {len(exclude_items)} interacted/recommended)")

        # Randomly sample the requested count
        if len(available_items) < count:
            print(f"⚠️  Only {len(available_items)} items available in mapping, requested {count}")
            selected_items = available_items
        else:
            selected_items = random.sample(available_items, count)

        # Create items with titles using the mapping
        items_with_titles = []
        for item_id in selected_items:
            title = self._extract_title_from_mapping(item_id, item_mapping, item_id)
            items_with_titles.append({
                item_key: item_id,
                "title": title
            })

        print(f"✅ Retrieved {len(items_with_titles)} items with titles from {self.dataset} mapping")
        return items_with_titles

    def _get_random_items_with_titles_douban(self, user_real_id: str, recommended_items: List[str], count: int = 99, user_internal_id: Optional[int] = None) -> List[Dict[str, str]]:
        """Get random items from douban dataset with their douban_id and titles"""
        # Get all items user has interacted with
        interacted_items = self.get_user_interacted_items(user_real_id)

        # Get test and validation items if user_internal_id is provided
        test_items = set()
        validation_items = set()
        if user_internal_id is not None:
            test_items = set(self.get_user_test_items(user_internal_id))
            validation_items = set(self.get_user_validation_items(user_internal_id))

        # Combine all items to exclude: historical + recommended + test + validation
        exclude_items = interacted_items.union(set(recommended_items)).union(test_items).union(validation_items)

        print(f"🚫 Excluding {len(exclude_items)} items: {len(interacted_items)} historical + {len(recommended_items)} recommended + {len(test_items)} test + {len(validation_items)} validation")

        # Get all available items from dataset
        all_available_items = []
        dataset_root = f"dataset/{self.dataset}"

        if os.path.exists(dataset_root):
            # Collect all unique douban_id and title pairs from all users' CSV files
            for user_folder in os.listdir(dataset_root):
                user_path = os.path.join(dataset_root, user_folder)
                if not os.path.isdir(user_path):
                    continue

                for file in os.listdir(user_path):
                    if file.endswith('.csv'):
                        try:
                            csv_path = os.path.join(user_path, file)
                            df = pd.read_csv(csv_path)
                            if 'doubanID' in df.columns and 'title' in df.columns:
                                for _, row in df.iterrows():
                                    douban_id = str(row['doubanID'])
                                    title = str(row['title'])
                                    all_available_items.append({
                                        'doubanID': douban_id,
                                        'title': title
                                    })
                        except Exception as e:
                            continue

        # Remove duplicates based on doubanID and exclude items
        seen_ids = set()
        unique_items = []
        for item in all_available_items:
            if item['doubanID'] not in seen_ids and item['doubanID'] not in exclude_items:
                unique_items.append(item)
                seen_ids.add(item['doubanID'])

        print(f"🎲 Found {len(unique_items)} available douban items (excluding {len(exclude_items)} interacted/recommended)")

        # Randomly sample the requested count
        if len(unique_items) < count:
            print(f"⚠️  Only {len(unique_items)} items available, requested {count}")
            selected_items = unique_items
        else:
            selected_items = random.sample(unique_items, count)

        # Create items with titles (for douban, doubanID is the identifier)
        items_with_titles = []
        for item in selected_items:
            items_with_titles.append({
                "doubanID": item['doubanID'],
                "title": item['title']
            })

        print(f"✅ Retrieved {len(items_with_titles)} douban items with titles")
        return items_with_titles

    def _get_random_items_with_titles_api(self, user_real_id: str, recommended_items: List[str], count: int = 99, user_internal_id: Optional[int] = None) -> List[Dict[str, str]]:
        """Fallback method using API calls (original implementation)"""
        # Get all items user has interacted with
        interacted_items = self.get_user_interacted_items(user_real_id)

        # Get test and validation items if user_internal_id is provided
        test_items = set()
        validation_items = set()
        if user_internal_id is not None:
            test_items = set(self.get_user_test_items(user_internal_id))
            validation_items = set(self.get_user_validation_items(user_internal_id))

        # Combine all items to exclude: historical + recommended + test + validation
        exclude_items = interacted_items.union(set(recommended_items)).union(test_items).union(validation_items)

        print(f"🚫 Excluding {len(exclude_items)} items: {len(interacted_items)} historical + {len(recommended_items)} recommended + {len(test_items)} test + {len(validation_items)} validation")

        # Get all available items from dataset
        all_available_items = []
        dataset_root = f"dataset/{self.dataset}"

        if os.path.exists(dataset_root):
            # Collect all unique item names from all users' CSV files
            if self.dataset == 'bilibili':
                item_id_col = 'bvid'
            elif self.dataset == 'douban':
                item_id_col = 'doubanID'  # Use doubanID as unique identifier for douban
            elif self.dataset == 'redbook':
                item_id_col = 'redbookID'  # Use redbookID as unique identifier for redbook
            elif self.dataset == 'hupu':
                item_id_col = 'hupuID'  # Use hupuID as unique identifier for hupu
            else:
                item_id_col = 'note_id'  # Default fallback

            for user_folder in os.listdir(dataset_root):
                user_path = os.path.join(dataset_root, user_folder)
                if not os.path.isdir(user_path):
                    continue

                for file in os.listdir(user_path):
                    if file.endswith('.csv'):
                        try:
                            csv_path = os.path.join(user_path, file)
                            df = pd.read_csv(csv_path)
                            if item_id_col in df.columns:
                                all_available_items.extend(df[item_id_col].tolist())
                        except Exception as e:
                            continue

        # Remove duplicates and excluded items
        available_items = list(set(all_available_items) - exclude_items)

        # Randomly sample the requested count
        if len(available_items) < count:
            print(f"⚠️  Only {len(available_items)} items available, requested {count}")
            selected_items = available_items
        else:
            selected_items = random.sample(available_items, count)

        print(f"🎲 Getting titles for {len(selected_items)} random items using API...")

        # Get titles for selected items in batches to avoid overwhelming the API
        items_with_titles = []
        batch_size = 5

        for i in range(0, len(selected_items), batch_size):
            batch = selected_items[i:i+batch_size]
            print(f"[INFO] Processing title batch {i//batch_size + 1}/{(len(selected_items)-1)//batch_size + 1}")

            # Get titles for current batch synchronously
            titles = []
            for bvid in batch:
                try:
                    title = self.get_video_title(bvid)
                    titles.append(title)
                except Exception as e:
                    titles.append(e)

            # Combine bvids with their titles
            for bvid, title in zip(batch, titles):
                if isinstance(title, str) and title:
                    items_with_titles.append({
                        "bvid": bvid,
                        "title": title
                    })
                else:
                    # Fallback to bvid if title fetch failed
                    items_with_titles.append({
                        "bvid": bvid,
                        "title": bvid  # Use bvid as fallback
                    })

            # Add delay between batches
            if i + batch_size < len(selected_items):
                time.sleep(2)

        print(f"✅ Retrieved titles for {len(items_with_titles)} items using API")
        return items_with_titles

    def get_random_items(self, user_real_id: str, recommended_items: List[str], count: int = 99) -> List[str]:
        """Get random items from dataset that user hasn't interacted with and weren't recommended (legacy method)"""
        # Get all items user has interacted with
        interacted_items = self.get_user_interacted_items(user_real_id)

        # Combine interacted and recommended items to exclude
        exclude_items = interacted_items.union(set(recommended_items))

        # Get all available items from dataset
        all_available_items = []
        dataset_root = f"dataset/{self.dataset}"

        if os.path.exists(dataset_root):
            # Collect all unique item names from all users' CSV files
            if self.dataset == 'bilibili':
                item_id_col = 'bvid'
            elif self.dataset == 'douban':
                item_id_col = 'doubanID'  # Use doubanID as unique identifier for douban
            elif self.dataset == 'redbook':
                item_id_col = 'redbookID'  # Use redbookID as unique identifier for redbook
            elif self.dataset == 'hupu':
                item_id_col = 'hupuID'  # Use hupuID as unique identifier for hupu
            else:
                item_id_col = 'note_id'  # Default fallback

            for user_folder in os.listdir(dataset_root):
                user_path = os.path.join(dataset_root, user_folder)
                if not os.path.isdir(user_path):
                    continue

                for file in os.listdir(user_path):
                    if file.endswith('.csv'):
                        try:
                            csv_path = os.path.join(user_path, file)
                            df = pd.read_csv(csv_path)
                            if item_id_col in df.columns:
                                all_available_items.extend(df[item_id_col].tolist())
                        except Exception as e:
                            continue

        # Remove duplicates and excluded items
        available_items = list(set(all_available_items) - exclude_items)

        # Randomly sample the requested count
        if len(available_items) < count:
            print(f"⚠️  Only {len(available_items)} items available, requested {count}")
            return available_items

        return random.sample(available_items, count)

    def _process_validation_items(self, validation_dir: str) -> List[str]:
        """
        Process validation items using the new concurrent processing approach

        Args:
            validation_dir: Directory containing validation items

        Returns:
            List of formatted prompts for validation items
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not os.path.exists(validation_dir):
            print(f"⚠️  Validation directory not found: {validation_dir}")
            return []

        # Collect all validation item tasks
        validation_tasks = []
        for item_name in os.listdir(validation_dir):
            item_path = os.path.join(validation_dir, item_name)
            if os.path.isdir(item_path):
                validation_tasks.append({
                    'user_id': 'validation',
                    'item_type': 'validation',
                    'item_name': item_name,
                    'item_path': item_path
                })

        if not validation_tasks:
            print(f"⚠️  No validation items found in: {validation_dir}")
            return []

        print(f"📊 Processing {len(validation_tasks)} validation items...")

        # Process validation items concurrently
        validation_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(_process_single_item_task, task, self.video_analyst, self.image_analyst, self.text_analyst): task
                for task in validation_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if 'formatted_prompt' in result and result['formatted_prompt']:
                        validation_results.append(result['formatted_prompt'])
                        print(f"  ✅ Processed validation item: {task['item_name']}")
                    else:
                        print(f"  ⚠️  No analysis result for validation item: {task['item_name']}")
                except Exception as e:
                    print(f"  ❌ Error processing validation item {task['item_name']}: {e}")

        print(f"✅ Completed processing {len(validation_results)} validation items")
        return validation_results

    def self_reflection_predict(self, user_profile: str, validation_item_names: List[str], random_items_with_titles: List[Dict[str, str]], user_real_id: str, max_reflections: int = 3, test_each_round: bool = True) -> Tuple[bool, str, List[Dict]]:
        """
        Implement self-reflection mechanism for user preference prediction

        Args:
            user_profile: Current user profile
            validation_item_names: List of validation item BVIDs
            random_items_with_titles: List of dicts with 'bvid' and 'title' keys
            user_real_id: User ID
            max_reflections: Maximum reflection rounds
            test_each_round: If True, test after each reflection round; if False, only test after final round

        Returns:
            Tuple[bool, str, List[Dict]]: (hit_success, final_user_profile, test_records)
        """
        # Handle title mapping based on dataset type
        if self.dataset == 'douban':
            # For douban, validation_item_names are DoubanIDs, need to get titles
            validation_titles = []
            validation_item_to_title = {}

            # Load douban mapping to get titles
            item_mapping = self._get_cached_item_mapping()
            if not item_mapping:
                print("⚠️  No douban mapping available, using fallback titles")
                for douban_id in validation_item_names:
                    title = f"Item_{douban_id}"  # Fallback title
                    validation_titles.append(title)
                    validation_item_to_title[douban_id] = title
            else:
                # Map validation DoubanIDs to titles using mapping file
                for douban_id in validation_item_names:
                    title = self._extract_title_from_mapping(douban_id, item_mapping, f"Item_{douban_id}")
                    validation_titles.append(title)
                    validation_item_to_title[douban_id] = title
        elif self.dataset == 'redbook':
            # For redbook, validation_item_names are redbookIDs, need to get titles
            validation_titles = []
            validation_item_to_title = {}

            # Load redbook mapping to get titles
            item_mapping = self._get_cached_item_mapping()
            if not item_mapping:
                print("⚠️  No redbook mapping available, using fallback titles")
                for redbook_id in validation_item_names:
                    title = f"Item_{redbook_id}"  # Fallback title
                    validation_titles.append(title)
                    validation_item_to_title[redbook_id] = title
            else:
                # Map validation redbookIDs to titles using mapping file
                for redbook_id in validation_item_names:
                    title = self._extract_title_from_mapping(redbook_id, item_mapping, f"Item_{redbook_id}")
                    validation_titles.append(title)
                    validation_item_to_title[redbook_id] = title
        elif self.dataset == 'hupu':
            # For hupu, validation_item_names are hupuIDs, need to get titles
            validation_titles = []
            validation_item_to_title = {}

            # Load hupu mapping to get titles
            item_mapping = self._get_cached_item_mapping()
            if not item_mapping:
                print("⚠️  No hupu mapping available, using fallback titles")
                for hupu_id in validation_item_names:
                    title = f"Item_{hupu_id}"  # Fallback title
                    validation_titles.append(title)
                    validation_item_to_title[hupu_id] = title
            else:
                # Map validation hupuIDs to titles using mapping file
                for hupu_id in validation_item_names:
                    title = self._extract_title_from_mapping(hupu_id, item_mapping, f"Item_{hupu_id}")
                    validation_titles.append(title)
                    validation_item_to_title[hupu_id] = title
        else:
            # Load item mapping to get real titles for validation items
            item_mapping = self._get_cached_item_mapping()

            # Convert validation BVIDs to real titles
            validation_titles = []
            validation_item_to_title = {}

            for bvid in validation_item_names:
                title = self._extract_title_from_mapping(bvid, item_mapping, bvid)
                validation_titles.append(title)
                validation_item_to_title[bvid] = title

        # Create list of all titles for prediction
        random_titles = [item['title'] for item in random_items_with_titles]
        all_titles = validation_titles + random_titles
        random.shuffle(all_titles)

        # Save all titles for observation
        titles_file = os.path.join("download", self.dataset, user_real_id, "self_reflection_titles.json")
        titles_data = {
            "all_titles": all_titles,
            "validation_titles": validation_titles,
            "random_titles": random_titles,
            "total_count": len(all_titles)
        }
        try:
            with open(titles_file, 'w', encoding='utf-8') as f:
                json.dump(titles_data, f, ensure_ascii=False, indent=2)
            print(f"📝 Saved {len(all_titles)} titles for observation to: {titles_file}")
        except Exception as e:
            print(f"⚠️  Error saving titles: {e}")

        # Create mapping from title back to item ID for validation checking
        title_to_item_id = {}
        for item in random_items_with_titles:
            if self.dataset == 'douban':
                title_to_item_id[item['title']] = item['doubanID']
            elif self.dataset == 'redbook':
                title_to_item_id[item['title']] = item['redbookID']
            elif self.dataset == 'hupu':
                title_to_item_id[item['title']] = item['hupuID']
            else:
                title_to_item_id[item['title']] = item['bvid']

        # For validation items, map title to item ID
        for item_id, title in validation_item_to_title.items():
            title_to_item_id[title] = item_id

        current_profile = user_profile
        valid_records = []  # Store test results for each reflection round

        print(f"🧪 测试策略: {'每轮都测试' if test_each_round else '仅最后一轮测试'}")

        for reflection_round in range(max_reflections):
            print(f"🔄 Self-reflection round {reflection_round + 1}/{max_reflections}")

            # Create prediction prompt with titles
            prediction_prompt = f"""
Based on the following user profile, predict the top 20 items that this user is most likely to interact with from the given available items. Rank them from most likely (1) to least likely (20).

User Profile:
{current_profile}

Available Items (including titles and tags):
{', '.join(all_titles)}

Please provide your top 20 predictions in order of likelihood (use the exact titles from the list, do not add any asterisks or bold formatting in your response):
1.
2.
3.
4.
5.
6.
7.
8.
9.
10.
11.
12.
13.
14.
15.
16.
17.
18.
19.
20.
"""

            # Get LLM prediction (using UserProfileGenerator as LLM interface)
            try:
                prediction_response = self.user_profile_generator.client.chat.completions.create(
                    model=os.getenv("CHAT_MODEL"),
                    messages=[
                        {"role": "system", "content": "Please recommend some items to users based on user profile and available item titles. Always use the exact titles provided in the list."},
                        {"role": "user", "content": prediction_prompt}
                    ],
                    temperature=1
                )

                predicted_titles = self._extract_predictions(prediction_response.choices[0].message.content)
                # Predictions generated (no detailed output)

                # Convert predicted titles back to item IDs and check for validation hits
                predicted_item_ids = []
                for title in predicted_titles:
                    item_id = title_to_item_id.get(title, title)  # Fallback to title if not found
                    predicted_item_ids.append(item_id)

                # Check if any validation item is in top 5 predictions
                hit_items = [item_id for item_id in predicted_item_ids if item_id in validation_item_names]

                # Record test results based on test_each_round parameter
                if test_each_round or reflection_round == max_reflections - 1:
                    # Calculate validation metrics
                    validation_metrics = self._calculate_test_metrics(predicted_item_ids, validation_item_names)

                    # Record test results for this round
                    valid_record = {
                        "round": reflection_round + 1,
                        "user_profile": current_profile,
                        "predicted_titles": predicted_titles,
                        "predicted_item_ids": predicted_item_ids,
                        "validation_items": validation_item_names,
                        "hit_items": hit_items,
                        "hit_count": len(hit_items),
                        "hit_success": len(hit_items) > 0,
                        "ndcg_at_1": validation_metrics["ndcg_at_1"],
                        "ndcg_at_5": validation_metrics["ndcg_at_5"],
                        "ndcg_at_10": validation_metrics["ndcg_at_10"],
                        "ndcg_at_20": validation_metrics["ndcg_at_20"],
                        "hit_rate_at_1": validation_metrics["hit_rate_at_1"],
                        "hit_rate_at_5": validation_metrics["hit_rate_at_5"],
                        "hit_rate_at_10": validation_metrics["hit_rate_at_10"],
                        "hit_rate_at_20": validation_metrics["hit_rate_at_20"],
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "is_final_round": reflection_round == max_reflections - 1
                    }
                    valid_records.append(valid_record)

                    # Validation record saved (no detailed output)

                if hit_items:
                    print(f"✅ Validation hit")
                    return True, current_profile, valid_records

                if test_each_round:
                    print(f"❌ No hits in round {reflection_round + 1}")
                elif reflection_round == max_reflections - 1:
                    print(f"❌ No hits in final round {reflection_round + 1}")

                # If not the last round, enhance profile using validation item analysis
                if reflection_round < max_reflections - 1:
                    enhanced_profile = self._enhance_user_profile(current_profile, validation_item_names, user_real_id)
                    if enhanced_profile == "NO_VALIDATION_ANALYSIS_FOUND":
                        print(f"❌ No validation analysis found for user {user_real_id}, stopping reflection process")
                        return False, current_profile, valid_records
                    current_profile = enhanced_profile
                    print(f"🔄 Enhanced user profile for next round")

            except Exception as e:
                print(f"⚠️  Error in prediction round {reflection_round + 1}: {e}")
                break

        print(f"❌ Self-reflection failed after {max_reflections} rounds")
        return False, current_profile, valid_records

    def _extract_predictions(self, response_text: str) -> List[str]:
        """Extract top 20 predictions from LLM response"""
        predictions = []
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            # Look for numbered items (1. item_name, 2. item_name, etc.)
            if line and any(line.startswith(f"{i}.") for i in range(1, 21)):
                # Extract item name after the number
                item_name = line.split('.', 1)[1].strip()
                if item_name:  # Only add non-empty predictions
                    predictions.append(item_name)

                if len(predictions) >= 20:
                    break

        return predictions

    def _enhance_user_profile(self, current_profile: str, validation_item_names: List[str], user_real_id: str) -> str:
        """Enhance user profile using validation item analysis"""
        # Get validation item analysis (this should be available from previous download/analysis)
        validation_analysis = self._get_validation_item_analysis(validation_item_names, user_real_id)

        # Check if no validation analysis was found
        if validation_analysis == "NO_VALIDATION_ANALYSIS_FOUND":
            print(f"❌ No validation analysis found for user {user_real_id}, skipping profile enhancement")
            return "NO_VALIDATION_ANALYSIS_FOUND"

        # Get historical and recommended item profiles for context
        historical_profiles = self._get_historical_item_profiles(user_real_id)
        recommended_profiles = self._get_recommended_item_profiles(user_real_id)

        enhancement_prompt = f"""
Based on the current user profile, historical interaction patterns, recommended items, and the analysis of the new interaction items that the user actually engaged with,
please generate an enhanced user profile that better captures the user's preferences.

Current User Profile:
{current_profile}

Historical Item Profiles (in chronological order):
{historical_profiles}

Recommended Item Profiles:
{recommended_profiles}

New Interaction Items Analysis:
{validation_analysis}

Requirements:
1. Do not mention the specific item names in the enhanced profile
2. Consider the user's historical interaction patterns to understand preference evolution
3. Use recommended items to understand what content the system thought the user might like
4. If the current user profile already contains preferences that align with these new interaction items, strengthen and emphasize those preferences
5. If the current user profile lacks preferences that would explain the user's interaction with these items, add new preference patterns based on the analysis
6. Extract deeper insights about user preferences from the new interaction items
7. Keep the profile concise and focused on key preference themes
8. Maintain consistency with existing preferences while expanding or refining them
9. Consider how the new interactions relate to both historical patterns and recommended content

Enhanced User Profile:
"""

        try:
            response = self.user_profile_generator.client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                messages=[
                    {"role": "system", "content": "Please analyze user behavior and model user preferences."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️  Error enhancing user profile: {e}")
            return current_profile

    def _get_validation_item_analysis(self, validation_item_names: List[str], user_real_id: str) -> str:
        """Get analysis text for validation items"""
        analysis_parts = []
        found_any_analysis = False

        for item_name in validation_item_names:
            # Look for analysis file in validation folder
            # Ensure item_name is string for path joining
            item_name_str = str(item_name)
            validation_dir = os.path.join("download", self.dataset, user_real_id, "validation", item_name_str)
            analysis_file = os.path.join(validation_dir, "analysis.json")

            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)

                    # Format analysis data into text
                    analysis_text = f"Item {item_name}:\n"

                    if "video" in analysis_data:
                        for idx, video_analysis in analysis_data["video"].items():
                            analysis_text += f"- Video: {video_analysis}\n"

                    if "image" in analysis_data:
                        for idx, image_analysis in analysis_data["image"].items():
                            analysis_text += f"- Image: {image_analysis}\n"

                    if "text" in analysis_data:
                        for idx, text_analysis in analysis_data["text"].items():
                            analysis_text += f"- Text: {text_analysis}\n"

                    analysis_parts.append(analysis_text)
                    found_any_analysis = True

                except Exception as e:
                    print(f"⚠️  Error reading analysis for {item_name}: {e}")

        if not found_any_analysis:
            # Return special value to indicate no analysis found
            return "NO_VALIDATION_ANALYSIS_FOUND"

        return "\n\n".join(analysis_parts) if analysis_parts else "No validation item analysis available."

    def _get_chronological_order_direct(self, user_real_id: str) -> List[str]:
        """Get chronological order of items directly without user ID conversion"""
        user_dataset_path = f"dataset/{self.dataset}/{user_real_id}"

        if not os.path.exists(user_dataset_path):
            print(f"⚠️  User dataset path not found: {user_dataset_path}")
            return []

        # Dataset-specific column mapping
        if self.dataset == 'bilibili':
            item_id_col = 'bvid'
            time_col = 'fav_time'
        elif self.dataset == 'douban':
            item_id_col = 'doubanID'  # Use doubanID as unique identifier for douban
            time_col = 'fav_time'
        elif self.dataset == 'redbook':
            item_id_col = 'redbookID'  # Use redbookID as unique identifier for redbook
            time_col = 'fav_time'
        elif self.dataset == 'hupu':
            item_id_col = 'hupuID'  # Use hupuID as unique identifier for hupu
            time_col = 'fav_time'
        else:
            item_id_col = 'note_id'  # Default fallback
            time_col = 'time'

        all_items = []

        # Read all CSV files and collect items with timestamps
        for file in os.listdir(user_dataset_path):
            if file.endswith('.csv'):
                try:
                    csv_path = os.path.join(user_dataset_path, file)
                    df = pd.read_csv(csv_path)

                    if item_id_col in df.columns and time_col in df.columns:
                        for _, row in df.iterrows():
                            all_items.append({
                                'item_id': row[item_id_col],
                                'timestamp': row[time_col]
                            })
                except Exception as e:
                    print(f"⚠️  Error reading {file}: {e}")
                    continue

        if not all_items:
            return []

        # Sort by timestamp and return item IDs
        try:
            # Convert timestamps to pandas datetime for proper sorting
            for item in all_items:
                if isinstance(item['timestamp'], str):
                    # Handle different date formats based on dataset
                    try:
                        # For hupu dataset, handle YYYYMMDDHHMMSS format
                        if self.dataset == 'hupu' and len(item['timestamp']) == 14 and item['timestamp'].isdigit():
                            # Parse YYYYMMDDHHMMSS format
                            timestamp_str = item['timestamp']
                            year = timestamp_str[:4]
                            month = timestamp_str[4:6]
                            day = timestamp_str[6:8]
                            hour = timestamp_str[8:10]
                            minute = timestamp_str[10:12]
                            second = timestamp_str[12:14]
                            formatted_timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                            item['timestamp'] = pd.to_datetime(formatted_timestamp)
                        else:
                            # Handle other date formats
                            item['timestamp'] = pd.to_datetime(item['timestamp'])
                    except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                        # Handle invalid or out-of-bounds timestamps
                        print(f"⚠️  Invalid timestamp '{item['timestamp']}' for item {item['item_id']}: {e}")
                        # Use a default timestamp (current time)
                        item['timestamp'] = pd.Timestamp.now()
                elif isinstance(item['timestamp'], (int, float)):
                    # Handle numeric timestamps
                    try:
                        # For hupu dataset, handle YYYYMMDDHHMMSS format as integer
                        if self.dataset == 'hupu' and len(str(int(item['timestamp']))) == 14:
                            # Parse YYYYMMDDHHMMSS format
                            timestamp_str = str(int(item['timestamp']))
                            year = timestamp_str[:4]
                            month = timestamp_str[4:6]
                            day = timestamp_str[6:8]
                            hour = timestamp_str[8:10]
                            minute = timestamp_str[10:12]
                            second = timestamp_str[12:14]
                            formatted_timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                            item['timestamp'] = pd.to_datetime(formatted_timestamp)
                        else:
                            # Handle Unix timestamps
                            # Check if timestamp is reasonable (between 1970 and 2100)
                            if 0 <= item['timestamp'] <= 4102444800:  # 2100-01-01
                                item['timestamp'] = pd.to_datetime(item['timestamp'], unit='s')
                            else:
                                print(f"⚠️  Out-of-range timestamp {item['timestamp']} for item {item['item_id']}")
                                item['timestamp'] = pd.Timestamp.now()
                    except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                        print(f"⚠️  Invalid timestamp {item['timestamp']} for item {item['item_id']}: {e}")
                        item['timestamp'] = pd.Timestamp.now()

            # Sort by timestamp in descending order (newest to oldest)
            sorted_items = sorted(all_items, key=lambda x: x['timestamp'], reverse=True)
            chronological_order = [item['item_id'] for item in sorted_items]

            print(f"📅 Found {len(chronological_order)} items in reverse chronological order (newest to oldest) for user {user_real_id}")
            return chronological_order

        except Exception as e:
            print(f"⚠️  Error sorting items chronologically: {e}")
            # Fallback: return items without sorting
            return [item['item_id'] for item in all_items]

    def _get_historical_item_profiles(self, user_real_id: str) -> str:
        """Get analysis text for historical items in chronological order"""
        historical_dir = os.path.join("download", self.dataset, user_real_id, "historical")

        if not os.path.exists(historical_dir):
            return "No historical item profiles available."

        # Get chronological order directly since folder name is already real user id
        chronological_order = self._get_chronological_order_direct(user_real_id)

        analysis_parts = []
        processed_items = set()

        # Process items in chronological order
        for item_name in chronological_order:
            if item_name in processed_items:
                continue

            # Ensure item_name is string for path joining
            item_name_str = str(item_name)
            item_dir = os.path.join(historical_dir, item_name_str)
            analysis_file = os.path.join(item_dir, "analysis.json")

            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)

                    # Format analysis data into text
                    analysis_text = f"Historical Item {item_name}:\n"

                    if "video" in analysis_data:
                        for idx, video_analysis in analysis_data["video"].items():
                            analysis_text += f"- Video: {video_analysis}\n"

                    if "image" in analysis_data:
                        for idx, image_analysis in analysis_data["image"].items():
                            analysis_text += f"- Image: {image_analysis}\n"

                    if "text" in analysis_data:
                        for idx, text_analysis in analysis_data["text"].items():
                            analysis_text += f"- Text: {text_analysis}\n"

                    analysis_parts.append(analysis_text)
                    processed_items.add(item_name)

                except Exception as e:
                    print(f"⚠️  Error reading historical analysis for {item_name}: {e}")

        # Process any remaining historical items not in chronological order
        for item_name in os.listdir(historical_dir):
            if item_name not in processed_items and os.path.isdir(os.path.join(historical_dir, item_name)):
                item_dir = os.path.join(historical_dir, item_name)
                analysis_file = os.path.join(item_dir, "analysis.json")

                if os.path.exists(analysis_file):
                    try:
                        with open(analysis_file, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)

                        # Format analysis data into text
                        analysis_text = f"Historical Item {item_name}:\n"

                        if "video" in analysis_data:
                            for idx, video_analysis in analysis_data["video"].items():
                                analysis_text += f"- Video: {video_analysis}\n"

                        if "image" in analysis_data:
                            for idx, image_analysis in analysis_data["image"].items():
                                analysis_text += f"- Image: {image_analysis}\n"

                        if "text" in analysis_data:
                            for idx, text_analysis in analysis_data["text"].items():
                                analysis_text += f"- Text: {text_analysis}\n"

                        analysis_parts.append(analysis_text)

                    except Exception as e:
                        print(f"⚠️  Error reading historical analysis for {item_name}: {e}")

        return "\n\n".join(analysis_parts) if analysis_parts else "No historical item profiles available."

    def _get_recommended_item_profiles(self, user_real_id: str) -> str:
        """Get analysis text for recommended items"""
        recommended_dir = os.path.join("download", self.dataset, user_real_id, "recommended")

        if not os.path.exists(recommended_dir):
            return "No recommended item profiles available."

        analysis_parts = []

        for item_name in os.listdir(recommended_dir):
            if not os.path.isdir(os.path.join(recommended_dir, item_name)):
                continue

            item_dir = os.path.join(recommended_dir, item_name)
            analysis_file = os.path.join(item_dir, "analysis.json")

            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)

                    # Format analysis data into text
                    analysis_text = f"Recommended Item {item_name}:\n"

                    if "video" in analysis_data:
                        for idx, video_analysis in analysis_data["video"].items():
                            analysis_text += f"- Video: {video_analysis}\n"

                    if "image" in analysis_data:
                        for idx, image_analysis in analysis_data["image"].items():
                            analysis_text += f"- Image: {image_analysis}\n"

                    if "text" in analysis_data:
                        for idx, text_analysis in analysis_data["text"].items():
                            analysis_text += f"- Text: {text_analysis}\n"

                    analysis_parts.append(analysis_text)

                except Exception as e:
                    print(f"⚠️  Error reading recommended analysis for {item_name}: {e}")

        return "\n\n".join(analysis_parts) if analysis_parts else "No recommended item profiles available."

    def _analyze_consistency(self, llm_hits: List[str], rec_hits: List[str], test_items: List[str], llm_predictions: List[str]) -> Dict:
        """Analyze consistency between LLM predictions and recommendation system"""

        # Convert to sets for easier comparison
        llm_hits_set = set(llm_hits)
        rec_hits_set = set(rec_hits)
        test_items_set = set(test_items)

        # Calculate various metrics
        common_hits = llm_hits_set.intersection(rec_hits_set)
        llm_only_hits = llm_hits_set - rec_hits_set
        rec_only_hits = rec_hits_set - llm_hits_set

        # Calculate hit rates
        llm_hit_rate = len(llm_hits) / len(llm_predictions) if llm_predictions else 0
        rec_hit_rate = len(rec_hits) / len(rec_hits) if rec_hits else 0  # This needs total recommendations

        # Consistency score: how much overlap there is
        total_unique_hits = len(llm_hits_set.union(rec_hits_set))
        consistency_score = len(common_hits) / total_unique_hits if total_unique_hits > 0 else 0

        analysis = {
            "common_hits": list(common_hits),
            "common_hits_count": len(common_hits),
            "llm_only_hits": list(llm_only_hits),
            "llm_only_hits_count": len(llm_only_hits),
            "rec_only_hits": list(rec_only_hits),
            "rec_only_hits_count": len(rec_only_hits),
            "llm_hit_rate": llm_hit_rate,
            "consistency_score": consistency_score,
            "total_test_items": len(test_items),
            "analysis_summary": self._generate_consistency_summary(
                common_hits, llm_only_hits, rec_only_hits, consistency_score
            )
        }

        return analysis

    def _generate_consistency_summary(self, common_hits: set, llm_only_hits: set, rec_only_hits: set, consistency_score: float) -> str:
        """Generate a human-readable summary of consistency analysis"""

        summary_parts = []

        if consistency_score >= 0.7:
            summary_parts.append("🟢 HIGH CONSISTENCY: LLM predictions align well with recommendation system")
        elif consistency_score >= 0.4:
            summary_parts.append("🟡 MODERATE CONSISTENCY: Some alignment between LLM and recommendation system")
        else:
            summary_parts.append("🔴 LOW CONSISTENCY: Significant differences between LLM and recommendation system")

        if common_hits:
            summary_parts.append(f"✅ Both systems correctly identified: {list(common_hits)}")

        if llm_only_hits:
            summary_parts.append(f"🤖 LLM found additional relevant items: {list(llm_only_hits)}")

        if rec_only_hits:
            summary_parts.append(f"🔧 Recommendation system found items LLM missed: {list(rec_only_hits)}")

        if not common_hits and not llm_only_hits and not rec_only_hits:
            summary_parts.append("❌ Neither system successfully identified test items")

        return " | ".join(summary_parts)

    def perform_iterative_reflection(self, user_real_id: str, user_internal_id: int, initial_user_profile: str, validation_items: List[str], test_items: List[str], recommended_items: List[str], max_rounds: int = 3):
        """
        Perform iterative reflection process:

        Round 0: Initial test with original profile
        - If test hits → End process
        - If test misses → Continue to Round 1+

        Round 1+: Validation reflection + test
        - Perform validation reflection to enhance profile
        - If validation hits → Always perform test immediately → End process
        - If validation misses → Perform test based on mode (test_each_round)

        Mode control only applies when validation misses:
        - test_each_round=True: Test after each validation miss
        - test_each_round=False: Test only at the very end
        """
        user_dir = os.path.join("download", self.dataset, user_real_id)

        test_mode = "每轮测试" if self.test_each_round else "仅最后一轮测试"
        print(f"🧪 测试模式: {test_mode}")

        # Prepare random items (same set used for all rounds)
        # Convert user_real_id to user_internal_id for proper exclusion
        user_internal_id_for_exclusion = None
        if hasattr(self, 'user_map') and self.user_map:
            for internal_id, real_id in self.user_map.items():
                if real_id == user_real_id:
                    user_internal_id_for_exclusion = int(internal_id)
                    break

        random_items_with_titles = self.get_random_items_with_titles(user_real_id, recommended_items, user_internal_id=user_internal_id_for_exclusion)
        print(f"🎲 Selected {len(random_items_with_titles)} random items for reflection process")

        # Save random items, validation and test data to JSON
        self._save_reflection_data(user_real_id, random_items_with_titles, validation_items, test_items, recommended_items)

        # Storage for all results
        all_valid_records = []
        all_test_records = []
        current_profile = initial_user_profile

        # Save initial profile as Round 0
        self._save_initial_profile(initial_user_profile, user_real_id)

        # === Round 0: Initial test with original profile ===
        print(f"\n🎯 Round 0: Initial test with original user profile")
        round_0_hit = self._perform_single_test(
            current_profile, test_items, random_items_with_titles, user_real_id, 0, "test"
        )

        if round_0_hit:
            print(f"✅ Round 0 test hit! Ending reflection process.")
            # Save final results
            self._save_final_results(user_dir, all_valid_records, all_test_records, initial_user_profile, current_profile, True, 0)
            return

        # === Rounds 1+: Validation reflection + optional test ===
        for round_num in range(1, max_rounds + 1):
            print(f"\n🔄 Round {round_num}: Validation reflection")

            # Perform single round validation reflection
            valid_hit_success, enhanced_profile = self._perform_single_validation_round(
                current_profile,
                validation_items,
                random_items_with_titles,
                user_real_id,
                round_num
            )

            # Update profile if enhanced
            if enhanced_profile != current_profile:
                current_profile = enhanced_profile
                print(f"📈 User profile enhanced in round {round_num}")

            if valid_hit_success:
                print(f"✅ Round {round_num} validation hit! Ending validation reflection.")

                # Always perform test immediately when validation hits
                print(f"🧪 Performing Round {round_num} test (validation hit)")
                test_hit = self._perform_single_test(
                    current_profile, test_items, random_items_with_titles, user_real_id, round_num, "test"
                )
                if test_hit:
                    print(f"✅ Round {round_num} test hit! Process completed successfully.")
                    self._save_final_results(user_dir, all_valid_records, all_test_records, initial_user_profile, current_profile, True, round_num)
                    return
                else:
                    print(f"❌ Round {round_num} test miss. Process completed.")
                    self._save_final_results(user_dir, all_valid_records, all_test_records, initial_user_profile, current_profile, False, round_num)
                    return

            else:
                print(f"❌ Round {round_num} validation miss.")

                # Decide whether to perform test based on mode
                if self.test_each_round:
                    print(f"🧪 Performing Round {round_num} test (mode: 每轮测试)")
                    test_hit = self._perform_single_test(
                        current_profile, test_items, random_items_with_titles, user_real_id, round_num, "test"
                    )
                    if test_hit:
                        print(f"✅ Round {round_num} test hit! Process completed successfully.")
                        self._save_final_results(user_dir, all_valid_records, all_test_records, initial_user_profile, current_profile, True, round_num)
                        return

        # If we reach here, all rounds completed without success
        print(f"❌ All {max_rounds} rounds completed without success.")

        # Perform final test if mode is "仅最后一轮测试" and we haven't tested in the last round
        if not self.test_each_round:
            print(f"🧪 Performing final test (mode: 仅最后一轮测试)")
            final_test_hit = self._perform_single_test(
                current_profile, test_items, random_items_with_titles, user_real_id, max_rounds, "test"
            )
            self._save_final_results(user_dir, all_valid_records, all_test_records, initial_user_profile, current_profile, final_test_hit, max_rounds)
        else:
            self._save_final_results(user_dir, all_valid_records, all_test_records, initial_user_profile, current_profile, False, max_rounds)

    def _perform_single_test(self, user_profile: str, target_items: List[str], random_items_with_titles: List[Dict[str, str]], user_real_id: str, round_num: int, test_type: str) -> bool:
        """Perform a single test and return whether it hit"""

        # Create list of all items with titles and tags for prediction
        item_mapping = self._get_cached_item_mapping()
        all_items_with_info = []
        title_to_item_id = {}

        # Add target items
        for item_id in target_items:
            title, tags = self._extract_title_and_tags_from_mapping(item_id, item_mapping, item_id)
            all_items_with_info.append({'id': item_id, 'title': title, 'tags': tags})
            title_to_item_id[title] = item_id

        # Add random items
        for item in random_items_with_titles:
            if self.dataset == 'douban':
                item_id = item['doubanID']
                title_to_item_id[item['title']] = item_id
            elif self.dataset == 'redbook':
                item_id = item['redbookID']
                title_to_item_id[item['title']] = item_id
            elif self.dataset == 'hupu':
                item_id = item['hupuID']
                title_to_item_id[item['title']] = item_id
            else:
                item_id = item['bvid']
                title_to_item_id[item['title']] = item_id

            title, tags = self._extract_title_and_tags_from_mapping(item_id, item_mapping, item['title'])
            all_items_with_info.append({'id': item_id, 'title': title, 'tags': tags})

        # Shuffle the items
        random.shuffle(all_items_with_info)

        # Create formatted item list with numbers, titles, and tags
        formatted_items = []
        for i, item in enumerate(all_items_with_info, 1):
            formatted_item = f"{i}. title: {item['title']}"
            if item['tags']:
                formatted_item += f"\ntags: {item['tags']}"
            formatted_items.append(formatted_item)

        # Create prediction prompt
        prediction_prompt = f"""
Based on the following user profile, predict the top 20 items that this user is most likely to interact with from the given list. Rank them from most likely (1) to least likely (20).

User Profile:
{user_profile}

Available Items:
{chr(10).join(formatted_items)}

Please provide your top 20 predictions in order of likelihood (use the exact titles from the list, do not add any asterisks or bold formatting in your response):
1.
2.
3.
4.
5.
6.
7.
8.
9.
10.
11.
12.
13.
14.
15.
16.
17.
18.
19.
20.
"""

        try:
            # Get LLM prediction
            response = self.user_profile_generator.client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                messages=[
                    {"role": "system", "content": "Please recommend some items to users based on user profile and available item titles. Always use the exact titles provided in the list."},
                    {"role": "user", "content": prediction_prompt}
                ],
                temperature=1
            )

            predicted_titles = self._extract_predictions(response.choices[0].message.content)
            print(f"🎯 Round {round_num} {test_type} predicted titles: {predicted_titles}")

            # Convert predicted titles back to item IDs
            predicted_item_ids = []
            for title in predicted_titles:
                item_id = title_to_item_id.get(title, title)
                predicted_item_ids.append(item_id)

            # Calculate comprehensive test metrics
            test_metrics = self._calculate_test_metrics(predicted_item_ids, target_items)
            hit_items = [item_id for item_id in predicted_item_ids if item_id in target_items]
            hit_success = test_metrics["hit_success"]

            # Record test result with enhanced metrics
            test_record = {
                "round": round_num,
                "test_type": test_type,
                "user_profile": user_profile,
                "predicted_titles": predicted_titles,
                "predicted_item_ids": predicted_item_ids,
                "target_items": target_items,
                "hit_items": hit_items,
                "hit_count": test_metrics["hit_count"],
                "hit_success": hit_success,
                "ndcg_at_1": test_metrics["ndcg_at_1"],
                "ndcg_at_5": test_metrics["ndcg_at_5"],
                "ndcg_at_10": test_metrics["ndcg_at_10"],
                "ndcg_at_20": test_metrics["ndcg_at_20"],
                "hit_rate_at_1": test_metrics["hit_rate_at_1"],
                "hit_rate_at_5": test_metrics["hit_rate_at_5"],
                "hit_rate_at_10": test_metrics["hit_rate_at_10"],
                "hit_rate_at_20": test_metrics["hit_rate_at_20"],
                "timestamp": pd.Timestamp.now().isoformat()
            }

            # Store test record globally (will be saved later)
            if not hasattr(self, '_current_test_records'):
                self._current_test_records = []
            self._current_test_records.append(test_record)

            if hit_success:
                print(f"✅ Round {round_num} {test_type} hit")
            else:
                print(f"❌ Round {round_num} {test_type} miss")

            return hit_success

        except Exception as e:
            print(f"⚠️  Error in Round {round_num} {test_type} test: {e}")
            return False

    def _save_final_results(self, user_dir: str, valid_records: List[Dict], test_records: List[Dict], initial_profile: str, final_profile: str, success: bool, final_round: int):
        """Save all final results to files"""

        # Add any remaining validation records
        if hasattr(self, '_current_valid_records'):
            valid_records.extend(self._current_valid_records)
            delattr(self, '_current_valid_records')

        # Add any remaining test records
        if hasattr(self, '_current_test_records'):
            test_records.extend(self._current_test_records)
            delattr(self, '_current_test_records')

        # Calculate statistics for saving to files (but don't display them)
        stats = None
        if valid_records or test_records:
            stats = self._calculate_reflection_statistics(valid_records, test_records)

        # Save validation reflection results
        if valid_records:
            validation_results = {
                "final_round": final_round,
                "success": success,
                "initial_profile": initial_profile,
                "final_profile": final_profile,
                "validation_records": valid_records,
                "test_each_round": self.test_each_round,
                "test_mode": "每轮测试" if self.test_each_round else "仅最后一轮测试",
                "reflection_type": "validation",
                "statistics": stats if 'stats' in locals() else None
            }

            valid_file = os.path.join(user_dir, "valid_reflection_results.json")
            with open(valid_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2)
            print(f"✅ Validation reflection results saved to: {valid_file}")

        # Save test reflection results
        if test_records:
            test_results = {
                "final_round": final_round,
                "success": success,
                "initial_profile": initial_profile,
                "final_profile": final_profile,
                "test_records": test_records,
                "test_each_round": self.test_each_round,
                "test_mode": "每轮测试" if self.test_each_round else "仅最后一轮测试",
                "reflection_type": "test",
                "statistics": stats if 'stats' in locals() else None
            }

            test_file = os.path.join(user_dir, "test_reflection_results.json")
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"✅ Test reflection results saved to: {test_file}")

        # Enhanced profiles are already saved in each round, no need to save again here

        print(f"🎉 Iterative reflection process completed: {'Success' if success else 'Failed'}")

    def _perform_single_validation_round(self, user_profile: str, validation_items: List[str], random_items_with_titles: List[Dict[str, str]], user_real_id: str, round_num: int) -> Tuple[bool, str]:
        """Perform a single round of validation reflection"""

        # Create list of all items with titles and tags for prediction
        item_mapping = self._get_cached_item_mapping()
        all_items_with_info = []
        title_to_item_id = {}

        # Add validation items
        for item_id in validation_items:
            title, tags = self._extract_title_and_tags_from_mapping(item_id, item_mapping, item_id)
            all_items_with_info.append({'id': item_id, 'title': title, 'tags': tags})
            title_to_item_id[title] = item_id

        # Add random items
        for item in random_items_with_titles:
            if self.dataset == 'douban':
                item_id = item['doubanID']
                title_to_item_id[item['title']] = item_id
            elif self.dataset == 'redbook':
                item_id = item['redbookID']
                title_to_item_id[item['title']] = item_id
            elif self.dataset == 'hupu':
                item_id = item['hupuID']
                title_to_item_id[item['title']] = item_id
            else:
                item_id = item['bvid']
                title_to_item_id[item['title']] = item_id

            title, tags = self._extract_title_and_tags_from_mapping(item_id, item_mapping, item['title'])
            all_items_with_info.append({'id': item_id, 'title': title, 'tags': tags})

        # Shuffle the items
        random.shuffle(all_items_with_info)

        # Create formatted item list with numbers, titles, and tags
        formatted_items = []
        for i, item in enumerate(all_items_with_info, 1):
            formatted_item = f"{i}. title: {item['title']}"
            if item['tags']:
                formatted_item += f"\ntags: {item['tags']}"
            formatted_items.append(formatted_item)

        # Create prediction prompt with titles and tags
        prediction_prompt = f"""
Based on the following user profile, predict the top 20 items that this user is most likely to interact with from the given available items. Rank them from most likely (1) to least likely (20).

User Profile:
{user_profile}

Available Items (including titles and tags):
{chr(10).join(formatted_items)}

Please provide your top 20 predictions in order of likelihood (use the exact titles from the list, do not add any asterisks or bold formatting in your response):
1.
2.
3.
4.
5.
6.
7.
8.
9.
10.
11.
12.
13.
14.
15.
16.
17.
18.
19.
20.
"""

        try:
            # Get LLM prediction
            response = self.user_profile_generator.client.chat.completions.create(
                model=os.getenv("CHAT_MODEL"),
                messages=[
                    {"role": "system", "content": "Please recommend some items to users based on user profile and available item titles. Always use the exact titles provided in the list."},
                    {"role": "user", "content": prediction_prompt}
                ],
                temperature=1
            )

            predicted_titles = self._extract_predictions(response.choices[0].message.content)
            print(f"🎯 Round {round_num} validation predicted titles: {predicted_titles}")

            # Convert predicted titles back to item IDs and check for validation hits
            predicted_item_ids = []
            for title in predicted_titles:
                item_id = title_to_item_id.get(title, title)  # Fallback to title if not found
                predicted_item_ids.append(item_id)

            # Check if any validation item is in top 5 predictions
            hit_items = [item_id for item_id in predicted_item_ids if item_id in validation_items]
            hit_success = len(hit_items) > 0

            # Calculate validation metrics (same as test metrics)
            validation_metrics = self._calculate_test_metrics(predicted_item_ids, validation_items)

            # Record validation result with NDCG metrics
            validation_record = {
                "round": round_num,
                "test_type": "validation",
                "user_profile": user_profile,
                "predicted_titles": predicted_titles,
                "predicted_item_ids": predicted_item_ids,
                "validation_items": validation_items,
                "hit_items": hit_items,
                "hit_count": len(hit_items),
                "hit_success": hit_success,
                "ndcg_at_1": validation_metrics["ndcg_at_1"],
                "ndcg_at_5": validation_metrics["ndcg_at_5"],
                "ndcg_at_10": validation_metrics["ndcg_at_10"],
                "ndcg_at_20": validation_metrics["ndcg_at_20"],
                "hit_rate_at_1": validation_metrics["hit_rate_at_1"],
                "hit_rate_at_5": validation_metrics["hit_rate_at_5"],
                "hit_rate_at_10": validation_metrics["hit_rate_at_10"],
                "hit_rate_at_20": validation_metrics["hit_rate_at_20"],
                "timestamp": pd.Timestamp.now().isoformat()
            }

            # Store validation record globally (will be saved later)
            if not hasattr(self, '_current_valid_records'):
                self._current_valid_records = []
            self._current_valid_records.append(validation_record)

            if hit_success:
                print(f"✅ Round {round_num} validation hit")

                # Enhance profile using validation item analysis
                enhanced_profile = self._enhance_user_profile(user_profile, validation_items, user_real_id)
                if enhanced_profile == "NO_VALIDATION_ANALYSIS_FOUND":
                    print(f"❌ No validation analysis found for user {user_real_id}, cannot enhance profile")
                    return False, user_profile  # Return original profile

                # Save enhanced profile for this round
                self._save_enhanced_profile(enhanced_profile, user_real_id, round_num, hit_success)

                return True, enhanced_profile
            else:
                print(f"❌ Round {round_num} validation miss")

                # Enhance profile for next round
                enhanced_profile = self._enhance_user_profile(user_profile, validation_items, user_real_id)
                if enhanced_profile == "NO_VALIDATION_ANALYSIS_FOUND":
                    print(f"❌ No validation analysis found for user {user_real_id}, cannot enhance profile")
                    return False, user_profile  # Return original profile

                # Save enhanced profile for this round
                self._save_enhanced_profile(enhanced_profile, user_real_id, round_num, hit_success)

                return False, enhanced_profile

        except Exception as e:
            print(f"⚠️  Error in Round {round_num} validation: {e}")
            return False, user_profile

    def _save_enhanced_profile(self, enhanced_profile: str, user_real_id: str, round_num: int, hit_success: bool):
        """Save enhanced user profile for each round"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Create enhanced_user_profiles directory if it doesn't exist
        enhanced_profiles_dir = os.path.join(user_dir, "enhanced_user_profiles")
        os.makedirs(enhanced_profiles_dir, exist_ok=True)

        # Create filename with round number and hit status
        hit_status = "hit" if hit_success else "miss"
        profile_filename = f"round_{round_num}_{hit_status}_profile.txt"
        profile_path = os.path.join(enhanced_profiles_dir, profile_filename)

        # Save the enhanced profile
        with open(profile_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_profile)

        print(f"💾 Enhanced profile saved: {profile_filename}")

        # Also update the main enhanced_user_profile.txt with the latest version
        main_profile_path = os.path.join(user_dir, "enhanced_user_profile.txt")
        with open(main_profile_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_profile)

    def _save_initial_profile(self, initial_profile: str, user_real_id: str):
        """Save initial user profile as Round 0"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Create enhanced_user_profiles directory if it doesn't exist
        enhanced_profiles_dir = os.path.join(user_dir, "enhanced_user_profiles")
        os.makedirs(enhanced_profiles_dir, exist_ok=True)

        # Save initial profile as round 0
        profile_filename = "round_0_initial_profile.txt"
        profile_path = os.path.join(enhanced_profiles_dir, profile_filename)

        with open(profile_path, 'w', encoding='utf-8') as f:
            f.write(initial_profile)

        print(f"💾 Initial profile saved: {profile_filename}")

    def _get_items_with_titles(self, item_ids: List[str]) -> List[Dict[str, str]]:
        """Get items with titles for validation, test, or recommended items"""
        items_with_titles = []

        if self.dataset == 'douban':
            # For douban, load mapping to get titles
            item_mapping = self._get_cached_item_mapping()

            for item_id in item_ids:
                if item_mapping:
                    title = self._extract_title_from_mapping(item_id, item_mapping, f"Item_{item_id}")
                else:
                    # Fallback: try to find title from dataset
                    title = self._get_douban_title_by_id(item_id)
                    if not title:
                        title = f"Item_{item_id}"

                items_with_titles.append({
                    "doubanID": item_id,
                    "title": title
                })
        elif self.dataset == 'redbook':
            # For redbook, load mapping to get titles
            item_mapping = self._get_cached_item_mapping()

            for item_id in item_ids:
                if item_mapping:
                    title = self._extract_title_from_mapping(item_id, item_mapping, f"Item_{item_id}")
                else:
                    # Fallback: try to find title from dataset
                    title = self._get_redbook_title_by_id(item_id)
                    if not title:
                        title = f"Item_{item_id}"

                items_with_titles.append({
                    "redbookID": item_id,
                    "title": title
                })
        elif self.dataset == 'hupu':
            # For hupu, load mapping to get titles
            item_mapping = self._get_cached_item_mapping()

            for item_id in item_ids:
                if item_mapping:
                    title = self._extract_title_from_mapping(item_id, item_mapping, f"Item_{item_id}")
                else:
                    # Fallback: try to find title from dataset
                    title = self._get_hupu_title_by_id(item_id)
                    if not title:
                        title = f"Item_{item_id}"

                items_with_titles.append({
                    "hupuID": item_id,
                    "title": title
                })
        elif self.dataset == 'bilibili':
            # For bilibili, use cached item mapping to get titles
            item_mapping = self._get_cached_item_mapping()

            for item_id in item_ids:
                title = self._extract_title_from_mapping(item_id, item_mapping, item_id)
                items_with_titles.append({
                    "bvid": item_id,
                    "title": title
                })

        return items_with_titles

    def _get_douban_title_by_id(self, douban_id: str) -> Optional[str]:
        """Get title for a DoubanID by searching through dataset"""
        dataset_root = f"dataset/{self.dataset}"

        if not os.path.exists(dataset_root):
            return None

        # Search through all user CSV files to find the title for this douban_id
        for user_folder in os.listdir(dataset_root):
            user_path = os.path.join(dataset_root, user_folder)
            if not os.path.isdir(user_path):
                continue

            for file in os.listdir(user_path):
                if file.endswith('.csv'):
                    try:
                        csv_path = os.path.join(user_path, file)
                        df = pd.read_csv(csv_path)
                        if 'doubanID' in df.columns and 'title' in df.columns:
                            # Find matching douban_id
                            matching_rows = df[df['doubanID'].astype(str) == str(douban_id)]
                            if not matching_rows.empty:
                                return str(matching_rows.iloc[0]['title'])
                    except Exception as e:
                        continue

        return None

    def _get_redbook_title_by_id(self, redbook_id: str) -> Optional[str]:
        """Get title for a redbookID by searching through dataset"""
        dataset_root = f"dataset/{self.dataset}"

        if not os.path.exists(dataset_root):
            return None

        # Search through all user CSV files to find the title for this redbook_id
        for user_folder in os.listdir(dataset_root):
            user_path = os.path.join(dataset_root, user_folder)
            if not os.path.isdir(user_path):
                continue

            for file in os.listdir(user_path):
                if file.endswith('.csv'):
                    try:
                        csv_path = os.path.join(user_path, file)
                        df = pd.read_csv(csv_path)
                        if 'redbookID' in df.columns and 'title' in df.columns:
                            # Find matching redbookID
                            matching_rows = df[df['redbookID'].astype(str) == str(redbook_id)]
                            if not matching_rows.empty:
                                return str(matching_rows.iloc[0]['title'])
                    except Exception as e:
                        continue

        return None

    def _get_hupu_title_by_id(self, hupu_id: str) -> Optional[str]:
        """Get title for a hupuID by searching through dataset"""
        dataset_root = f"dataset/{self.dataset}"

        if not os.path.exists(dataset_root):
            return None

        # Search through all user CSV files to find the title for this hupu_id
        for user_folder in os.listdir(dataset_root):
            user_path = os.path.join(dataset_root, user_folder)
            if not os.path.isdir(user_path):
                continue

            for file in os.listdir(user_path):
                if file.endswith('.csv'):
                    try:
                        csv_path = os.path.join(user_path, file)
                        df = pd.read_csv(csv_path)
                        if 'hupuID' in df.columns and 'title' in df.columns:
                            # Find matching hupuID
                            matching_rows = df[df['hupuID'].astype(str) == str(hupu_id)]
                            if not matching_rows.empty:
                                return str(matching_rows.iloc[0]['title'])
                    except Exception as e:
                        continue

        return None

    def _save_reflection_data(self, user_real_id: str, random_items_with_titles: List[Dict[str, str]], validation_items: List[str], test_items: List[str], recommended_items: List[str]):
        """Save random items, validation and test data to JSON file"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Get titles for validation and test items
        validation_items_with_titles = self._get_items_with_titles(validation_items)
        test_items_with_titles = self._get_items_with_titles(test_items)
        recommended_items_with_titles = self._get_items_with_titles(recommended_items)

        # Create the data structure
        reflection_data = {
            "user_id": user_real_id,
            "dataset": self.dataset,
            "timestamp": pd.Timestamp.now().isoformat(),
            "random_items": {
                "count": len(random_items_with_titles),
                "items": random_items_with_titles
            },
            "validation_items": {
                "count": len(validation_items_with_titles),
                "items": validation_items_with_titles
            },
            "test_items": {
                "count": len(test_items_with_titles),
                "items": test_items_with_titles
            },
            "recommended_items": {
                "count": len(recommended_items_with_titles),
                "items": recommended_items_with_titles
            },
            "summary": {
                "total_random_items": len(random_items_with_titles),
                "total_validation_items": len(validation_items_with_titles),
                "total_test_items": len(test_items_with_titles),
                "total_recommended_items": len(recommended_items_with_titles)
            }
        }

        # Save to JSON file
        reflection_data_file = os.path.join(user_dir, "reflection_data.json")
        with open(reflection_data_file, 'w', encoding='utf-8') as f:
            json.dump(reflection_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Reflection data saved for {user_real_id}: {reflection_data_file}")
        print(f"   - Random items: {len(random_items_with_titles)}")
        print(f"   - Validation items: {len(validation_items_with_titles)}")
        print(f"   - Test items: {len(test_items_with_titles)}")
        print(f"   - Recommended items: {len(recommended_items_with_titles)}")




def _process_single_user_sync(analyzer: EnhancedAnalyzer, user_folder: str) -> Dict:
    """
    Synchronous version of _process_single_user using ThreadPoolExecutor

    Args:
        analyzer: EnhancedAnalyzer instance
        user_folder: User folder name

    Returns:
        Dict with processing results
    """
    try:
        dataset_path = os.path.join("download", analyzer.dataset)
        user_dir = os.path.join(dataset_path, user_folder)

        # Check if user has already been analyzed (skip if reflection_data.json exists)
        reflection_data_file = os.path.join(user_dir, "reflection_data.json")
        if os.path.exists(reflection_data_file):
            print(f"\n⏭️  [SKIP] User {user_folder} already analyzed (reflection_data.json exists)")
            return {
                'user': user_folder,
                'status': 'skipped',
                'reason': 'already_analyzed'
            }

        print(f"\n🏠 [THREAD] Processing user: {user_folder}")

        # Convert user folder name to internal ID for matrix operations
        user_internal_id = None
        for internal_id, real_id in analyzer.user_map.items():
            if real_id == user_folder:
                user_internal_id = int(internal_id)
                break

        if user_internal_id is None:
            print(f"⚠️  Could not find internal ID for user {user_folder}")
            if analyzer.dataset == 'douban':
                print(f"📝 For douban dataset, this might be expected if SSL data doesn't match dataset folders")
                print(f"📝 Skipping SSL-dependent features for user {user_folder}")
                return {'user': user_folder, 'status': 'skipped', 'reason': 'douban_no_ssl_mapping'}
            else:
                return {'user': user_folder, 'status': 'failed', 'reason': 'no_internal_id'}

        # 1. Analyze recommendation hits
        recommended_items = []
        recommended_dir = os.path.join(user_dir, "recommended")
        if os.path.exists(recommended_dir):
            recommended_items = [item for item in os.listdir(recommended_dir)
                               if os.path.isdir(os.path.join(recommended_dir, item))]

            hit_analysis = analyzer.check_recommendation_hits(user_internal_id, recommended_items)

            print(f"📊 Recommendation Hit Analysis for {user_folder}:")
            print(f"   - Total recommendations: {hit_analysis['total_recommendations']}")
            print(f"   - Test items: {len(hit_analysis['test_items'])}")
            print(f"   - Hits: {hit_analysis['hit_count']}")
            print(f"   - Hit rate: {hit_analysis['hit_rate']:.2%}")
            print(f"   - Hit items: {hit_analysis['hits']}")

        # 2. Download and analyze validation items
        print(f"📥 Processing validation items for {user_folder}...")

        # 3. Load current user profile
        user_profile_file = os.path.join(user_dir, "user_profile.txt")
        if os.path.exists(user_profile_file):
            with open(user_profile_file, 'r', encoding='utf-8') as f:
                current_user_profile = f.read()
        else:
            print(f"⚠️  User profile not found for {user_folder}: {user_profile_file}")
            return {'user': user_folder, 'status': 'failed', 'reason': 'no_user_profile'}

        # 4. Start iterative reflection process
        validation_items = analyzer.get_user_validation_items(user_internal_id)
        test_items = analyzer.get_user_test_items(user_internal_id)

        if validation_items and test_items:
            # Check if validation analysis exists before starting reflection process
            print(f"🔍 Checking validation analysis for user {user_folder}...")
            validation_analysis = analyzer._get_validation_item_analysis(validation_items, user_folder)

            if validation_analysis == "NO_VALIDATION_ANALYSIS_FOUND":
                print(f"❌ No validation analysis found for user {user_folder}, skipping reflection process")
                return {'user': user_folder, 'status': 'skipped', 'reason': 'no_validation_analysis'}

            print(f"✅ Validation analysis found for user {user_folder}")
            print(f"🚀 Starting iterative reflection process for {user_folder}")
            print(f"   - Validation items: {len(validation_items)}")
            print(f"   - Test items: {len(test_items)}")

            # Run the synchronous method
            analyzer.perform_iterative_reflection(
                user_folder,
                user_internal_id,
                current_user_profile,
                validation_items,
                test_items,
                recommended_items
            )

            print(f"✅ Enhanced analysis completed for user {user_folder}")
            return {
                'user': user_folder,
                'status': 'completed',
                'validation_items': len(validation_items),
                'test_items': len(test_items),
                'recommended_items': len(recommended_items)
            }
        else:
            reasons = []
            if not validation_items:
                print(f"⚠️  No validation items found for user {user_folder}")
                reasons.append('no_validation_items')
            if not test_items:
                print(f"⚠️  No test items found for user {user_folder}")
                reasons.append('no_test_items')

            return {'user': user_folder, 'status': 'skipped', 'reason': ', '.join(reasons)}

    except Exception as e:
        print(f"❌ Error processing user {user_folder}: {e}")
        return {'user': user_folder, 'status': 'error', 'error': str(e)}


def _collect_global_statistics(dataset_path: str, user_folders: List[str]) -> Dict:
    """Collect global statistics from all users' reflection results"""
    global_stats = {
        "validation_stats": {"all_records": []},
        "test_stats": {"all_records": []},
        "user_count": 0,
        "successful_users": 0,
        "total_test_users": 0  # Total users with test records
    }

    for user_folder in user_folders:
        user_dir = os.path.join(dataset_path, user_folder)
        has_test_records = False

        # Load validation reflection results
        valid_file = os.path.join(user_dir, "valid_reflection_results.json")
        if os.path.exists(valid_file):
            try:
                with open(valid_file, 'r', encoding='utf-8') as f:
                    valid_data = json.load(f)
                    if 'validation_records' in valid_data:
                        # Add user_id to each record for tracking
                        for record in valid_data['validation_records']:
                            record['user_id'] = user_folder
                        global_stats["validation_stats"]["all_records"].extend(valid_data['validation_records'])
            except Exception as e:
                print(f"⚠️  Error loading validation results for {user_folder}: {e}")

        # Load test reflection results
        test_file = os.path.join(user_dir, "test_reflection_results.json")
        if os.path.exists(test_file):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                    if 'test_records' in test_data and len(test_data['test_records']) > 0:
                        # Add user_id to each record for tracking
                        for record in test_data['test_records']:
                            record['user_id'] = user_folder
                        global_stats["test_stats"]["all_records"].extend(test_data['test_records'])
                        has_test_records = True
            except Exception as e:
                print(f"⚠️  Error loading test results for {user_folder}: {e}")

        if has_test_records:
            global_stats["successful_users"] += 1
            global_stats["total_test_users"] += 1
        global_stats["user_count"] += 1

    return global_stats


def _calculate_global_reflection_statistics(global_stats: Dict) -> Dict:
    """Calculate cumulative global NDCG and Hit Rate statistics across all users.

    For Hit Rate:
    - Hit Rate = cumulative hit users / total users (not average of individual hit rates)
    - A user is counted as "hit" if they have hit_success=True in any record up to that round
    - The denominator is always the total number of users, not just users with records

    For NDCG:
    - For each user, take the NDCG value from their latest record up to that round
    - For users without records up to that round, their NDCG is counted as 0
    - Then average across ALL users (not just users with records)

    The last round's statistics represent the global statistics across all rounds.
    """
    stats = {
        "validation_global": {},
        "test_global": {}
    }

    # Total users should be the total number of users processed (user_count)
    total_users = global_stats.get("user_count", global_stats.get("total_test_users", global_stats.get("successful_users", 0)))
    if total_users == 0:
        total_users = 1  # Avoid division by zero

    # Calculate cumulative validation statistics by round
    validation_records = global_stats["validation_stats"]["all_records"]
    validation_by_round = {}
    for record in validation_records:
        round_num = record.get('round', 0)
        if round_num not in validation_by_round:
            validation_by_round[round_num] = []
        validation_by_round[round_num].append(record)

    # Get all round numbers and sort them
    validation_rounds = sorted(validation_by_round.keys())

    # Track cumulative hit users across rounds for validation
    cumulative_validation_hit_5 = set()
    cumulative_validation_hit_10 = set()
    cumulative_validation_hit_20 = set()
    cumulative_validation_ndcg_5 = {}  # user_id -> latest ndcg value
    cumulative_validation_ndcg_10 = {}
    cumulative_validation_ndcg_20 = {}

    # Calculate cumulative statistics for each round
    for target_round in validation_rounds:
        # Process records for this round
        if target_round in validation_by_round:
            for record in validation_by_round[target_round]:
                user_id = record.get('user_id', 'unknown')

                # Update NDCG (always use latest value for each user)
                cumulative_validation_ndcg_5[user_id] = record.get('ndcg_at_5', 0)
                cumulative_validation_ndcg_10[user_id] = record.get('ndcg_at_10', 0)
                cumulative_validation_ndcg_20[user_id] = record.get('ndcg_at_20', 0)

                # Update hit status (once hit, always counted as hit)
                if record.get('hit_rate_at_5', 0) > 0:
                    cumulative_validation_hit_5.add(user_id)
                if record.get('hit_rate_at_10', 0) > 0:
                    cumulative_validation_hit_10.add(user_id)
                if record.get('hit_rate_at_20', 0) > 0:
                    cumulative_validation_hit_20.add(user_id)

        hit_users_5 = len(cumulative_validation_hit_5)
        hit_users_10 = len(cumulative_validation_hit_10)
        hit_users_20 = len(cumulative_validation_hit_20)

        # Calculate NDCG average across ALL users (users without records count as 0)
        avg_ndcg_5 = sum(cumulative_validation_ndcg_5.values()) / total_users
        avg_ndcg_10 = sum(cumulative_validation_ndcg_10.values()) / total_users
        avg_ndcg_20 = sum(cumulative_validation_ndcg_20.values()) / total_users

        stats["validation_global"][f"round_{target_round}"] = {
            "avg_ndcg_at_5": avg_ndcg_5,
            "avg_ndcg_at_10": avg_ndcg_10,
            "avg_ndcg_at_20": avg_ndcg_20,
            "hit_rate_at_5": hit_users_5 / total_users,
            "hit_rate_at_10": hit_users_10 / total_users,
            "hit_rate_at_20": hit_users_20 / total_users,
            "hit_users_5": hit_users_5,
            "hit_users_10": hit_users_10,
            "hit_users_20": hit_users_20,
            "total_users": total_users,
            "users_with_records": len(cumulative_validation_ndcg_5)
        }

    # Calculate cumulative test statistics by round
    test_records = global_stats["test_stats"]["all_records"]
    test_by_round = {}
    for record in test_records:
        round_num = record.get('round', 0)
        if round_num not in test_by_round:
            test_by_round[round_num] = []
        test_by_round[round_num].append(record)

    # Get all round numbers and sort them
    test_rounds = sorted(test_by_round.keys())

    # Track cumulative hit users across rounds for test
    cumulative_test_hit_5 = set()
    cumulative_test_hit_10 = set()
    cumulative_test_hit_20 = set()
    cumulative_test_ndcg_5 = {}  # user_id -> latest ndcg value
    cumulative_test_ndcg_10 = {}
    cumulative_test_ndcg_20 = {}

    # Calculate cumulative statistics for each round
    for target_round in test_rounds:
        # Process records for this round
        if target_round in test_by_round:
            for record in test_by_round[target_round]:
                user_id = record.get('user_id', 'unknown')

                # Update NDCG (always use latest value for each user)
                cumulative_test_ndcg_5[user_id] = record.get('ndcg_at_5', 0)
                cumulative_test_ndcg_10[user_id] = record.get('ndcg_at_10', 0)
                cumulative_test_ndcg_20[user_id] = record.get('ndcg_at_20', 0)

                # Update hit status (once hit, always counted as hit)
                if record.get('hit_rate_at_5', 0) > 0:
                    cumulative_test_hit_5.add(user_id)
                if record.get('hit_rate_at_10', 0) > 0:
                    cumulative_test_hit_10.add(user_id)
                if record.get('hit_rate_at_20', 0) > 0:
                    cumulative_test_hit_20.add(user_id)

        hit_users_5 = len(cumulative_test_hit_5)
        hit_users_10 = len(cumulative_test_hit_10)
        hit_users_20 = len(cumulative_test_hit_20)

        # Calculate NDCG average across ALL users (users without records count as 0)
        avg_ndcg_5 = sum(cumulative_test_ndcg_5.values()) / total_users
        avg_ndcg_10 = sum(cumulative_test_ndcg_10.values()) / total_users
        avg_ndcg_20 = sum(cumulative_test_ndcg_20.values()) / total_users

        stats["test_global"][f"round_{target_round}"] = {
            "avg_ndcg_at_5": avg_ndcg_5,
            "avg_ndcg_at_10": avg_ndcg_10,
            "avg_ndcg_at_20": avg_ndcg_20,
            "hit_rate_at_5": hit_users_5 / total_users,
            "hit_rate_at_10": hit_users_10 / total_users,
            "hit_rate_at_20": hit_users_20 / total_users,
            "hit_users_5": hit_users_5,
            "hit_users_10": hit_users_10,
            "hit_users_20": hit_users_20,
            "total_users": total_users,
            "users_with_records": len(cumulative_test_ndcg_5)
        }

    return stats


def enhanced_analyze(max_concurrent_users: int = 3):
    """
    Enhanced analysis function with hit analysis and self-reflection using ThreadPoolExecutor for concurrent processing

    Args:
        max_concurrent_users: Maximum number of users to process concurrently
    """
    print("🚀 Starting enhanced analysis with ThreadPoolExecutor...")
    print("=" * 70)

    analyzer = EnhancedAnalyzer()
    dataset_path = os.path.join("download", analyzer.dataset)

    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset path not found: {dataset_path}")
        return

    # 1. Collect all user folders
    user_folders = []
    for user_folder in os.listdir(dataset_path):
        user_dir = os.path.join(dataset_path, user_folder)
        if os.path.isdir(user_dir):
            user_folders.append(user_folder)

    if not user_folders:
        print("❌ No user folders found")
        return

    print(f"📊 Found {len(user_folders)} users to process")
    print(f"🔄 Using {max_concurrent_users} concurrent workers")

    # 2. Process all users concurrently using ThreadPoolExecutor
    print(f"\n🔄 Processing all users concurrently...")
    print(f"🚀 Starting {len(user_folders)} concurrent tasks with max {max_concurrent_users} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent_users) as executor:
        # Submit all tasks
        future_to_user = {
            executor.submit(_process_single_user_sync, analyzer, user_folder): user_folder
            for user_folder in user_folders
        }

        # Process results as they complete
        completed_count = 0
        for future in future_to_user:
            result = future.result()
            results.append(result)
            completed_count += 1

            status_emoji = {
                'completed': '✅',
                'skipped': '⏭️',
                'failed': '❌',
                'error': '💥'
            }.get(result['status'], '❓')

            print(f"  {status_emoji} [{completed_count}/{len(user_folders)}] User {result['user']}: {result['status']}")
            if result['status'] == 'completed':
                print(f"    📊 Validation: {result.get('validation_items', 0)}, Test: {result.get('test_items', 0)}, Recommended: {result.get('recommended_items', 0)}")

    # 3. Print summary
    print("\n" + "=" * 70)
    print("🎉 Enhanced analysis completed!")

    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"📊 Summary:")
    print(f"  - Total users: {len(user_folders)}")
    for status, count in status_counts.items():
        emoji = {
            'completed': '✅',
            'skipped': '⏭️',
            'failed': '❌',
            'error': '💥'
        }.get(status, '❓')
        print(f"  - {emoji} {status.title()}: {count}")

    # 4. Calculate and display global statistics
    completed_users = [result for result in results if result['status'] == 'completed']
    if completed_users:
        print(f"\n🌍 计算全局统计数据...")

        # Collect global statistics from all users
        global_stats = _collect_global_statistics(dataset_path, [r['user'] for r in completed_users])

        if global_stats["validation_stats"]["all_records"] or global_stats["test_stats"]["all_records"]:
            print(f"\n📊 全局反思统计结果 (基于 {global_stats['successful_users']} 个用户):")

            # Calculate global statistics
            global_reflection_stats = _calculate_global_reflection_statistics(global_stats)

            # Display global validation statistics by round (cumulative)
            if global_reflection_stats["validation_global"]:
                print(f"🔍 全局验证阶段统计 (累积):")
                for round_key, round_stats in sorted(global_reflection_stats["validation_global"].items()):
                    round_num = round_key.replace("round_", "")
                    print(f"   Round {round_num}: NDCG@5={round_stats['avg_ndcg_at_5']:.3f}, NDCG@10={round_stats['avg_ndcg_at_10']:.3f}, NDCG@20={round_stats['avg_ndcg_at_20']:.3f}")
                    print(f"            HR@5={round_stats['hit_rate_at_5']:.3f} ({round_stats['hit_users_5']}/{round_stats['total_users']}), HR@10={round_stats['hit_rate_at_10']:.3f} ({round_stats['hit_users_10']}/{round_stats['total_users']}), HR@20={round_stats['hit_rate_at_20']:.3f} ({round_stats['hit_users_20']}/{round_stats['total_users']})")

            # Display global test statistics by round (cumulative)
            if global_reflection_stats["test_global"]:
                print(f"🧪 全局测试阶段统计 (累积):")
                for round_key, round_stats in sorted(global_reflection_stats["test_global"].items()):
                    round_num = round_key.replace("round_", "")
                    print(f"   Round {round_num}: NDCG@5={round_stats['avg_ndcg_at_5']:.3f}, NDCG@10={round_stats['avg_ndcg_at_10']:.3f}, NDCG@20={round_stats['avg_ndcg_at_20']:.3f}")
                    print(f"            HR@5={round_stats['hit_rate_at_5']:.3f} ({round_stats['hit_users_5']}/{round_stats['total_users']}), HR@10={round_stats['hit_rate_at_10']:.3f} ({round_stats['hit_users_10']}/{round_stats['total_users']}), HR@20={round_stats['hit_rate_at_20']:.3f} ({round_stats['hit_users_20']}/{round_stats['total_users']})")

            # Save global statistics to file
            global_stats_file = os.path.join(dataset_path, "global_reflection_statistics.json")
            global_stats_data = {
                "dataset": analyzer.dataset,
                "total_users_processed": len(user_folders),
                "successful_users": global_stats["successful_users"],
                "global_statistics": global_reflection_stats,
                "timestamp": pd.Timestamp.now().isoformat()
            }

            with open(global_stats_file, 'w', encoding='utf-8') as f:
                json.dump(global_stats_data, f, ensure_ascii=False, indent=2)
            print(f"💾 全局统计数据已保存到: {global_stats_file}")
        else:
            print(f"⚠️  未找到有效的反思统计数据")
    else:
        print(f"⚠️  没有成功完成的用户，无法计算全局统计")

    print("=" * 70)

