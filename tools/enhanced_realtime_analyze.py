import os
import json
import pickle
import random
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.video_analyst import VideoAnalyst
from agents.image_analyst import ImageAnalyst
from agents.text_analyst import TextAnalyst
from agents.user_profile_generator import UserProfileGenerator
from tools.analyze import _convert_user_id, get_chronological_order, _collect_all_items, _process_single_item_task, format_analysis_to_prompt


class EnhancedRealtimeAnalyzer:
    def __init__(self):
        self.dataset = os.getenv("DATASET", "bilibili")
        self.video_analyst = VideoAnalyst(max_workers=2)
        self.image_analyst = ImageAnalyst(max_workers=2)
        self.text_analyst = TextAnalyst(max_workers=2)
        self.user_profile_generator = UserProfileGenerator()

        # Test control parameter: True = test each round, False = test only final round
        self.test_each_round = os.getenv("TEST_EACH_ROUND", "true").lower() == "true"

        # Thread pool for API calls
        self.api_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="API-")

    def __del__(self):
        """Cleanup method to shutdown thread pool"""
        if hasattr(self, 'api_thread_pool'):
            self.api_thread_pool.shutdown(wait=False)

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

    def get_user_validation_items(self, user_real_id: str) -> List[str]:
        """Get validation items for a user from analysis files (second newest item globally)"""
        try:
            # Get all items with analysis from user's directories
            all_items = self._get_user_analyzed_items(user_real_id)

            if len(all_items) < 2:
                return []

            # Sort by timestamp (newest first)
            all_items.sort(key=lambda x: x['timestamp'], reverse=True)

            # Take second newest item as validation set
            validation_item = all_items[1]['item_id']
            return [validation_item]

        except Exception as e:
            print(f"⚠️  Error getting validation items for user {user_real_id}: {e}")
            return []

    def get_user_test_items(self, user_real_id: str) -> List[str]:
        """Get test items for a user from analysis files (newest item globally)"""
        try:
            # Get all items with analysis from user's directories
            all_items = self._get_user_analyzed_items(user_real_id)

            if len(all_items) < 1:
                return []

            # Sort by timestamp (newest first)
            all_items.sort(key=lambda x: x['timestamp'], reverse=True)

            # Take newest item as test set
            test_item = all_items[0]['item_id']
            return [test_item]

        except Exception as e:
            print(f"⚠️  Error getting test items for user {user_real_id}: {e}")
            return []

    def check_recommendation_hits(self, user_real_id: str, recommended_items: List[str]) -> Dict:
        """Check if recommended items hit test items (same as enhanced_analyze)"""
        test_items = self.get_user_test_items(user_real_id)
        print(f"[INFO] Test videos: {test_items}")
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

    def _get_old_test_validation_items_from_folders(self, user_real_id: str) -> Tuple[List[str], List[str]]:
        """Get items from existing test/ and validation/ folders"""
        user_dir = os.path.join("download", self.dataset, user_real_id)
        old_test_items = []
        old_validation_items = []

        # Get items from test/ folder
        test_dir = os.path.join(user_dir, "test")
        if os.path.exists(test_dir):
            for item_name in os.listdir(test_dir):
                item_path = os.path.join(test_dir, item_name)
                if os.path.isdir(item_path):
                    analysis_file = os.path.join(item_path, "analysis.json")
                    if os.path.exists(analysis_file):
                        old_test_items.append(item_name)

        # Get items from validation/ folder
        validation_dir = os.path.join(user_dir, "validation")
        if os.path.exists(validation_dir):
            for item_name in os.listdir(validation_dir):
                item_path = os.path.join(validation_dir, item_name)
                if os.path.isdir(item_path):
                    analysis_file = os.path.join(item_path, "analysis.json")
                    if os.path.exists(analysis_file):
                        old_validation_items.append(item_name)

        return old_test_items, old_validation_items

    def generate_realtime_item_profile(self, user_real_id: str) -> str:
        """Generate realtime item profile from all analysis files (historical + recommended + realtime + old test/valid, excluding new test/valid)"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Get NEW validation and test items to exclude (based on timestamp - newest and second newest)
        new_validation_items = self.get_user_validation_items(user_real_id)
        new_test_items = self.get_user_test_items(user_real_id)
        new_exclude_items = set(new_validation_items + new_test_items)

        # Get OLD test/validation items from folders
        old_test_items, old_validation_items = self._get_old_test_validation_items_from_folders(user_real_id)
        old_items_set = set(old_test_items + old_validation_items)

        # Items from old test/valid folders that are NOT in new test/valid should be INCLUDED in profile
        old_items_to_include = old_items_set - new_exclude_items

        print(f"📋 New test/valid items (to exclude): {new_exclude_items}")
        print(f"📋 Old test/valid items from folders: {old_items_set}")
        print(f"📋 Old items to include in profile: {old_items_to_include}")

        # Get all analyzed items
        all_items = self._get_user_analyzed_items(user_real_id)

        if not all_items:
            print(f"⚠️  No analyzed items found for user {user_real_id}")
            return ""

        # Filter items:
        # - Include: historical, recommended, realtime sources
        # - Include: old test/valid items that are not in new test/valid
        # - Exclude: new test/valid items
        allowed_sources = ['historical', 'recommended', 'realtime']
        filtered_items = []
        for item in all_items:
            item_id = item['item_id']
            # Skip new test/valid items
            if item_id in new_exclude_items:
                continue
            # Include if source is allowed OR if it's an old test/valid item not in new exclusions
            if item['source'] in allowed_sources or item_id in old_items_to_include:
                filtered_items.append(item)

        if not filtered_items:
            print(f"⚠️  No items left after filtering for user {user_real_id}")
            return ""

        # Sort ALL items by timestamp (oldest first for chronological profile)
        filtered_items.sort(key=lambda x: x['timestamp'])

        profile_parts = []
        profile_parts.append("=== All Interaction Records (Chronological Order) ===\n")
        profile_parts.append(f"# Items from: historical + recommended + realtime + old_test/valid (excluding new test/valid)\n")

        # Count items by source
        source_counts = {}
        for item in filtered_items:
            source_label = item['source']
            if item['item_id'] in old_items_to_include and item['source'] in ['test', 'validation']:
                source_label = f"old_{item['source']}"
            source_counts[source_label] = source_counts.get(source_label, 0) + 1

        print(f"📊 Realtime item profile for {user_real_id}:")
        for source, count in source_counts.items():
            print(f"   - {source}: {count} items")

        # Process all items chronologically
        for item in filtered_items:
            analysis_content = self._load_analysis_file(item['analysis_file'])
            if analysis_content:
                # Mark the source type with appropriate label
                source_labels = {
                    'historical': 'Historical Interaction',
                    'recommended': 'Recommended Item',
                    'realtime': 'Realtime Interaction',
                    'test': 'Previous Test Item',
                    'validation': 'Previous Validation Item'
                }
                source_label = source_labels.get(item['source'], item['source'])

                formatted_analysis = self._format_analysis_to_realtime_prompt(
                    analysis_content, item['item_id'], source_label, user_real_id
                )
                profile_parts.append(formatted_analysis)
                profile_parts.append("")  # Add spacing

        # Save to file
        profile_content = "\n".join(profile_parts)
        profile_file = os.path.join(user_dir, "realtime_item_profile.txt")

        try:
            with open(profile_file, 'w', encoding='utf-8') as f:
                f.write(profile_content)
            print(f"💾 Realtime item profile saved: {profile_file}")
        except Exception as e:
            print(f"⚠️  Error saving realtime item profile: {e}")

        return profile_content

    def _load_analysis_file(self, analysis_file_path: str) -> Optional[Dict]:
        """Load analysis data from JSON file"""
        try:
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading analysis file {analysis_file_path}: {e}")
            return None

    def _format_analysis_to_realtime_prompt(self, analysis_data: Dict, item_name: str, source_label: str, user_real_id: str = None) -> str:
        """Format analysis results into natural language prompt with source label and item title"""
        # Get item title from dataset files or mapping
        if user_real_id:
            item_title = self._get_item_title_from_dataset_files(item_name, user_real_id)
        else:
            item_title = item_name

        # Format: "## Source Label: Title (item_id)" or just "## Source Label: Title" if title != item_name
        if item_title and item_title != item_name:
            prompt_parts = [f"## {source_label}: {item_title}\n"]
        else:
            prompt_parts = [f"## {source_label}: {item_name}\n"]

        # Process text content
        if "text" in analysis_data and analysis_data["text"]:
            prompt_parts.append("### Text Content:")
            for idx, text_analysis in analysis_data["text"].items():
                prompt_parts.append(f"- {text_analysis}")
            prompt_parts.append("")

        # Process image content
        if "image" in analysis_data and analysis_data["image"]:
            prompt_parts.append("### Image Content:")
            for idx, image_analysis in analysis_data["image"].items():
                prompt_parts.append(f"- Image {idx}: {image_analysis}")
            prompt_parts.append("")

        # Process video content
        if "video" in analysis_data and analysis_data["video"]:
            prompt_parts.append("### Video Content:")
            for idx, video_analysis in analysis_data["video"].items():
                prompt_parts.append(f"- Video {idx}: {video_analysis}")
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    def generate_realtime_user_profile(self, user_real_id: str) -> str:
        """Generate realtime user profile from realtime item profile"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # First generate or load item profile
        item_profile_file = os.path.join(user_dir, "realtime_item_profile.txt")
        if not os.path.exists(item_profile_file):
            print(f"📝 Generating realtime item profile for user {user_real_id}")
            item_profile_content = self.generate_realtime_item_profile(user_real_id)
        else:
            print(f"📖 Loading existing realtime item profile for user {user_real_id}")
            try:
                with open(item_profile_file, 'r', encoding='utf-8') as f:
                    item_profile_content = f.read()
            except Exception as e:
                print(f"⚠️  Error loading item profile: {e}")
                item_profile_content = self.generate_realtime_item_profile(user_real_id)

        if not item_profile_content:
            print(f"⚠️  No item profile content available for user {user_real_id}")
            return ""

        # Generate user profile using LLM
        profile_prompt = f"""
Based on the following user's real-time interaction records and content analysis, please generate a detailed user profile. This profile should capture the user's interest preferences, behavioral patterns, and content consumption habits.

User Interaction Content Analysis:
{item_profile_content}

Please generate a comprehensive user profile including:
1. Main interest areas and preferred topics
2. Content consumption patterns and behavioral characteristics
3. Possible user attributes (age group, professional background, etc.)
4. Content preference characteristics (video types, duration, style, etc.)
5. Interest development trends and changes

User Profile:
"""

        try:
            user_profile = self.user_profile_generator(item_profile_content)

            # Save user profile
            profile_file = os.path.join(user_dir, "realtime_user_profile.txt")
            with open(profile_file, 'w', encoding='utf-8') as f:
                f.write(user_profile)
            print(f"💾 Realtime user profile saved: {profile_file}")

            return user_profile.strip()

        except Exception as e:
            print(f"⚠️  Error generating realtime user profile: {e}")
            return ""

    def perform_iterative_realtime_reflection_sync(self, user_profile: str, user_real_id: str, max_reflections: int = 3) -> Tuple[bool, str]:
        """
        Perform iterative reflection process (same logic as enhanced_analyze):

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
        print(f"🔄 Starting iterative realtime reflection for user {user_real_id}...")

        # Get validation and test items
        validation_items = self.get_user_validation_items(user_real_id)
        test_items = self.get_user_test_items(user_real_id)

        if not validation_items:
            print(f"⚠️  No validation items found for user {user_real_id}")
            return False, user_profile

        if not test_items:
            print(f"⚠️  No test items found for user {user_real_id}")
            return False, user_profile

        print(f"📊 Found validation item: {validation_items[0]}")
        print(f"📊 Found test item: {test_items[0]}")

        test_mode = "Test each round" if self.test_each_round else "Test only final round"
        print(f"🧪 Test mode: {test_mode}")

        # Get user's all items to exclude from random selection
        user_all_items = self._get_user_all_items(user_real_id)
        exclude_items = validation_items + test_items + user_all_items

        # Prepare random items (same set used for all rounds)
        print(f"🎲 Generating random items for reflection process...")
        random_items_with_titles = self._get_random_items_with_titles(user_real_id, exclude_items)
        print(f"✅ Generated {len(random_items_with_titles)} random items for reflection")

        # Save reflection data
        self._save_realtime_reflection_data_sync(user_real_id, random_items_with_titles, validation_items, test_items)

        # Storage for all results
        all_valid_records = []
        all_test_records = []
        current_profile = user_profile

        # Save initial profile as Round 0
        self._save_initial_realtime_profile_sync(current_profile, user_real_id)

        # === Round 0: Initial test with original profile ===
        print(f"\n🎯 Round 0: Initial test with original user profile")
        round_0_hit = self._perform_single_realtime_test(
            current_profile, test_items, random_items_with_titles, user_real_id, 0, "test"
        )

        if round_0_hit:
            print(f"✅ Round 0 test hit! Ending reflection process.")
            self._save_final_realtime_results(user_real_id, all_valid_records, all_test_records, user_profile, current_profile, True, 0)
            return True, current_profile

        # === Rounds 1+: Validation reflection + optional test ===
        for round_num in range(1, max_reflections + 1):
            print(f"\n🔄 Round {round_num}: Validation reflection")

            # Perform single round validation reflection
            valid_hit_success, enhanced_profile = self._perform_single_realtime_validation_round(
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
                test_hit = self._perform_single_realtime_test(
                    current_profile, test_items, random_items_with_titles, user_real_id, round_num, "test"
                )
                if test_hit:
                    print(f"✅ Round {round_num} test hit! Process completed successfully.")
                    self._save_final_realtime_results(user_real_id, all_valid_records, all_test_records, user_profile, current_profile, True, round_num)
                    return True, current_profile
                else:
                    print(f"❌ Round {round_num} test miss. Process completed.")
                    self._save_final_realtime_results(user_real_id, all_valid_records, all_test_records, user_profile, current_profile, False, round_num)
                    return False, current_profile

            else:
                print(f"❌ Round {round_num} validation miss.")

                # Decide whether to perform test based on mode
                if self.test_each_round:
                    print(f"🧪 Performing Round {round_num} test (mode: test each round)")
                    test_hit = self._perform_single_realtime_test(
                        current_profile, test_items, random_items_with_titles, user_real_id, round_num, "test"
                    )
                    if test_hit:
                        print(f"✅ Round {round_num} test hit! Process completed successfully.")
                        self._save_final_realtime_results(user_real_id, all_valid_records, all_test_records, user_profile, current_profile, True, round_num)
                        return True, current_profile

        # If we reach here, all rounds completed without success
        print(f"❌ All {max_reflections} rounds completed without success.")

        # Perform final test if mode is "test only final round" and we haven't tested in the last round
        if not self.test_each_round:
            print(f"🧪 Performing final test (mode: test only final round)")
            final_test_hit = self._perform_single_realtime_test(
                current_profile, test_items, random_items_with_titles, user_real_id, max_reflections, "test"
            )
            self._save_final_realtime_results(user_real_id, all_valid_records, all_test_records, user_profile, current_profile, final_test_hit, max_reflections)
            return final_test_hit, current_profile
        else:
            self._save_final_realtime_results(user_real_id, all_valid_records, all_test_records, user_profile, current_profile, False, max_reflections)
            return False, current_profile

    def _perform_single_realtime_test(self, user_profile: str, target_items: List[str], random_items_with_titles: List[Dict[str, str]], user_real_id: str, round_num: int, test_type: str) -> bool:
        """Perform a single test and return whether it hit"""
        # Get real titles for target items from dataset files first, then fallback to mapping
        target_titles = []
        target_item_to_title = {}
        for item_id in target_items:
            title = self._get_item_title_from_dataset_files(item_id, user_real_id)
            target_titles.append(title)
            target_item_to_title[item_id] = title

        # Create list of all titles for prediction
        random_titles = [item['title'] for item in random_items_with_titles]
        all_titles = target_titles + random_titles
        random.shuffle(all_titles)

        # Create mapping from title back to item ID
        title_to_item_id = {}
        for item in random_items_with_titles:
            if self.dataset == 'redbook':
                title_to_item_id[item['title']] = item['redbookID']
            else:
                title_to_item_id[item['title']] = item['bvid']
        for item_id, title in target_item_to_title.items():
            title_to_item_id[title] = item_id

        # Create prediction prompt
        prediction_prompt = f"""
Based on the following user profile, predict the top 20 video titles that this user is most likely to interact with from the given list. Rank them from most likely (1) to least likely (20).

User Profile:
{user_profile}

Available Video Titles:
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
            response_text = response.choices[0].message.content

            predicted_titles = self._extract_predictions_from_response(response_text)
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
                print(f"✅ Round {round_num} {test_type} hit: {hit_items}")
                print(f"   📊 Metrics - NDCG@1: {test_metrics['ndcg_at_1']:.3f}, NDCG@5: {test_metrics['ndcg_at_5']:.3f}, NDCG@10: {test_metrics['ndcg_at_10']:.3f}, NDCG@20: {test_metrics['ndcg_at_20']:.3f}")
                print(f"            HR@1: {test_metrics['hit_rate_at_1']:.3f}, HR@5: {test_metrics['hit_rate_at_5']:.3f}, HR@10: {test_metrics['hit_rate_at_10']:.3f}, HR@20: {test_metrics['hit_rate_at_20']:.3f}")
            else:
                print(f"❌ Round {round_num} {test_type} miss")
                print(f"   📊 Metrics - NDCG@1: {test_metrics['ndcg_at_1']:.3f}, NDCG@5: {test_metrics['ndcg_at_5']:.3f}, NDCG@10: {test_metrics['ndcg_at_10']:.3f}, NDCG@20: {test_metrics['ndcg_at_20']:.3f}")
                print(f"            HR@1: {test_metrics['hit_rate_at_1']:.3f}, HR@5: {test_metrics['hit_rate_at_5']:.3f}, HR@10: {test_metrics['hit_rate_at_10']:.3f}, HR@20: {test_metrics['hit_rate_at_20']:.3f}")

            return hit_success

        except Exception as e:
            print(f"⚠️  Error in Round {round_num} {test_type} test: {e}")
            return False

    def _perform_single_realtime_validation_round(self, user_profile: str, validation_items: List[str], random_items_with_titles: List[Dict[str, str]], user_real_id: str, round_num: int) -> Tuple[bool, str]:
        """Perform a single round of validation reflection"""
        # Get real titles for validation items from dataset files first, then fallback to mapping
        validation_titles = []
        validation_item_to_title = {}

        for item_id in validation_items:
            title = self._get_item_title_from_dataset_files(item_id, user_real_id)
            validation_titles.append(title)
            validation_item_to_title[item_id] = title

        # Create list of all titles for prediction
        random_titles = [item['title'] for item in random_items_with_titles]
        all_titles = validation_titles + random_titles
        random.shuffle(all_titles)

        # Create mapping from title back to item ID for validation checking
        title_to_item_id = {}
        for item in random_items_with_titles:
            if self.dataset == 'redbook':
                title_to_item_id[item['title']] = item['redbookID']
            else:
                title_to_item_id[item['title']] = item['bvid']

        # For validation items, map title to item ID
        for item_id, title in validation_item_to_title.items():
            title_to_item_id[title] = item_id

        # Create prediction prompt with titles
        prediction_prompt = f"""
Based on the following user profile, predict the top 20 video titles that this user is most likely to interact with from the given list. These represent new interaction items that the user might engage with based on their preferences. Rank them from most likely (1) to least likely (20).

User Profile:
{user_profile}

Available Video Titles (New Interaction Items):
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
            response_text = response.choices[0].message.content

            predicted_titles = self._extract_predictions_from_response(response_text)
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

            # Record validation result
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
                print(f"✅ Round {round_num} validation hit: {hit_items}")

                # Enhance profile using validation item analysis
                enhanced_profile = self._enhance_user_profile_with_realtime_sync(user_profile, validation_items, user_real_id)
                if enhanced_profile == "NO_VALIDATION_ANALYSIS_FOUND":
                    print(f"❌ No validation analysis found for user {user_real_id}, cannot enhance profile")
                    return False, user_profile  # Return original profile

                # Save enhanced profile for this round
                self._save_enhanced_realtime_profile_sync(enhanced_profile, user_real_id, round_num, hit_success)

                return True, enhanced_profile
            else:
                print(f"❌ Round {round_num} validation miss")

                # Enhance profile for next round
                enhanced_profile = self._enhance_user_profile_with_realtime_sync(user_profile, validation_items, user_real_id)
                if enhanced_profile == "NO_VALIDATION_ANALYSIS_FOUND":
                    print(f"❌ No validation analysis found for user {user_real_id}, cannot enhance profile")
                    return False, user_profile  # Return original profile

                # Save enhanced profile for this round
                self._save_enhanced_realtime_profile_sync(enhanced_profile, user_real_id, round_num, hit_success)

                return False, enhanced_profile

        except Exception as e:
            print(f"⚠️  Error in Round {round_num} validation: {e}")
            return False, user_profile

    def _save_final_realtime_results(self, user_real_id: str, valid_records: List[Dict], test_records: List[Dict], initial_profile: str, final_profile: str, success: bool, final_round: int):
        """Save all final results to files"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Add any remaining validation records
        if hasattr(self, '_current_valid_records'):
            valid_records.extend(self._current_valid_records)
            delattr(self, '_current_valid_records')

        # Add any remaining test records
        if hasattr(self, '_current_test_records'):
            test_records.extend(self._current_test_records)
            delattr(self, '_current_test_records')

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
                "reflection_type": "validation"
            }

            valid_file = os.path.join(user_dir, "realtime_valid_reflection_results.json")
            with open(valid_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, ensure_ascii=False, indent=2)
            print(f"✅ Realtime validation reflection results saved to: {valid_file}")

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
                "reflection_type": "test"
            }

            test_file = os.path.join(user_dir, "realtime_test_reflection_results.json")
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"✅ Realtime test reflection results saved to: {test_file}")

        print(f"🎉 Iterative realtime reflection process completed: {'Success' if success else 'Failed'}")


    def _load_item_mapping(self) -> Dict[str, str]:
        """Load item mapping from item ID to title based on dataset type"""
        # Configuration for dataset mapping files
        dataset_mapping_config = {
            'douban': {'file': 'douban_mapping.json', 'name': 'Douban'},
            'bilibili': {'file': 'bilibili_mapping.json', 'name': 'Bilibili'},
            'bilibili_realtime': {'file': 'bilibili_mapping.json', 'name': 'Bilibili'},
            'redbook': {'file': 'redbook_mapping.json', 'name': 'Redbook'},
            'hupu': {'file': 'hupu_mapping.json', 'name': 'Hupu'}
        }

        config = dataset_mapping_config.get(self.dataset)
        if not config:
            print(f"⚠️  No mapping file configured for dataset: {self.dataset}")
            return {}

        mapping_file = config['file']
        mapping_name = config['name']

        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                print(f"✅ Loaded {len(mapping_data)} items from {mapping_name} mapping")
                return mapping_data
            except Exception as e:
                print(f"⚠️  Error loading {mapping_name} mapping: {e}")
                return {}
        else:
            print(f"⚠️  {mapping_name} mapping file not found: {mapping_file}")
            return {}

    def _get_item_title_from_dataset_files(self, item_id: str, user_real_id: str) -> str:
        """Get item title from dataset CSV files or download directory files"""
        # PRIORITY 1: Check download directory for media files (mp4, txt, etc.)
        # The filename (without extension) is often the title
        title = self._get_title_from_download_dir(item_id, user_real_id)
        if title:
            return title

        # PRIORITY 2: Check main dataset CSV
        main_dataset_dir = f"dataset/{self.dataset}/{user_real_id}"
        title = self._search_title_in_dataset_dir(main_dataset_dir, item_id)
        if title:
            return title

        # PRIORITY 3: Check realtime dataset CSV
        realtime_dataset_dir = f"dataset/{self.dataset}_realtime/{user_real_id}"
        title = self._search_title_in_dataset_dir(realtime_dataset_dir, item_id)
        if title:
            return title

        # PRIORITY 4: Fallback to video mapping file
        item_mapping = self._load_item_mapping()
        title_data = item_mapping.get(item_id, item_id)
        # Handle both dict format (with 'title' key) and string format
        if isinstance(title_data, dict):
            return title_data.get('title', item_id)
        return title_data

    def _get_title_from_download_dir(self, item_id: str, user_real_id: str) -> Optional[str]:
        """Get item title from download directory by looking at media file names"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Check all possible subdirectories
        subdirs = ['realtime', 'historical', 'recommended', 'validation', 'test']

        for subdir in subdirs:
            item_dir = os.path.join(user_dir, subdir, item_id)
            if os.path.exists(item_dir) and os.path.isdir(item_dir):
                try:
                    for filename in os.listdir(item_dir):
                        # Skip analysis.json and other non-media files
                        if filename == 'analysis.json':
                            continue

                        # Get title from media files (mp4, avi, mov, txt, jpg, png, etc.)
                        media_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.webp']
                        name, ext = os.path.splitext(filename)
                        if ext.lower() in media_extensions and name:
                            # Skip if filename is just the item_id
                            if name == item_id or name.startswith(f"{item_id}_"):
                                continue
                            return name
                except Exception as e:
                    print(f"⚠️  Error reading download dir for {item_id}: {e}")

        return None

    def _search_title_in_dataset_dir(self, dataset_dir: str, item_id: str) -> Optional[str]:
        """Search for item title in a dataset directory"""
        if not os.path.exists(dataset_dir):
            return None

        try:
            for csv_file in os.listdir(dataset_dir):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(dataset_dir, csv_file)
                    df = pd.read_csv(csv_path)

                    # Handle different dataset formats
                    if self.dataset == 'redbook':
                        # For redbook: redbookID -> title mapping
                        if 'redbookID' in df.columns and 'title' in df.columns:
                            matching_rows = df[df['redbookID'] == item_id]
                            if not matching_rows.empty:
                                return matching_rows.iloc[0]['title']
                    else:
                        # Handle bilibili format
                        if 'bvid' in df.columns and 'title' in df.columns:
                            matching_rows = df[df['bvid'] == item_id]
                            if not matching_rows.empty:
                                return matching_rows.iloc[0]['title']
                        elif 'item_id' in df.columns and 'title' in df.columns:
                            matching_rows = df[df['item_id'] == item_id]
                            if not matching_rows.empty:
                                return matching_rows.iloc[0]['title']
        except Exception as e:
            print(f"⚠️  Error searching title in {dataset_dir}: {e}")

        return None

    def _extract_predictions_from_response(self, response_text: str) -> List[str]:
        """Extract predictions from LLM response text"""
        predictions = []
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            # Look for numbered items (1. title, 2. title, etc.)
            if line and any(line.startswith(f"{i}.") for i in range(1, 21)):
                # Extract title after the number
                title = line.split('.', 1)[1].strip()
                if title:  # Only add non-empty predictions
                    predictions.append(title)

                if len(predictions) >= 20:
                    break

        return predictions

    def _get_user_analyzed_items(self, user_real_id: str) -> List[Dict]:
        """Get all items with analysis files from user's directories (historical + validation + test + recommended + realtime)"""
        all_items = []
        user_dir = os.path.join("download", self.dataset, user_real_id)

        if not os.path.exists(user_dir):
            print(f"⚠️  User directory not found: {user_dir}")
            return []

        # Define directories to check for analysis files
        analysis_dirs = {
            'historical': 'historical',
            'validation': 'validation',
            'test': 'test',
            'recommended': 'recommended',
            'realtime': 'realtime'
        }

        for dir_type, dir_name in analysis_dirs.items():
            dir_path = os.path.join(user_dir, dir_name)
            if os.path.exists(dir_path):
                for item_name in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item_name)
                    if os.path.isdir(item_path):
                        analysis_file = os.path.join(item_path, "analysis.json")
                        if os.path.exists(analysis_file):
                            # Get timestamp from analysis file or use default
                            timestamp = self._get_item_timestamp(analysis_file, item_name, user_real_id, dir_type)
                            all_items.append({
                                'item_id': item_name,
                                'timestamp': timestamp,
                                'source': dir_type,
                                'analysis_file': analysis_file
                            })

        return all_items

    def _get_item_timestamp(self, analysis_file: str, item_name: str, user_real_id: str, source: str) -> int:
        """Get timestamp for an item from dataset files or use default based on source"""
        # Try to get timestamp from dataset files first
        timestamp = self._get_timestamp_from_dataset(item_name, user_real_id)
        if timestamp is not None:
            return timestamp

        # Fallback: use source-based default timestamps to maintain order
        # Historical items get older timestamps, realtime items get newer timestamps
        source_timestamps = {
            'historical': 1000000000,  # Very old timestamp
            'validation': 1500000000,  # Medium timestamp
            'test': 1600000000,        # Newer timestamp
            'recommended': 1700000000, # Even newer timestamp
            'realtime': 1800000000     # Newest timestamp
        }

        base_timestamp = source_timestamps.get(source, 1000000000)
        # Add small random offset to avoid exact duplicates
        import random
        return base_timestamp + random.randint(0, 86400)  # Add up to 1 day

    def _get_timestamp_from_dataset(self, item_id: str, user_real_id: str) -> Optional[int]:
        """Get timestamp for an item from dataset CSV files"""
        # Check main dataset
        main_dataset_dir = f"dataset/{self.dataset}/{user_real_id}"
        timestamp = self._search_timestamp_in_dir(main_dataset_dir, item_id)
        if timestamp is not None:
            return timestamp

        # Check realtime dataset
        realtime_dataset_dir = f"dataset/{self.dataset}_realtime/{user_real_id}"
        timestamp = self._search_timestamp_in_dir(realtime_dataset_dir, item_id)
        if timestamp is not None:
            return timestamp

        return None

    def _search_timestamp_in_dir(self, dataset_dir: str, item_id: str) -> Optional[int]:
        """Search for item timestamp in a dataset directory"""
        if not os.path.exists(dataset_dir):
            return None

        try:
            for csv_file in os.listdir(dataset_dir):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(dataset_dir, csv_file)
                    df = pd.read_csv(csv_path)

                    # Handle different dataset formats
                    if self.dataset == 'redbook':
                        # For redbook: redbookID -> fav_time mapping
                        if 'redbookID' in df.columns:
                            matching_rows = df[df['redbookID'] == item_id]
                            if not matching_rows.empty:
                                # Use fav_time as primary timestamp
                                timestamp = matching_rows.iloc[0].get('fav_time', 0)
                                return int(timestamp)
                    else:
                        # Handle bilibili format
                        if 'bvid' in df.columns:
                            matching_rows = df[df['bvid'] == item_id]
                            if not matching_rows.empty:
                                # Use fav_time as primary timestamp
                                timestamp = matching_rows.iloc[0].get('fav_time',
                                           matching_rows.iloc[0].get('pubtime',
                                           matching_rows.iloc[0].get('ctime', 0)))
                                return int(timestamp)
                        elif 'item_id' in df.columns:
                            matching_rows = df[df['item_id'] == item_id]
                            if not matching_rows.empty:
                                timestamp = matching_rows.iloc[0].get('timestamp',
                                           matching_rows.iloc[0].get('fav_time', 0))
                                return int(timestamp)
        except Exception as e:
            print(f"⚠️  Error searching timestamp in {dataset_dir}: {e}")

        return None

    def _get_user_items_from_dataset(self, user_real_id: str) -> List[Dict]:
        """Get all items for a user from both dataset and realtime dataset files"""
        all_items = []

        # Load from main dataset (user-specific directory)
        main_dataset_dir = f"dataset/{self.dataset}/{user_real_id}"
        if os.path.exists(main_dataset_dir):
            try:
                for csv_file in os.listdir(main_dataset_dir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(main_dataset_dir, csv_file)
                        df = pd.read_csv(csv_path)

                        # Handle different dataset formats
                        if self.dataset == 'redbook':
                            # For redbook: redbookID -> fav_time mapping
                            if 'redbookID' in df.columns:
                                for _, row in df.iterrows():
                                    # Use fav_time as timestamp (when user favorited the item)
                                    timestamp = row.get('fav_time', 0)
                                    all_items.append({
                                        'item_id': row['redbookID'],
                                        'timestamp': int(timestamp),
                                        'source': 'main'
                                    })
                        else:
                            # Handle bilibili format
                            if 'bvid' in df.columns:
                                for _, row in df.iterrows():
                                    # Use fav_time as timestamp (when user favorited the item)
                                    timestamp = row.get('fav_time', row.get('pubtime', row.get('ctime', 0)))
                                    all_items.append({
                                        'item_id': row['bvid'],
                                        'timestamp': int(timestamp),
                                        'source': 'main'
                                    })
                            elif 'item_id' in df.columns:  # Generic format
                                for _, row in df.iterrows():
                                    timestamp = row.get('timestamp', row.get('fav_time', 0))
                                    all_items.append({
                                        'item_id': row['item_id'],
                                        'timestamp': int(timestamp),
                                        'source': 'main'
                                    })
            except Exception as e:
                print(f"⚠️  Error loading main dataset from {main_dataset_dir}: {e}")

        # Load from realtime dataset (user-specific directory)
        realtime_dataset_dir = f"dataset/{self.dataset}_realtime/{user_real_id}"
        if os.path.exists(realtime_dataset_dir):
            try:
                for csv_file in os.listdir(realtime_dataset_dir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(realtime_dataset_dir, csv_file)
                        df = pd.read_csv(csv_path)

                        # Handle different dataset formats
                        if self.dataset == 'redbook':
                            # For redbook: redbookID -> fav_time mapping
                            if 'redbookID' in df.columns:
                                for _, row in df.iterrows():
                                    # Use fav_time as timestamp (when user favorited the item)
                                    timestamp = row.get('fav_time', 0)
                                    all_items.append({
                                        'item_id': row['redbookID'],
                                        'timestamp': int(timestamp),
                                        'source': 'realtime'
                                    })
                        else:
                            # Handle bilibili format
                            if 'bvid' in df.columns:
                                for _, row in df.iterrows():
                                    # Use fav_time as timestamp (when user favorited the item)
                                    timestamp = row.get('fav_time', row.get('pubtime', row.get('ctime', 0)))
                                    all_items.append({
                                        'item_id': row['bvid'],
                                        'timestamp': int(timestamp),
                                        'source': 'realtime'
                                    })
                            elif 'item_id' in df.columns:  # Generic format
                                for _, row in df.iterrows():
                                    timestamp = row.get('timestamp', row.get('fav_time', 0))
                                    all_items.append({
                                        'item_id': row['item_id'],
                                        'timestamp': int(timestamp),
                                        'source': 'realtime'
                                    })
            except Exception as e:
                print(f"⚠️  Error loading realtime dataset from {realtime_dataset_dir}: {e}")

        print(f"📊 Found {len(all_items)} total items for user {user_real_id} from dataset files")
        return all_items

    def _get_user_all_items(self, user_real_id: str) -> List[str]:
        """Get all items that the user has interacted with (historical + recommended + realtime)"""
        user_dir = os.path.join("download", self.dataset, user_real_id)
        all_user_items = []

        # Get historical items
        historical_dir = os.path.join(user_dir, "historical")
        if os.path.exists(historical_dir):
            historical_items = [item for item in os.listdir(historical_dir)
                              if os.path.isdir(os.path.join(historical_dir, item))]
            all_user_items.extend(historical_items)

        # Get recommended items
        recommended_dir = os.path.join(user_dir, "recommended")
        if os.path.exists(recommended_dir):
            recommended_items = [item for item in os.listdir(recommended_dir)
                               if os.path.isdir(os.path.join(recommended_dir, item))]
            all_user_items.extend(recommended_items)

        # Get realtime items
        realtime_dir = os.path.join(user_dir, "realtime")
        if os.path.exists(realtime_dir):
            realtime_items = [item for item in os.listdir(realtime_dir)
                            if os.path.isdir(os.path.join(realtime_dir, item))]
            all_user_items.extend(realtime_items)

        # Remove duplicates
        all_user_items = list(set(all_user_items))
        return all_user_items


    def _enhance_user_profile_with_realtime_sync(self, current_profile: str, validation_item_names: List[str], user_real_id: str) -> str:
        """Enhance user profile using validation item analysis from analysis files"""

        # Get validation item analysis from analysis files
        validation_analysis = self._get_validation_item_analysis_from_files(validation_item_names, user_real_id)

        # Get historical item profiles for context
        historical_profiles = self._get_historical_item_profiles_from_files(user_real_id)

        enhancement_prompt = f"""
Based on the current user profile, historical interaction patterns, and analysis of new interaction items that the user actually engaged with, please generate an enhanced user profile that better captures the user's preferences.

Current User Profile:
{current_profile}

Historical Interaction Item Analysis (for reference):
{historical_profiles}

New Interaction Item Analysis:
{validation_analysis}

Please generate an enhanced user profile with the following requirements:
1. Integrate insights from new interaction items
2. Maintain consistency with historical preferences where appropriate
3. Identify any evolving interests or new preference patterns
4. Provide specific, actionable insights about user preferences
5. Comprehensive but concise (target 200-400 words)

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
            enhanced_profile = response.choices[0].message.content

            return enhanced_profile.strip()

        except Exception as e:
            print(f"❌ Error enhancing user profile with realtime data: {e}")
            return current_profile

    def _get_validation_item_analysis_from_files(self, item_names: List[str], user_real_id: str) -> str:
        """Get analysis for validation items from analysis files"""
        user_dir = os.path.join("download", self.dataset, user_real_id)
        analyses = []

        for item_name in item_names:
            # Search in all possible directories for the item
            search_dirs = ['validation', 'test', 'historical', 'recommended', 'realtime']
            analysis_found = False

            for search_dir in search_dirs:
                item_dir = os.path.join(user_dir, search_dir, item_name)
                analysis_file = os.path.join(item_dir, "analysis.json")

                if os.path.exists(analysis_file):
                    try:
                        analysis_data = self._load_analysis_file(analysis_file)
                        if analysis_data:
                            formatted_analysis = self._format_analysis_to_realtime_prompt(
                                analysis_data, item_name, f"Validation Item ({search_dir})", user_real_id
                            )
                            analyses.append(formatted_analysis)
                            analysis_found = True
                            break
                    except Exception as e:
                        print(f"⚠️  Error reading analysis for {item_name} in {search_dir}: {e}")

            if not analysis_found:
                print(f"⚠️  No analysis found for validation item: {item_name}")

        return "\n\n".join(analyses) if analyses else "No validation item analysis available."

    def _get_historical_item_profiles_from_files(self, user_real_id: str) -> str:
        """Get historical item profiles from analysis files"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Get all analyzed items
        all_items = self._get_user_analyzed_items(user_real_id)

        # Filter for historical items only
        historical_items = [item for item in all_items if item['source'] == 'historical']

        if not historical_items:
            return "No historical item profiles available."

        # Sort by timestamp (oldest first for chronological context)
        historical_items.sort(key=lambda x: x['timestamp'])

        analyses = []
        for item in historical_items:
            analysis_data = self._load_analysis_file(item['analysis_file'])
            if analysis_data:
                formatted_analysis = self._format_analysis_to_realtime_prompt(
                    analysis_data, item['item_id'], "Historical Interaction", user_real_id
                )
                analyses.append(formatted_analysis)

        return "\n\n".join(analyses) if analyses else "No historical item profiles available."

    def _get_realtime_item_analysis(self, item_names: List[str], user_real_id: str) -> str:
        """Get analysis for realtime items"""
        user_dir = os.path.join("download", self.dataset, user_real_id)
        realtime_dir = os.path.join(user_dir, "realtime")

        analyses = []

        for item_name in item_names:
            item_dir = os.path.join(realtime_dir, item_name)
            analysis_file = os.path.join(item_dir, "analysis.json")

            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)

                    # Format analysis data
                    analysis_text = f"Item: {item_name}\n"
                    if 'video_analysis' in analysis_data:
                        analysis_text += f"Video Analysis: {analysis_data['video_analysis']}\n"
                    if 'text_analysis' in analysis_data:
                        analysis_text += f"Text Analysis: {analysis_data['text_analysis']}\n"
                    if 'image_analysis' in analysis_data:
                        analysis_text += f"Image Analysis: {analysis_data['image_analysis']}\n"

                    analyses.append(analysis_text)

                except Exception as e:
                    print(f"⚠️  Error reading analysis for {item_name}: {e}")

        return "\n\n".join(analyses) if analyses else "No realtime item analysis available."

    def _get_historical_item_profiles(self, user_real_id: str) -> str:
        """Get historical item profiles for context"""
        user_dir = os.path.join("download", self.dataset, user_real_id)
        item_profiles_file = os.path.join(user_dir, "item_profiles.txt")

        if os.path.exists(item_profiles_file):
            try:
                with open(item_profiles_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️  Error reading item profiles: {e}")

        return "No historical item profiles available."

    def _get_random_items_with_titles(self, user_real_id: str, exclude_items: List[str], count: int = 99) -> List[Dict[str, str]]:
        """Get random items with titles for prediction task (excluding new test/valid items)"""
        print(f"🎲 Generating {count} random items (excluding {len(exclude_items)} test/valid items)")

        # Get all items from dataset files
        all_items = self._get_all_items_from_dataset()

        # Filter out exclude items (new test and valid items)
        available_items = [item for item in all_items if item['item_id'] not in exclude_items]

        if len(available_items) < count:
            count = len(available_items)

        selected_items = random.sample(available_items, count)

        items_with_titles = []
        if self.dataset == 'redbook':
            # For redbook, get titles from redbook_mapping.json
            video_mapping = self._load_item_mapping()
            for item in selected_items:
                title_data = video_mapping.get(item['item_id'], item['item_id'])  # Fallback to redbookID if not found
                # Handle both dict format (with 'title' key) and string format
                if isinstance(title_data, dict):
                    title = title_data.get('title', item['item_id'])
                else:
                    title = title_data
                items_with_titles.append({'redbookID': item['item_id'], 'title': title})
        else:
            # For bilibili, get titles from bilibili_mapping.json
            video_mapping = self._load_item_mapping()
            for item in selected_items:
                title_data = video_mapping.get(item['item_id'], item['item_id'])  # Fallback to bvid if not found
                # Handle both dict format (with 'title' key) and string format
                if isinstance(title_data, dict):
                    title = title_data.get('title', item['item_id'])
                else:
                    title = title_data
                items_with_titles.append({'bvid': item['item_id'], 'title': title})

        return items_with_titles

    def _get_all_items_from_dataset(self) -> List[Dict]:
        """Get all unique items from both main and realtime dataset files"""
        all_items = set()

        # Load from main dataset (all user directories)
        main_dataset_dir = f"dataset/{self.dataset}"
        if os.path.exists(main_dataset_dir):
            try:
                for user_dir in os.listdir(main_dataset_dir):
                    user_path = os.path.join(main_dataset_dir, user_dir)
                    if os.path.isdir(user_path):
                        for csv_file in os.listdir(user_path):
                            if csv_file.endswith('.csv'):
                                csv_path = os.path.join(user_path, csv_file)
                                df = pd.read_csv(csv_path)

                                # Handle different dataset formats
                                if self.dataset == 'redbook':
                                    if 'redbookID' in df.columns:
                                        for item_id in df['redbookID'].unique():
                                            all_items.add(item_id)
                                else:
                                    if 'item_id' in df.columns:
                                        for item_id in df['item_id'].unique():
                                            all_items.add(item_id)
                                    elif 'bvid' in df.columns:  # For bilibili format
                                        for bvid in df['bvid'].unique():
                                            all_items.add(bvid)
            except Exception as e:
                print(f"⚠️  Error loading main dataset for random items: {e}")

        # Load from realtime dataset (all user directories)
        realtime_dataset_dir = f"dataset/{self.dataset}_realtime"
        if os.path.exists(realtime_dataset_dir):
            try:
                for user_dir in os.listdir(realtime_dataset_dir):
                    user_path = os.path.join(realtime_dataset_dir, user_dir)
                    if os.path.isdir(user_path):
                        for csv_file in os.listdir(user_path):
                            if csv_file.endswith('.csv'):
                                csv_path = os.path.join(user_path, csv_file)
                                df = pd.read_csv(csv_path)

                                # Handle different dataset formats
                                if self.dataset == 'redbook':
                                    if 'redbookID' in df.columns:
                                        for item_id in df['redbookID'].unique():
                                            all_items.add(item_id)
                                else:
                                    if 'item_id' in df.columns:
                                        for item_id in df['item_id'].unique():
                                            all_items.add(item_id)
                                    elif 'bvid' in df.columns:  # For bilibili format
                                        for bvid in df['bvid'].unique():
                                            all_items.add(bvid)
            except Exception as e:
                print(f"⚠️  Error loading realtime dataset for random items: {e}")

        # Convert to list of dicts (removed duplicate logging)
        return [{'item_id': item_id} for item_id in all_items]

    def _get_item_title(self, item_id: str, user_real_id: str) -> str:
        """Get item title from analysis files"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Check in realtime folder first
        realtime_item_dir = os.path.join(user_dir, "realtime", item_id)
        if os.path.exists(realtime_item_dir):
            # Look for video files or analysis files to get title
            for file in os.listdir(realtime_item_dir):
                if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.mov'):
                    return os.path.splitext(file)[0]

        # Check in historical folder
        historical_item_dir = os.path.join(user_dir, "historical", item_id)
        if os.path.exists(historical_item_dir):
            for file in os.listdir(historical_item_dir):
                if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.mov'):
                    return os.path.splitext(file)[0]

        # Fallback to item_id
        return item_id

    def _get_item_title_from_dataset(self, item_id: str) -> Optional[str]:
        """Get item title from dataset files (fallback method)"""
        # This is a placeholder - implement based on your dataset structure
        return None

    def _save_enhanced_realtime_profile_sync(self, enhanced_profile: str, user_real_id: str, round_num: int, hit_success: bool):
        """Save enhanced realtime user profile for each round"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Create enhanced_realtime_user_profiles directory if it doesn't exist
        enhanced_profiles_dir = os.path.join(user_dir, "enhanced_realtime_user_profiles")
        os.makedirs(enhanced_profiles_dir, exist_ok=True)

        # Create filename with round number and hit status
        hit_status = "hit" if hit_success else "miss"
        profile_filename = f"round_{round_num}_{hit_status}_realtime_profile.txt"
        profile_path = os.path.join(enhanced_profiles_dir, profile_filename)

        # Save the enhanced profile
        with open(profile_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_profile)

        print(f"💾 Enhanced realtime profile saved: {profile_filename}")

        # Also update the main enhanced_realtime_user_profile.txt with the latest version
        main_profile_path = os.path.join(user_dir, "enhanced_realtime_user_profile.txt")
        with open(main_profile_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_profile)

    def _save_initial_realtime_profile_sync(self, initial_profile: str, user_real_id: str):
        """Save initial realtime user profile as Round 0"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Create enhanced_realtime_user_profiles directory if it doesn't exist
        enhanced_profiles_dir = os.path.join(user_dir, "enhanced_realtime_user_profiles")
        os.makedirs(enhanced_profiles_dir, exist_ok=True)

        # Save initial profile as round 0
        profile_filename = "round_0_initial_realtime_profile.txt"
        profile_path = os.path.join(enhanced_profiles_dir, profile_filename)

        with open(profile_path, 'w', encoding='utf-8') as f:
            f.write(initial_profile)

        print(f"💾 Initial realtime profile saved: {profile_filename}")

    def _save_realtime_reflection_data_sync(self, user_real_id: str, random_items_with_titles: List[Dict[str, str]], validation_items: List[str], test_items: List[str]):
        """Save realtime reflection data for analysis"""
        user_dir = os.path.join("download", self.dataset, user_real_id)

        # Convert validation and test items to include titles
        validation_items_with_titles = []
        for item_id in validation_items:
            title = self._get_item_title_from_dataset_files(item_id, user_real_id)
            if self.dataset == 'redbook':
                validation_items_with_titles.append({
                    'redbookID': item_id,
                    'title': title
                })
            else:
                validation_items_with_titles.append({
                    'bvid': item_id,
                    'title': title
                })

        test_items_with_titles = []
        for item_id in test_items:
            title = self._get_item_title_from_dataset_files(item_id, user_real_id)
            if self.dataset == 'redbook':
                test_items_with_titles.append({
                    'redbookID': item_id,
                    'title': title
                })
            else:
                test_items_with_titles.append({
                    'bvid': item_id,
                    'title': title
                })

        reflection_data = {
            'user_id': user_real_id,
            'random_items': random_items_with_titles,
            'validation_items': validation_items_with_titles,
            'test_items': test_items_with_titles,
            'total_random_items': len(random_items_with_titles),
            'total_validation_items': len(validation_items),
            'total_test_items': len(test_items)
        }

        reflection_file = os.path.join(user_dir, "realtime_reflection_data.json")
        with open(reflection_file, 'w', encoding='utf-8') as f:
            json.dump(reflection_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Realtime reflection data saved to: realtime_reflection_data.json")
        print(f"   - Random items: {len(random_items_with_titles)}")
        print(f"   - Validation items: {len(validation_items)}")
        print(f"   - Test items: {len(test_items)}")


def _process_single_realtime_user_sync(analyzer: EnhancedRealtimeAnalyzer, user_folder: str) -> Dict:
    """
    Process a single user with enhanced realtime analysis

    Args:
        analyzer: EnhancedRealtimeAnalyzer instance
        user_folder: User folder name
        semaphore: Semaphore to control concurrency

    Returns:
        Dict with processing results
    """
    try:
        dataset_path = os.path.join("download", analyzer.dataset)
        user_dir = os.path.join(dataset_path, user_folder)

        print(f"\n🏠 [CONCURRENT] Processing realtime user: {user_folder}")

        # No need for internal ID mapping - we work directly with real user IDs
        print(f"📝 Processing user with real ID: {user_folder}")

        # 1. Collect all items from historical + recommended + realtime
        all_items = []

        # Historical items
        historical_dir = os.path.join(user_dir, "historical")
        if os.path.exists(historical_dir):
            historical_items = [item for item in os.listdir(historical_dir)
                              if os.path.isdir(os.path.join(historical_dir, item))]
            all_items.extend(historical_items)
            print(f"📚 Found {len(historical_items)} historical items")

        # Recommended items
        recommended_dir = os.path.join(user_dir, "recommended")
        if os.path.exists(recommended_dir):
            recommended_items = [item for item in os.listdir(recommended_dir)
                               if os.path.isdir(os.path.join(recommended_dir, item))]
            all_items.extend(recommended_items)
            print(f"🎯 Found {len(recommended_items)} recommended items")

            # Analyze recommendation hits (same as enhanced_analyze)
            hit_analysis = analyzer.check_recommendation_hits(user_folder, recommended_items)
            print(f"📊 Recommendation Hit Analysis for {user_folder}:")
            print(f"   - Total recommendations: {hit_analysis['total_recommendations']}")
            print(f"   - Test items: {len(hit_analysis['test_items'])}")
            print(f"   - Hits: {hit_analysis['hit_count']}")
            print(f"   - Hit rate: {hit_analysis['hit_rate']:.2%}")
            print(f"   - Hit items: {hit_analysis['hits']}")

        # Realtime items
        realtime_dir = os.path.join(user_dir, "realtime")
        if os.path.exists(realtime_dir):
            realtime_items = [item for item in os.listdir(realtime_dir)
                            if os.path.isdir(os.path.join(realtime_dir, item))]
            all_items.extend(realtime_items)
            print(f"⚡ Found {len(realtime_items)} realtime items")
        else:
            print(f"⚠️  No realtime directory found for user {user_folder}")
            return {'user': user_folder, 'status': 'failed', 'reason': 'no_realtime_dir'}

        print(f"📊 Total items for {user_folder}: {len(all_items)} (historical + recommended + realtime)")

        # 2. Generate realtime user profile from analysis files
        print(f"📝 Generating realtime user profile for {user_folder}...")
        current_user_profile = analyzer.generate_realtime_user_profile(user_folder)

        if not current_user_profile:
            print(f"⚠️  Failed to generate realtime user profile for {user_folder}")
            return {'user': user_folder, 'status': 'failed', 'reason': 'no_realtime_user_profile'}

        # 3. Start iterative realtime reflection process
        validation_items = analyzer.get_user_validation_items(user_folder)
        test_items = analyzer.get_user_test_items(user_folder)

        if validation_items and test_items:
            print(f"🔄 Starting realtime reflection for {user_folder}...")
            reflection_success, final_profile = analyzer.perform_iterative_realtime_reflection_sync(
                current_user_profile, user_folder, max_reflections=3
            )

            if reflection_success:
                print(f"✅ Realtime reflection completed successfully for {user_folder}")
                return {'user': user_folder, 'status': 'success', 'reflection_success': True}
            else:
                print(f"⚠️  Realtime reflection completed with issues for {user_folder}")
                return {'user': user_folder, 'status': 'completed_with_issues', 'reflection_success': False}
        else:
            print(f"⚠️  Insufficient validation/test data for realtime reflection for {user_folder}")
            return {'user': user_folder, 'status': 'skipped', 'reason': 'insufficient_validation_test_data'}

    except Exception as e:
        print(f"❌ Error processing realtime user {user_folder}: {e}")
        return {'user': user_folder, 'status': 'error', 'error': str(e)}


def _collect_realtime_global_statistics(dataset_path: str, user_folders: List[str]) -> Dict:
    """Collect global statistics from all users' realtime reflection results"""
    global_stats = {
        "validation_stats": {"all_records": []},
        "test_stats": {"all_records": []},
        "user_count": 0,
        "successful_users": 0,
        "total_test_users": 0
    }

    for user_folder in user_folders:
        user_dir = os.path.join(dataset_path, user_folder)
        has_test_records = False

        # Load validation reflection results
        valid_file = os.path.join(user_dir, "realtime_valid_reflection_results.json")
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
        test_file = os.path.join(user_dir, "realtime_test_reflection_results.json")
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


def _calculate_realtime_global_reflection_statistics(global_stats: Dict) -> Dict:
    """Calculate cumulative global NDCG and Hit Rate statistics across all users for realtime analysis.

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
    cumulative_validation_hit_1 = set()
    cumulative_validation_hit_5 = set()
    cumulative_validation_hit_10 = set()
    cumulative_validation_hit_20 = set()
    cumulative_validation_ndcg_1 = {}  # user_id -> latest ndcg value
    cumulative_validation_ndcg_5 = {}
    cumulative_validation_ndcg_10 = {}
    cumulative_validation_ndcg_20 = {}

    # Calculate cumulative statistics for each round
    for target_round in validation_rounds:
        # Process records for this round
        if target_round in validation_by_round:
            for record in validation_by_round[target_round]:
                user_id = record.get('user_id', 'unknown')

                # Update NDCG (always use latest value for each user)
                cumulative_validation_ndcg_1[user_id] = record.get('ndcg_at_1', 0)
                cumulative_validation_ndcg_5[user_id] = record.get('ndcg_at_5', 0)
                cumulative_validation_ndcg_10[user_id] = record.get('ndcg_at_10', 0)
                cumulative_validation_ndcg_20[user_id] = record.get('ndcg_at_20', 0)

                # Update hit status (once hit, always counted as hit)
                if record.get('hit_rate_at_1', 0) > 0:
                    cumulative_validation_hit_1.add(user_id)
                if record.get('hit_rate_at_5', 0) > 0:
                    cumulative_validation_hit_5.add(user_id)
                if record.get('hit_rate_at_10', 0) > 0:
                    cumulative_validation_hit_10.add(user_id)
                if record.get('hit_rate_at_20', 0) > 0:
                    cumulative_validation_hit_20.add(user_id)

        hit_users_1 = len(cumulative_validation_hit_1)
        hit_users_5 = len(cumulative_validation_hit_5)
        hit_users_10 = len(cumulative_validation_hit_10)
        hit_users_20 = len(cumulative_validation_hit_20)

        # Calculate NDCG average across ALL users (users without records count as 0)
        avg_ndcg_1 = sum(cumulative_validation_ndcg_1.values()) / total_users
        avg_ndcg_5 = sum(cumulative_validation_ndcg_5.values()) / total_users
        avg_ndcg_10 = sum(cumulative_validation_ndcg_10.values()) / total_users
        avg_ndcg_20 = sum(cumulative_validation_ndcg_20.values()) / total_users

        stats["validation_global"][f"round_{target_round}"] = {
            "avg_ndcg_at_1": avg_ndcg_1,
            "avg_ndcg_at_5": avg_ndcg_5,
            "avg_ndcg_at_10": avg_ndcg_10,
            "avg_ndcg_at_20": avg_ndcg_20,
            "hit_rate_at_1": hit_users_1 / total_users,
            "hit_rate_at_5": hit_users_5 / total_users,
            "hit_rate_at_10": hit_users_10 / total_users,
            "hit_rate_at_20": hit_users_20 / total_users,
            "hit_users_1": hit_users_1,
            "hit_users_5": hit_users_5,
            "hit_users_10": hit_users_10,
            "hit_users_20": hit_users_20,
            "total_users": total_users,
            "users_with_records": len(cumulative_validation_ndcg_1)
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
    cumulative_test_hit_1 = set()
    cumulative_test_hit_5 = set()
    cumulative_test_hit_10 = set()
    cumulative_test_hit_20 = set()
    cumulative_test_ndcg_1 = {}  # user_id -> latest ndcg value
    cumulative_test_ndcg_5 = {}
    cumulative_test_ndcg_10 = {}
    cumulative_test_ndcg_20 = {}

    # Calculate cumulative statistics for each round
    for target_round in test_rounds:
        # Process records for this round
        if target_round in test_by_round:
            for record in test_by_round[target_round]:
                user_id = record.get('user_id', 'unknown')

                # Update NDCG (always use latest value for each user)
                cumulative_test_ndcg_1[user_id] = record.get('ndcg_at_1', 0)
                cumulative_test_ndcg_5[user_id] = record.get('ndcg_at_5', 0)
                cumulative_test_ndcg_10[user_id] = record.get('ndcg_at_10', 0)
                cumulative_test_ndcg_20[user_id] = record.get('ndcg_at_20', 0)

                # Update hit status (once hit, always counted as hit)
                if record.get('hit_rate_at_1', 0) > 0:
                    cumulative_test_hit_1.add(user_id)
                if record.get('hit_rate_at_5', 0) > 0:
                    cumulative_test_hit_5.add(user_id)
                if record.get('hit_rate_at_10', 0) > 0:
                    cumulative_test_hit_10.add(user_id)
                if record.get('hit_rate_at_20', 0) > 0:
                    cumulative_test_hit_20.add(user_id)

        hit_users_1 = len(cumulative_test_hit_1)
        hit_users_5 = len(cumulative_test_hit_5)
        hit_users_10 = len(cumulative_test_hit_10)
        hit_users_20 = len(cumulative_test_hit_20)

        # Calculate NDCG average across ALL users (users without records count as 0)
        avg_ndcg_1 = sum(cumulative_test_ndcg_1.values()) / total_users
        avg_ndcg_5 = sum(cumulative_test_ndcg_5.values()) / total_users
        avg_ndcg_10 = sum(cumulative_test_ndcg_10.values()) / total_users
        avg_ndcg_20 = sum(cumulative_test_ndcg_20.values()) / total_users

        stats["test_global"][f"round_{target_round}"] = {
            "avg_ndcg_at_1": avg_ndcg_1,
            "avg_ndcg_at_5": avg_ndcg_5,
            "avg_ndcg_at_10": avg_ndcg_10,
            "avg_ndcg_at_20": avg_ndcg_20,
            "hit_rate_at_1": hit_users_1 / total_users,
            "hit_rate_at_5": hit_users_5 / total_users,
            "hit_rate_at_10": hit_users_10 / total_users,
            "hit_rate_at_20": hit_users_20 / total_users,
            "hit_users_1": hit_users_1,
            "hit_users_5": hit_users_5,
            "hit_users_10": hit_users_10,
            "hit_users_20": hit_users_20,
            "total_users": total_users,
            "users_with_records": len(cumulative_test_ndcg_1)
        }

    return stats


def enhanced_realtime_analyze(max_concurrent_users: int = 4):
    """
    Enhanced realtime analysis function with hit analysis and self-reflection using ThreadPoolExecutor

    Args:
        max_concurrent_users: Maximum number of users to process concurrently
    """
    print("🚀 Starting enhanced realtime analysis with ThreadPoolExecutor...")
    print("=" * 70)

    analyzer = EnhancedRealtimeAnalyzer()
    dataset_path = os.path.join("download", analyzer.dataset)

    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset path not found: {dataset_path}")
        return

    # 1. Collect all user folders that have realtime data
    user_folders = []
    for user_folder in os.listdir(dataset_path):
        user_dir = os.path.join(dataset_path, user_folder)
        if os.path.isdir(user_dir):
            # Check if user has realtime data
            realtime_dir = os.path.join(user_dir, "realtime")
            if os.path.exists(realtime_dir) and os.listdir(realtime_dir):
                user_folders.append(user_folder)

    if not user_folders:
        print("❌ No user folders with realtime data found")
        return

    print(f"📊 Found {len(user_folders)} users with realtime data to process")
    print(f"🔄 Using {max_concurrent_users} concurrent workers")

    # 2. Process all users concurrently using ThreadPoolExecutor
    print(f"\n🔄 Processing all realtime users concurrently...")
    print(f"🚀 Starting {len(user_folders)} concurrent tasks with max {max_concurrent_users} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent_users) as executor:
        # Submit all tasks
        future_to_user = {
            executor.submit(_process_single_realtime_user_sync, analyzer, user_folder): user_folder
            for user_folder in user_folders
        }

        completed_count = 0
        for future in as_completed(future_to_user):
            user_folder = future_to_user[future]
            try:
                result = future.result()
                completed_count += 1
                results.append(result)

                print(f"✅ [{completed_count}/{len(user_folders)}] Completed realtime processing for user: {result['user']} (Status: {result['status']})")

            except Exception as e:
                completed_count += 1
                print(f"❌ [{completed_count}/{len(user_folders)}] Error processing user {user_folder}: {e}")
                results.append({'user': user_folder, 'status': 'error', 'error': str(e)})

    # 3. Summary
    print(f"\n📊 Enhanced Realtime Analysis Summary:")
    print(f"=" * 50)

    success_count = len([r for r in results if r['status'] == 'success'])
    completed_with_issues = len([r for r in results if r['status'] == 'completed_with_issues'])
    failed_count = len([r for r in results if r['status'] in ['failed', 'error']])
    skipped_count = len([r for r in results if r['status'] == 'skipped'])

    print(f"✅ Successful: {success_count}")
    print(f"⚠️  Completed with issues: {completed_with_issues}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"📈 Total processed: {len(results)}")

    if failed_count > 0:
        print(f"\n❌ Failed users:")
        for result in results:
            if result['status'] in ['failed', 'error']:
                reason = result.get('reason', result.get('error', 'Unknown'))
                print(f"   - {result['user']}: {reason}")

    # 4. Calculate and display global statistics
    completed_users = [result for result in results if result['status'] == 'success']
    if completed_users:
        print(f"\n🌍 计算全局实时统计数据...")

        # Collect global statistics from all users
        global_stats = _collect_realtime_global_statistics(dataset_path, [r['user'] for r in completed_users])

        if global_stats["validation_stats"]["all_records"] or global_stats["test_stats"]["all_records"]:
            print(f"\n📊 全局实时反思统计结果 (基于 {global_stats['successful_users']} 个用户):")

            # Calculate global statistics
            global_reflection_stats = _calculate_realtime_global_reflection_statistics(global_stats)

            # Display global validation statistics by round (cumulative)
            if global_reflection_stats["validation_global"]:
                print(f"🔍 全局验证阶段统计 (累积):")
                for round_key, round_stats in sorted(global_reflection_stats["validation_global"].items()):
                    round_num = round_key.replace("round_", "")
                    print(f"   Round {round_num}: NDCG@1={round_stats['avg_ndcg_at_1']:.3f}, NDCG@5={round_stats['avg_ndcg_at_5']:.3f}, NDCG@10={round_stats['avg_ndcg_at_10']:.3f}, NDCG@20={round_stats['avg_ndcg_at_20']:.3f}")
                    print(f"            HR@1={round_stats['hit_rate_at_1']:.3f} ({round_stats['hit_users_1']}/{round_stats['total_users']}), HR@5={round_stats['hit_rate_at_5']:.3f} ({round_stats['hit_users_5']}/{round_stats['total_users']}), HR@10={round_stats['hit_rate_at_10']:.3f} ({round_stats['hit_users_10']}/{round_stats['total_users']}), HR@20={round_stats['hit_rate_at_20']:.3f} ({round_stats['hit_users_20']}/{round_stats['total_users']})")

            # Display global test statistics by round (cumulative)
            if global_reflection_stats["test_global"]:
                print(f"🧪 全局测试阶段统计 (累积):")
                for round_key, round_stats in sorted(global_reflection_stats["test_global"].items()):
                    round_num = round_key.replace("round_", "")
                    print(f"   Round {round_num}: NDCG@1={round_stats['avg_ndcg_at_1']:.3f}, NDCG@5={round_stats['avg_ndcg_at_5']:.3f}, NDCG@10={round_stats['avg_ndcg_at_10']:.3f}, NDCG@20={round_stats['avg_ndcg_at_20']:.3f}")
                    print(f"            HR@1={round_stats['hit_rate_at_1']:.3f} ({round_stats['hit_users_1']}/{round_stats['total_users']}), HR@5={round_stats['hit_rate_at_5']:.3f} ({round_stats['hit_users_5']}/{round_stats['total_users']}), HR@10={round_stats['hit_rate_at_10']:.3f} ({round_stats['hit_users_10']}/{round_stats['total_users']}), HR@20={round_stats['hit_rate_at_20']:.3f} ({round_stats['hit_users_20']}/{round_stats['total_users']})")

            # Save global statistics to file
            global_stats_file = os.path.join(dataset_path, "global_realtime_reflection_statistics.json")
            global_stats_data = {
                "dataset": analyzer.dataset,
                "total_users_processed": len(user_folders),
                "successful_users": global_stats["successful_users"],
                "global_statistics": global_reflection_stats,
                "timestamp": pd.Timestamp.now().isoformat()
            }

            with open(global_stats_file, 'w', encoding='utf-8') as f:
                json.dump(global_stats_data, f, ensure_ascii=False, indent=2)
            print(f"💾 全局实时统计数据已保存到: {global_stats_file}")
        else:
            print(f"⚠️  未找到有效的实时反思统计数据")
    else:
        print(f"⚠️  没有成功完成的用户，无法计算全局统计")

    print("=" * 70)
    print(f"🎉 Enhanced realtime analysis completed!")
