import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from user_profile_generator import UserProfileGenerator
from commentproduct_generator import CommentProductGenerator
from hupu_english import HupuEnglishConverter

class HupuProcessor:
    def __init__(self, data_root="download/hupu", output_base="generated_hupu_posts"):
        self.data_root = data_root
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)

    def setup_environment(self):
        """Setup environment variables for different API models"""
        os.environ.update({
            "CHAT_API_KEY": "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG",
            "CHAT_BASE_URL": "https://yunwu.ai/v1",
            "CHAT_MODEL": "gemini-2.5-pro",

            "IMAGE_API_KEY": "sk-W7qxtvbQUxwIo9PLlSGh89cUKKuTTo1oUXmqpGoYIhqQULjI",
            "IMAGE_BASE_URL": "https://yunwu.ai/v1",
            "IMAGE_MODEL": "gemini-2.5-pro",

            "GENERATE_API_KEY": "sk-W7qxtvbQUxwIo9PLlSGh89cUKKuTTo1oUXmqpGoYIhqQULjI",
            "GENERATE_BASE_URL": "https://yunwu.ai/v1beta",
            "GENERATE_MODEL": "gemini-2.5-flash-image-preview",

            # Search model for hot topic discovery
            "SEARCH_MODEL": "gpt-5-all",
            "SEARCH_API_KEY": "sk-dVaSXmTEMBh0Gygx49ResSvaONvErml5QV8McBAGkbPmX2mG",
            "SEARCH_BASE_URL": "https://yunwu.ai/v1",
        })

    def get_available_users(self) -> List[str]:
        """Get list of available users from data root directory"""
        users = []
        if os.path.exists(self.data_root):
            for name in os.listdir(self.data_root):
                user_path = os.path.join(self.data_root, name)
                if os.path.isdir(user_path):
                    profile_path = os.path.join(user_path, "user_profile.txt")
                    if os.path.exists(profile_path):
                        users.append(name)
        return sorted(users)

    def _extract_first_preference(self, user_profile_text: str) -> str:
        """Extract the first preference from user profile text.
        
        The user profile format may vary, but always contains:
        "Ordering by user preference level, from highest to lowest:"
        
        Returns the first preference section with the marker line included.
        """
        # Find the marker line (may have ** markdown formatting)
        lines = user_profile_text.split('\n')
        marker_idx = -1
        marker_line = None
        
        for i, line in enumerate(lines):
            # Check if line contains the ordering marker (with or without markdown)
            if "Ordering by user preference level, from highest to lowest" in line:
                marker_idx = i
                marker_line = line
                break
        
        if marker_idx == -1:
            # If marker not found, return original text
            print("⚠️  Preference marker not found, using full user profile")
            return user_profile_text
        
        # Extract first preference
        # Look for the start of the first preference (usually "## 1." or "1. Preference 1:" or similar)
        first_pref_start = -1
        for i in range(marker_idx + 1, len(lines)):
            line = lines[i].strip()
            # Skip empty lines
            if not line:
                continue
            # Check if this is the start of first preference
            # Pattern: "## 1.", "1. Preference 1:", "1. ", etc.
            if (line.startswith("## 1.") or 
                line.startswith("**## 1.") or
                (line.startswith("1.") and ("Preference 1" in line or len(line) > 3)) or
                line.startswith("1. Preference 1")):
                first_pref_start = i
                break
        
        if first_pref_start == -1:
            # If can't find first preference start, return from marker onwards
            first_pref_start = marker_idx + 1
        
        # Find the end of first preference (start of second preference or end of text)
        first_pref_end = len(lines)
        for i in range(first_pref_start + 1, len(lines)):
            line = lines[i].strip()
            # Check if this is the start of second preference
            if (line.startswith("## 2.") or 
                line.startswith("**## 2.") or
                (line.startswith("2.") and ("Preference 2" in line or len(line) > 3)) or
                line.startswith("2. Preference 2")):
                first_pref_end = i
                break
        
        # Extract the first preference section, including the marker line
        first_preference_lines = [marker_line] + lines[first_pref_start:first_pref_end]
        first_preference = '\n'.join(first_preference_lines).strip()
        
        if not first_preference:
            print("⚠️  Failed to extract first preference, using full user profile")
            return user_profile_text
        
        print(f"✅ Extracted Preference 1 (length: {len(first_preference)} chars)")
        return first_preference

    def load_user_profile(self, user_id: str) -> Optional[Dict]:
        """Load user profile from user_profile.txt"""
        user_path = os.path.join(self.data_root, user_id)
        profile_path = os.path.join(user_path, "user_profile.txt")

        if not os.path.exists(profile_path):
            print(f"❌ User profile not found: {profile_path}")
            return None

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_text = f.read().strip()
            
            print(f"✅ Successfully loaded user profile (length: {len(profile_text)} chars)")
            
            # Extract first preference only
            first_preference = self._extract_first_preference(profile_text)
            
            return {
                "user_id": user_id,
                "profile_text": first_preference,
                "source": profile_path,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error loading profile: {e}")
            return None

    def process_user(self, user_id: str, max_workers: int = 4, user_index: Optional[int] = None, generate_english: bool = True) -> Optional[Dict]:
        """Process single user and generate discussion post based on hot topics"""
        print(f"\n{'='*50}")
        print(f"Processing user [{user_index}] {user_id}" if user_index is not None else f"Processing user {user_id}")
        print(f"{'='*50}")

        # Create output directory (with index)
        if user_index is not None:
            output_dir_name = f"{user_index}_{user_id}"
        else:
            output_dir_name = user_id
        user_output_dir = os.path.join(self.output_base, output_dir_name)
        os.makedirs(user_output_dir, exist_ok=True)

        # Resume from checkpoint: Skip if final_results.json already exists
        final_results_path = os.path.join(user_output_dir, "final_results.json")
        if os.path.exists(final_results_path):
            try:
                # Verify file is valid (contains required fields)
                with open(final_results_path, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                    if existing_result.get("personalized_post") or existing_result.get("discussion_post"):
                        print(f"✅ Detected existing final_results.json, skipping processing (resume from checkpoint)")
                        print(f"   📄 File path: {final_results_path}")
                        return existing_result
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️  Detected final_results.json but parsing failed: {e}, will regenerate")
                # Continue processing, overwrite old file

        try:
            # Load user profile from user_profile.txt
            print("Loading existing user profile...")
            user_profile = self.load_user_profile(user_id)
            
            if not user_profile:
                print(f"User {user_id} has no valid profile, skipping")
                return None

            # Save profile as JSON
            profile_path = os.path.join(user_output_dir, "profile.json")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(user_profile, f, ensure_ascii=False, indent=2)
            
            print(f"   Profile saved to: {profile_path}")

            # Generate discussion post based on hot topics
            print("Generating hot-topic-based discussion post...")
            content_generator = CommentProductGenerator()
            discussion_post = content_generator(profile_path, user_output_dir)

            # Save intermediate results (without english_post) for English converter to read
            intermediate_result = {
                "user_id": user_id,
                "user_profile": user_profile,
                "discussion_post": discussion_post
            }
            
            with open(os.path.join(user_output_dir, "final_results.json"), "w", encoding="utf-8") as f:
                json.dump(intermediate_result, f, ensure_ascii=False, indent=2)

            # ===== Auto-generate English version =====
            english_post = None
            if generate_english:
                try:
                    print("\n🌍 Generating English version (forum style)...")
                    
                    # Create English converter
                    english_converter = HupuEnglishConverter(generated_dir=self.output_base)
                    
                    # Convert to English
                    user_dir_name = output_dir_name  # Use previously defined directory name
                    english_post = english_converter.convert_post(user_dir_name)
                    
                    if english_post:
                        print(f"✅ English version generated successfully!")
                        print(f"📄 English HTML: {english_post['html_path']}")
                    else:
                        print(f"⚠️ English version generation failed, but Chinese version is complete")
                
                except Exception as e:
                    print(f"⚠️ Error generating English version: {e}")
                    print(f"Chinese version is complete, you can convert manually later")
                    import traceback
                    traceback.print_exc()

            # Save final results (update with english_post)
            final_result = {
                "user_id": user_id,
                "user_profile": user_profile,
                "discussion_post": discussion_post,
                "english_post": english_post,
                "generation_mode": discussion_post.get("mode", "unknown")
            }

            with open(os.path.join(user_output_dir, "final_results.json"), "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)

            img_count = len(discussion_post.get('images', []))
            link_count = len(discussion_post.get('links', []))
            
            print(f"\n✅ User {user_id} processing complete!")
            
            # Display generation mode
            if discussion_post.get("mode") == "fallback":
                print(f"📋 Generation mode: Based on user profile (no real-time hot topics)")
            else:
                print(f"📋 Generation mode: Based on real-time hot topics")
            
            print(f"📝 Post text: {len(discussion_post['text'])} chars")
            print(f"🖼️  Generated images: {img_count} images")
            print(f"🔗 Related links: {link_count} links")
            print(f"📄 HTML file: {discussion_post['html_post']}")
            if english_post:
                print(f"🌍 English HTML: {english_post['html_path']}")
            
            # Clean up embedding cache for this user
            try:
                content_generator.rag_helper.clear_user_cache("hupu", user_id)
            except Exception as e:
                print(f"⚠️  Failed to clear embedding cache: {e}")

            return final_result

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_users(self, user_ids: Optional[List[str]] = None, max_workers: int = 2, start_index: int = 0, generate_english: bool = True, parallel_users: int = 40) -> Dict:
        """Process specified users (or all if None)
        
        Args:
            user_ids: List of user IDs to process
            max_workers: Number of parallel worker threads for single user (for image generation, etc.)
            start_index: Starting index for display and directory naming
            generate_english: Whether to automatically generate English version (default True)
            parallel_users: Number of users to process in parallel (default 40)
        """
        if user_ids is None:
            user_ids = self.get_available_users()
            print(f"Auto-detected {len(user_ids)} users: {user_ids[:5]}{'...' if len(user_ids) > 5 else ''}")

        results = {}
        all_available_users = self.get_available_users()
        
        print(f"\n{'='*60}")
        print(f"🚀 Parallel processing mode: Up to {parallel_users} users simultaneously")
        print(f"   Internal concurrency per user: {max_workers}")
        print(f"{'='*60}\n")
        
        # Use ThreadPoolExecutor for parallel user processing
        with ThreadPoolExecutor(max_workers=parallel_users) as executor:
            # Submit all tasks
            future_to_user = {}
            for user_id in user_ids:
                # Get user's index in the complete user list
                if user_id in all_available_users:
                    user_index = all_available_users.index(user_id)
                else:
                    user_index = None
                
                future = executor.submit(
                    self._process_user_wrapper,
                    user_id,
                    max_workers,
                    user_index,
                    generate_english
                )
                future_to_user[future] = user_id
            
            # Process completed tasks
            completed_count = 0
            total_count = len(user_ids)
            
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    if result:
                        results[user_id] = result
                        print(f"\n✅ [{completed_count}/{total_count}] User {user_id} processing complete")
                    else:
                        print(f"\n⚠️  [{completed_count}/{total_count}] User {user_id} produced no results")
                except Exception as e:
                    print(f"\n❌ [{completed_count}/{total_count}] Error processing user {user_id}: {e}")
                    import traceback
                    traceback.print_exc()

        # Generate summary report
        if results:
            self._generate_summary(results)

        return results
    
    def _process_user_wrapper(self, user_id: str, max_workers: int, user_index: Optional[int], generate_english: bool) -> Optional[Dict]:
        """Wrapper function to execute process_user in thread pool
        
        This wrapper ensures each user's processing runs independently without interference
        """
        try:
            return self.process_user(user_id, max_workers, user_index=user_index, generate_english=generate_english)
        except Exception as e:
            print(f"User {user_id} processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_summary(self, results: Dict):
        """Generate processing summary report in append mode"""
        summary_path = os.path.join(self.output_base, "processing_summary.json")

        # Load existing summary if exists
        existing_summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    existing_summary = json.load(f)
            except:
                existing_summary = {}

        # Get current timestamp
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update summary data
        if "processing_sessions" not in existing_summary:
            existing_summary["processing_sessions"] = []

        # Create current session record
        session_id = f"session_{len(existing_summary['processing_sessions']) + 1}"
        current_session = {
            "session_id": session_id,
            "timestamp": current_time,
            "total_users_processed": len(results),
            "user_results": {}
        }

        for user_id, result in results.items():
            post_data = result.get("discussion_post", {})
            text_len = len(post_data.get("text", ""))
            img_count = len(post_data.get("images", []))
            
            # Check if English version exists
            english_data = result.get("english_post")
            has_english = english_data is not None
            
            current_session["user_results"][user_id] = {
                "post_length": text_len,
                "image_count": img_count,
                "link_count": len(post_data.get("links", [])),
                "html_path": post_data.get("html_post", ""),
                "hot_topics": post_data.get("hot_topics", []),
                "generation_mode": post_data.get("mode", "unknown"),
                "has_english_version": has_english,
                "english_html_path": english_data.get("html_path", "") if has_english else ""
            }

        # Add to session list
        existing_summary["processing_sessions"].append(current_session)

        # Calculate overall statistics
        total_sessions = len(existing_summary["processing_sessions"])
        total_users = sum(session["total_users_processed"] for session in existing_summary["processing_sessions"])

        existing_summary["overall_statistics"] = {
            "total_processing_sessions": total_sessions,
            "total_users_processed": total_users,
            "last_updated": current_time
        }

        # Save summary report
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(existing_summary, f, ensure_ascii=False, indent=2)

        print(f"\n📊 Summary report updated: {summary_path}")
        print(f"📈 Current session: {session_id} | Processed users: {len(results)}")
        print(f"📊 Cumulative stats: {total_sessions} sessions | {total_users} users")


def resolve_user_selection(inputs: List[str], available_users: List[str]) -> List[str]:
    """Helper function: Convert input indices or IDs to real user ID list
    
    Supports:
    - Single index: "0", "5", "10"
    - Range: "10-20" (inclusive)
    - User ID: direct user ID string
    - Multiple: "0 5 10-15 20"
    """
    selected_users = []
    
    for item in inputs:
        item = item.strip()
        if not item:
            continue
        
        # Check for range format (e.g., "10-20")
        if '-' in item and not item.startswith('-'):
            parts = item.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start_idx = int(parts[0])
                end_idx = int(parts[1])
                
                # Validate range
                if start_idx > end_idx:
                    print(f"⚠️  Warning: Invalid range [{item}] - start must be <= end")
                    continue
                
                if start_idx < 0 or end_idx >= len(available_users):
                    print(f"⚠️  Warning: Range [{item}] out of bounds (valid: 0-{len(available_users)-1})")
                    # Clamp to valid range
                    start_idx = max(0, start_idx)
                    end_idx = min(len(available_users) - 1, end_idx)
                
                # Add all users in range (inclusive)
                for idx in range(start_idx, end_idx + 1):
                    real_id = available_users[idx]
                    if real_id not in selected_users:
                        selected_users.append(real_id)
                
                print(f"✅ Added range [{start_idx}-{end_idx}]: {end_idx - start_idx + 1} users")
                continue
            
        # Try as single index
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(available_users):
                real_id = available_users[idx]
                if real_id not in selected_users:
                    selected_users.append(real_id)
            else:
                print(f"⚠️  Warning: Index [{idx}] out of range (0-{len(available_users)-1})")
        
        # Try as real ID match
        elif item in available_users:
            if item not in selected_users:
                selected_users.append(item)
        else:
            print(f"⚠️  Warning: Index or user ID not found: {item}")
            
    return selected_users


def main():
    """Main function - supports user selection by index"""
    import argparse

    parser = argparse.ArgumentParser(description='Hupu hot topic discussion post generator')
    parser.add_argument('--users', nargs='+', help='Specify user indices, ranges, or IDs (e.g., 0 5-10 15)')
    parser.add_argument('--all', action='store_true', help='Process all available users')
    parser.add_argument('--workers', type=int, default=4, help='Internal parallel worker threads per user (for image generation, etc.)')
    parser.add_argument('--parallel', type=int, default=40, help='Number of users to process in parallel (default: 40)')
    parser.add_argument('--no-english', action='store_true', help='Do not generate English version (default: auto-generate)')
    parser.add_argument('--max-reflections', type=int, default=None, help='Maximum reflection iterations (default: 10, can be set via MAX_REFLECTION_ITERATIONS_HUPU env var)')

    args = parser.parse_args()

    # Create processor
    processor = HupuProcessor()
    
    # Get available users
    available_users = processor.get_available_users()
    if not available_users:
        print("❌ No user data found, please check download/hupu12 directory structure")
        return

    processor.setup_environment()
    
    # Set reflection iterations (prioritize command line, otherwise interactive input)
    if args.max_reflections is not None:
        max_reflections = args.max_reflections
        os.environ["MAX_REFLECTION_ITERATIONS_HUPU"] = str(max_reflections)
        print(f"✅ Reflection iterations set to: {max_reflections}")
    else:
        # Interactive input for reflection iterations
        try:
            default_reflections = int(os.getenv("MAX_REFLECTION_ITERATIONS_HUPU", "10"))
            print(f"\n💡 Reflection iterations setting (default: {default_reflections})")
            reflection_input = input(f"Enter maximum reflection iterations (press Enter for default {default_reflections}): ").strip()
            if reflection_input:
                max_reflections = int(reflection_input)
                os.environ["MAX_REFLECTION_ITERATIONS_HUPU"] = str(max_reflections)
                print(f"✅ Reflection iterations set to: {max_reflections}")
            else:
                max_reflections = default_reflections
                print(f"✅ Using default reflection iterations: {max_reflections}")
        except ValueError:
            print(f"⚠️  Invalid input, using default: {default_reflections}")
            max_reflections = default_reflections
        except KeyboardInterrupt:
            print("\n⚠️  User cancelled input, using default")
            max_reflections = default_reflections

    # Print user mapping table
    print(f"\n{'='*20} Available Users {'='*20}")
    print(f"{'Index':<6} | {'User ID'}")
    print("-" * 40)
    for idx, user_id in enumerate(available_users):
        print(f"[{idx:<4}] : {user_id}")
    print("-" * 40)

    target_user_ids = []

    # Determine which users to process
    if args.all:
        print("🚀 Selected to process all users")
        target_user_ids = available_users
        
    elif args.users:
        # Command line arguments (could be indices or IDs)
        target_user_ids = resolve_user_selection(args.users, available_users)
        
    else:
        # Interactive selection
        print(f"\nFound {len(available_users)} users.")
        print("💡 Tip: You can use ranges (e.g., '0-5' or '10-20') or individual indices (e.g., '0 5 10')")
        user_input = input("Enter indices/ranges to process (space-separated, or 'all' for all): ")

        if user_input.strip().lower() == 'all':
            target_user_ids = available_users
        else:
            target_user_ids = resolve_user_selection(user_input.split(), available_users)

    # Final confirmation
    if not target_user_ids:
        print("❌ No valid users selected, exiting")
        return

    print(f"\n✅ Will process the following {len(target_user_ids)} users:")
    for uid in target_user_ids:
        idx = available_users.index(uid)
        print(f"  [{idx}] {uid}")
    print(f"{'='*50}\n")

    # Start processing
    results = processor.process_users(
        target_user_ids, 
        max_workers=args.workers, 
        generate_english=not args.no_english,
        parallel_users=args.parallel
    )

    if results:
        print(f"\n🎉 Processing complete! Generated {len(results)} user discussion posts")
        for user_id in results.keys():
            idx = available_users.index(user_id) if user_id in available_users else None
            dir_name = f"{idx}_{user_id}" if idx is not None else user_id
            print(f"  User {user_id}: View generated_posts/{dir_name}/discussion_post.html")
    else:
        print("❌ No user data processed")


if __name__ == "__main__":
    main()

























