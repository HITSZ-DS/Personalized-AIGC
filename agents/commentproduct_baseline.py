import os
import json
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
# Load environment variables
load_dotenv()


class Gemini3HupuBaselineGenerator:
    """Baseline generator using Gemini 3 Pro Image Preview to generate complete Hupu discussion post screenshots"""
    
    def __init__(self):
        # Use the same API key as the image generation model
        self.api_key = os.getenv("IMAGE_API_KEY")
        self.base_url = os.getenv("IMAGE_BASE_URL")
        
        if not self.api_key:
            raise ValueError("IMAGE_API_KEY not found in environment variables")
        if not self.base_url:
            raise ValueError("IMAGE_BASE_URL not found in environment variables")
        
        #self.model = "gemini-3-pro-image-preview"
        self.model = "gemini-2.5-flash-image-preview"
        #self.model = "gpt-4o-image-vip"
        self.output_dir = "generated_hupu_baseline"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def parse_item_profiles(self, item_profiles_path):
        """Parse item_profiles.txt file and return all notes
        
        Args:
            item_profiles_path: Path to item_profiles.txt file
            
        Returns:
            tuple of (user_id, list of notes)
        """
        if not os.path.exists(item_profiles_path):
            raise FileNotFoundError(f"Item profiles file not found: {item_profiles_path}")
        
        with open(item_profiles_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract user ID from path: download/hupu/{user_id}/item_profiles.txt
        path_parts = Path(item_profiles_path).parts
        user_id = None
        for i, part in enumerate(path_parts):
            if part == "hupu" and i + 1 < len(path_parts):
                user_id = path_parts[i + 1]
                break
        
        if not user_id:
            raise ValueError(f"Could not extract user_id from path: {item_profiles_path}")
        
        # Parse content to extract notes (both Recommended and Historical)
        notes = []
        current_note = None
        current_section = None
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('## Note:'):
                # Save previous note if exists
                if current_note:
                    notes.append(current_note)
                
                # Start new note
                note_title = line.replace('## Note:', '').strip()
                current_note = {
                    'title': note_title,
                    'text_content': '',
                    'image_descriptions': []
                }
                current_section = None
            elif line.startswith('### Text Content:'):
                current_section = 'text'
            elif line.startswith('### Image Content:'):
                current_section = 'image'
            elif current_note and line:
                if current_section == 'text' and line.startswith('- Text Contents:'):
                    # Skip the label, next lines are content
                    continue
                elif current_section == 'text':
                    if line and not line.startswith('-'):
                        current_note['text_content'] += line + ' '
                elif current_section == 'image' and line.startswith('- Image'):
                    # Image description starts
                    continue
                elif current_section == 'image' and line.startswith('**Image Contents:**'):
                    # Skip label
                    continue
                elif current_section == 'image' and line:
                    if line and not line.startswith('-') and not line.startswith('*'):
                        current_note['image_descriptions'].append(line)
        
        # Save last note
        if current_note:
            notes.append(current_note)
        
        return user_id, notes
    
    def load_example_data(self, item_profiles_path, note_index=0):
        """Load user profile and content from item_profiles.txt file
        
        Args:
            item_profiles_path: Path to item_profiles.txt file
            note_index: Index of the note to use (0-based, defaults to 0 for first note)
        """
        user_id, notes = self.parse_item_profiles(item_profiles_path)
        
        if not notes:
            raise ValueError("No notes found in item_profiles.txt")
        
        # Select note by index (use first note by default)
        selected_note = notes[note_index % len(notes)]
        
        # Build profile text from user's historical content
        # Extract common themes from all notes
        all_text = ' '.join([n.get('text_content', '')[:200] for n in notes[:5]])
        profile_summary = f"User {user_id} with {len(notes)} historical posts. Content themes: {all_text[:300]}..."
        
        # Create a simplified profile structure
        profile = {
            "user_id": user_id,
            "profile_text": profile_summary
        }
        
        # Create idea structure from selected note
        idea = {
            "idea": selected_note['title'],
            "text_summary": selected_note['text_content'][:500] if selected_note['text_content'] else "",
            "main_type": "Discussion Post"
        }
        
        return profile, idea
    
    def _build_english_prompt(self, profile_text, idea):
        """Build English prompt for generating complete Hupu discussion post screenshot
        
        Args:
            profile_text: User profile text
            idea: Idea dict with title and text summary
        """
        idea_text = idea.get('idea', '')
        text_summary = idea.get('text_summary', '')
        
        # Build idea description
        idea_desc = f"**Topic:** {idea_text}\n"
        if text_summary:
            idea_desc += f"**Content Summary:** {text_summary[:300]}\n"
        
        return f"""You are generating a complete Hupu (虎扑) forum discussion post screenshot in English. The output should be a SINGLE IMAGE that contains text arranged like a real forum discussion post.

**User Profile:**
{profile_text}

**Content Idea:**
{idea_desc}

**CRITICAL REQUIREMENTS:**

1. **Output Format**: Generate ONE complete screenshot image that includes:
   - Text content embedded in the image (English, 200-400 words)
   - Hupu forum style layout (no images, text-only discussion post)
   - Title/header visible at the top

2. **Text Content Style** (Hupu forum style):
   - Direct, concise, and opinionated
   - Natural conversational tone
   - Can use casual expressions like "JRs怎么看？" (translated to English equivalent)
   - Clear paragraph breaks
   - Avoid repetitive phrases, vary expressions
   - Keep it authentic, like a real forum user sharing opinions
   - Base the content on the user's historical posts and interests from their profile

3. **Visual Layout:**
   - Mimic the actual Hupu app interface
   - Title/header at the top (if applicable)
   - Text content in the middle section
   - Professional, clean, web-friendly, mobile-friendly design
   - White or light background for text areas
   - Forum-style typography

4. **Content Focus:**
   - Discussion post style (not news article)
   - Personal opinions and analysis based on user's interests
   - Engaging and thought-provoking
   - Should reflect the user's posting style and topics from their historical content

**IMPORTANT**: The final output should be a SINGLE IMAGE file that looks like a screenshot of a complete Hupu discussion post with text content visible. Generate this complete post screenshot now:"""
    
    def generate_complete_post(self, user_profile, idea):
        """Generate a complete post screenshot using Gemini 3 Pro Image Preview"""
        
        # Extract profile text
        if isinstance(user_profile, dict):
            profile_text = user_profile.get("profile_text", json.dumps(user_profile, ensure_ascii=False))
        else:
            profile_text = str(user_profile)
        
        # Build prompt
        prompt = self._build_english_prompt(profile_text, idea)
        
        try:
            # Call Gemini 3 Pro Image Preview API
            response = requests.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topP": 0.9,
                        "topK": 40
                    }
                },
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"❌ API Error {response.status_code}: {response.text[:500]}")
                return None
            
            result = response.json()
            
            # Extract the generated image
            candidates = result.get("candidates", [])
            if not candidates:
                print("❌ No candidates in response")
                return None
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            image_data = None
            text_content = None
            mime_type = "image/png"
            
            for part in parts:
                if "inlineData" in part:
                    # This is an image
                    image_data = part["inlineData"]["data"]
                    mime_type = part["inlineData"].get("mimeType", "image/png")
                elif "text" in part:
                    # This is text content
                    text_content = part["text"]
            
            return {
                "image_data": image_data,
                "image_mime_type": mime_type if image_data else None,
                "text_content": text_content
            }
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
    
    def save_result(self, result, index, user_id, idea_title):
        """Save the generated post screenshot
        
        Args:
            result: Generation result dict
            index: Index number (0-based)
            user_id: User ID
            idea_title: Idea title
        """
        if not result:
            return None
        
        # Create output directory: {index}_{user_id}
        output_subdir = os.path.join(self.output_dir, f"{index}_{user_id}")
        os.makedirs(output_subdir, exist_ok=True)
        
        saved_files = []
        
        # Save image if available (named as discussion_post_{index}.jpg)
        if result.get("image_data"):
            # Always save as jpg
            image_path = os.path.join(output_subdir, f"discussion_post_{index}.jpg")
            image_bytes = base64.b64decode(result["image_data"])
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            saved_files.append(image_path)
        
        # Save text content if available
        if result.get("text_content"):
            text_path = os.path.join(output_subdir, "post_text.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(result["text_content"])
            saved_files.append(text_path)
        
        # Save metadata
        metadata = {
            "index": index,
            "user_id": user_id,
            "idea_title": idea_title,
            "model": self.model,
            "has_image": result.get("image_data") is not None,
            "has_text": result.get("text_content") is not None
        }
        metadata_path = os.path.join(output_subdir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        saved_files.append(metadata_path)
        
        return saved_files
    
    def process_single_user(self, index, item_profiles_path, skip_if_exists=True):
        """Process a single user's item_profiles.txt file
        
        Args:
            index: Index number (0-based)
            item_profiles_path: Path to item_profiles.txt file
            skip_if_exists: If True, skip if output jpg already exists (resume mechanism)
            
        Returns:
            dict with success status and info
        """
        try:
            # Load example data from item_profiles.txt (use first note)
            profile, idea = self.load_example_data(item_profiles_path, note_index=0)
            user_id = profile.get("user_id", "unknown")
            idea_title = idea.get("idea", "unknown")
            
            # Check if output already exists (resume mechanism)
            if skip_if_exists:
                output_subdir = os.path.join(self.output_dir, f"{index}_{user_id}")
                image_path = os.path.join(output_subdir, f"discussion_post_{index}.jpg")
                if os.path.exists(image_path):
                    print(f"[{index}] ⏭️  Skipping user: {user_id} (already processed)")
                    return {
                        "index": index,
                        "user_id": user_id,
                        "success": True,
                        "skipped": True,
                        "message": "Already processed"
                    }
            
            print(f"[{index}] Processing user: {user_id} - {idea_title[:50]}...")
            
            # Generate complete post (based only on user's item_profiles)
            result = self.generate_complete_post(profile, idea)
            
            if not result or not result.get("image_data"):
                print(f"[{index}] ❌ Generation failed for user: {user_id}")
                return {
                    "index": index,
                    "user_id": user_id,
                    "success": False,
                    "error": "Generation failed"
                }
            
            # Save results
            saved_files = self.save_result(result, index, user_id, idea_title)
            
            if saved_files:
                print(f"[{index}] ✅ Successfully generated and saved: {user_id}")
                return {
                    "index": index,
                    "user_id": user_id,
                    "idea_title": idea_title,
                    "success": True,
                    "files": saved_files
                }
            else:
                print(f"[{index}] ❌ Failed to save results for user: {user_id}")
                return {
                    "index": index,
                    "user_id": user_id,
                    "success": False,
                    "error": "Save failed"
                }
                
        except Exception as e:
            print(f"[{index}] ❌ Error processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                "index": index,
                "user_id": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def get_available_users(self, max_count=5):
        """Scan download/hupu directory for all users with item_profiles.txt
        
        Args:
            max_count: Maximum number of users to process
            
        Returns:
            List of (index, item_profiles_path) tuples
        """
        hupu_dir = Path("download/hupu")
        
        if not hupu_dir.exists():
            raise ValueError(f"Hupu directory not found: {hupu_dir}")
        
        # Find all item_profiles.txt files
        item_profiles_files = list(hupu_dir.glob("*/item_profiles.txt"))
        
        if not item_profiles_files:
            raise ValueError(f"No item_profiles.txt files found in {hupu_dir}")
        
        print(f"   📝 Found {len(item_profiles_files)} users with item_profiles.txt")
        
        # Limit to max_count users
        selected_files = item_profiles_files[:max_count]
        
        # Return tuples with (index, path)
        return [(i, str(path)) for i, path in enumerate(selected_files)]
    
    def run_batch_test(self, max_workers=5, max_count=5, skip_if_exists=True):
        """Run batch test with parallel processing for multiple users
        
        Args:
            max_workers: Maximum number of parallel workers
            max_count: Maximum number of users to process
            skip_if_exists: If True, skip users that already have output jpg files (resume mechanism)
        """
        print("=" * 80)
        print(f"🧪 Batch Testing Gemini 3 Pro Image Preview - Hupu Discussion Post Generation")
        print(f"   Processing {max_count} users in parallel (max {max_workers} workers)")
        if skip_if_exists:
            print(f"   ⏭️  Resume mode: will skip already processed users")
        print("=" * 80)
        
        # Get available users
        print(f"\n1️⃣ Scanning for available users...")
        try:
            users = self.get_available_users(max_count=max_count)
            print(f"   ✅ Found {len(users)} users to process")
        except Exception as e:
            print(f"   ❌ Failed to get users: {e}")
            return None
        
        if not users:
            print("   ⚠️ No users found")
            return None
        
        # Process in parallel
        print(f"\n2️⃣ Processing {len(users)} users in parallel (max {max_workers} workers)...")
        start_time = time.time()
        
        results = []
        success_count = 0
        skipped_count = 0
        fail_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_single_user, index, item_profiles_path, skip_if_exists=skip_if_exists): index
                for index, item_profiles_path in users
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.get("skipped"):
                        skipped_count += 1
                    elif result.get("success"):
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"[{index}] ❌ Exception in future: {e}")
                    fail_count += 1
                    results.append({
                        "index": index,
                        "success": False,
                        "error": str(e)
                    })
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 Batch Processing Summary")
        print(f"{'='*80}")
        print(f"Total processed: {len(results)}")
        print(f"✅ Success: {success_count}")
        print(f"⏭️  Skipped: {skipped_count}")
        print(f"❌ Failed: {fail_count}")
        print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
        
        # Print failed cases
        if fail_count > 0:
            print("Failed cases:")
            for result in results:
                if not result.get("success") and not result.get("skipped"):
                    print(f"  [{result.get('index', '?')}] {result.get('user_id', 'unknown')}: {result.get('error', 'Unknown error')}")
        
        return results


if __name__ == "__main__":
    generator = Gemini3HupuBaselineGenerator()
    
    # Run batch test with 5 users, 5 parallel workers
    generator.run_batch_test(max_workers=40, max_count=1000)

