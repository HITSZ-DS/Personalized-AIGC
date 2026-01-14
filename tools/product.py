
from agents.itproduct_generator import ITProductGenerator
from agents.profile2idea import Profile2Idea
from agents.vproduct_generator import VideoProductGenerator
import os
import json
import re

def _extract_json_from_response(response_str):
    """Extract JSON array from LLM response that might contain markdown or extra text"""
    print(f"Attempting to extract JSON from response (length: {len(response_str)})")

    # Method 1: Try to find JSON array between ```json and ```
    json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_str, re.DOTALL)
    if json_match:
        try:
            json_content = json_match.group(1).strip()
            print(f"Found JSON in ```json``` block: {json_content[:100]}...")
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from ```json``` block: {e}")

    # Method 2: Try to find JSON array between ``` and ``` (without json specifier)
    json_match = re.search(r'```\s*(\[.*?\])\s*```', response_str, re.DOTALL)
    if json_match:
        try:
            json_content = json_match.group(1).strip()
            print(f"Found JSON in ``` block: {json_content[:100]}...")
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from ``` block: {e}")

    # Method 3: Try to find any JSON array in the text (greedy match)
    json_match = re.search(r'\[.*\]', response_str, re.DOTALL)
    if json_match:
        try:
            json_content = json_match.group().strip()
            print(f"Found JSON array in text: {json_content[:100]}...")
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON array from text: {e}")

    # Method 4: Try to clean up the response and extract JSON
    # Remove common prefixes/suffixes that might interfere
    cleaned_response = response_str.strip()
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:]  # Remove ```json
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3]  # Remove ```
    cleaned_response = cleaned_response.strip()

    if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
        try:
            print(f"Trying cleaned response: {cleaned_response[:100]}...")
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse cleaned response: {e}")

    print("Could not extract valid JSON from response, using empty list")
    print(f"Response preview: {response_str[:200]}...")
    return []

def product():
    # idea generation
    dataset_path = os.path.join("download", os.getenv("DATASET"))
    for user in os.listdir(dataset_path):
        print(f"start to generate product for user {user}...")
        user_dir = os.path.join(dataset_path, user)
        user_profile_path = os.path.join(user_dir, "user_profile.txt")
        with open(user_profile_path, "r", encoding="utf-8") as f:
            user_profile = f.read()
        print("start to generate ideas...")
        idea_generator = Profile2Idea()
        ideas_str = idea_generator(user_profile, user_dir)
        print("ideas generated...")
        # Parse the JSON string to get the ideas list
        import json
        try:
            ideas = json.loads(ideas_str)
        except json.JSONDecodeError:
            print("Error: Could not parse ideas JSON, attempting to extract JSON from response...")
            # Try multiple extraction methods
            ideas = _extract_json_from_response(ideas_str)
        # Save ideas to file first
        with open(os.path.join(user_dir, "product_ideas.json"), "w", encoding="utf-8") as f:
            json.dump(ideas, f, ensure_ascii=False, indent=2)

        # 按照ideas的main type分成videa, itidea
        # Use the already parsed ideas instead of reading from file again
        videa = [idea for idea in ideas if idea["main_type"] == "Video Content"]
        itidea = [idea for idea in ideas if idea["main_type"] == "Text-Image Content"]
        # video product generation
        print(f"start to generate video product based on {len(videa)} ideas...")
        video_product_generator = VideoProductGenerator(videa)
        video_product_generator()
        # text-image product generation
        print(f"start to generate text-image product based on {len(itidea)} ideas...")
        it_product_generator = ITProductGenerator(itidea)
        it_product_generator(user_profile, user_dir)
