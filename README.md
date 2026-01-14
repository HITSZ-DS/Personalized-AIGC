<div align="center">
  <img src="./assets/logo.png">
  <br>
  <br>
  <h1 align="center">🎨 TailorRec: From User Behavior to Creative Content</h1>

  <p align="center">
    <img src="https://img.shields.io/badge/🐍Python-3.10+-00d9ff?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
    <img src="https://img.shields.io/badge/🔥PyTorch-2.3+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=1a1a2e">
    <img src="https://img.shields.io/badge/License-Apache_2.0-4ecdc4?style=for-the-badge&logo=opensourceinitiative&logoColor=white&labelColor=1a1a2e">
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/🤖SSLRec-WSDM'24-ff6b6b?style=for-the-badge&labelColor=1a1a2e">
    <img src="https://img.shields.io/badge/🎬VideoAgent-Multi--Agent-9b59b6?style=for-the-badge&labelColor=1a1a2e">
  </p>

</div>

<div align="center">
  <a href="#-quick-start" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/🚀Quick%20Start-Get%20Started%20Now-FFC107?style=for-the-badge&logo=rocket&logoColor=white&labelColor=1a1a2e">
  </a>
</div>

---

### 🚨 Current Personalized Content Generation Challenges:
- ❌ **Generic Outputs** - Most AI content generators produce one-size-fits-all content without understanding individual preferences <br>
- ❌ **Single Modality** - Tools typically generate only text OR images OR videos
- ❌ **Platform Isolation** - No integration with real social platforms for authentic user understanding <br>

### 💡 Personalized-AIGC Solution:
🎯 **Recommendation Engine**, **Multimodal Analyzer**, **Profile Generator**, and **Content Creator** **All-in-One**! We're building a future where AI truly understands YOU. 💡 Simply connect your social accounts. Personalized-AIGC autonomously analyzes your behavior, builds your preference profile, and generates tailor-made content—videos, posts, and more! 🚀

</div>

---

## 📑 Table of Contents

- [💡 Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📦 Modules](#-modules)
- [📊 Evaluation](#-evaluation)

---

## 💡 Key Features

<br/>

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="25%" align="center" style="vertical-align: top; padding: 20px;">

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">📊 <strong>Behavior2Profile</strong></h3>
</div>

<div align="center" style="margin: 15px 0;">
  <img src="https://img.shields.io/badge/USER-PROFILING-ff6b6b?style=for-the-badge&logo=user&logoColor=white" alt="Profiling Badge" />
</div>

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Automatic Preference Learning</strong></p>
</div>

<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center">Transform <strong>raw user behavior data</strong> from social platforms into comprehensive preference profiles through <strong>multimodal LLM analysis</strong> of videos, images, and text.
</p>
</div>

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 20px;">

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🎬 <strong>Profile2Video</strong></h3>
</div>

<div align="center" style="margin: 15px 0;">
  <img src="https://img.shields.io/badge/VIDEO-GENERATION-4ecdc4?style=for-the-badge&logo=video&logoColor=white" alt="Video Badge" />
</div>

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>AI Video Production Studio</strong></p>
</div>

<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center">Generate <strong>personalized video content</strong> including crosstalk, talk shows, music videos, meme edits, and news summaries tailored to user interests.</p>
</div>

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 20px;">

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">📱 <strong>Profile2Post</strong></h3>
</div>

<div align="center" style="margin: 15px 0;">
  <img src="https://img.shields.io/badge/SOCIAL-CONTENT-9b59b6?style=for-the-badge&logo=instagram&logoColor=white" alt="Social Badge" />
</div>

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>Publication-Ready Posts</strong></p>
</div>

<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center">Create <strong>engaging social media posts</strong> with AI-generated copy, images, and platform-specific links for Bilibili, Xiaohongshu, Zhihu, Douyin, and Weibo.</p>
</div>

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 20px;">

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<h3 style="margin: 0; padding: 0;">🔮 <strong>SSL-RecSys</strong></h3>
</div>

<div align="center" style="margin: 15px 0;">
  <img src="https://img.shields.io/badge/RECOMMENDATION-ENGINE-FFC107?style=for-the-badge&logo=tensorflow&logoColor=white" alt="RecSys Badge" />
</div>

<div style="height: 80px; display: flex; align-items: center; justify-content: center;">
<p align="center"><strong>30+ SOTA Models</strong></p>
</div>

<div style="height: 60px; display: flex; align-items: center; justify-content: center;">
<p align="center">Leverage <strong>self-supervised learning</strong> recommendation with LightGCN, SGL, BERT4Rec, and more for precise user preference prediction.</p>
</div>

</td>
</tr>
</table>

<br/>

---

## 🎯 **End-to-End Personalized Content Pipeline**

**The Challenges We Solve**:

| Challenge | Traditional Approach | Our Solution |
|-----------|---------------------|--------------|
| 🎭 **User Understanding** | Manual input of preferences | Automatic behavior analysis from social platforms |
| 🧠 **Content Ideation** | Generic prompts | Profile-driven creative idea generation |
| 🎬 **Video Production** | Single clips, no consistency | Multi-agent studio with 7 video types |
| 📱 **Social Posts** | Text-only generation | Rich multimedia with platform links |
| 🔄 **Real-time Adaptation** | Static profiles | Dynamic profiling from live data |

---

### 🔥 **Why Personalized-AIGC?**

| 🧠 **Zero Manual Input** | 🎬 **Multi-Modal Output** | 🌐 **Platform Native** | 🤖 **SOTA RecSys** | ⚡ **Real-Time** |
|:---:|:---:|:---:|:---:|:---:|
| Automatic Profiling | Video + Text + Image | Bilibili, Xiaohongshu, Hupu | 30+ SSL Models | Live Data Analysis |
| No need to describe yourself—we learn from your actual behavior across platforms | Generate complete video productions, social media posts with images, and cross-platform content | Native integration with major Chinese social platforms for authentic user data | Powered by SSLRec (WSDM'24) with LightGCN, BERT4Rec, SGL, and more | Continuously update profiles with fresh user interaction data |

---

## 🏗️ Architecture

### 📊 **System Overview**

**Personalized-AIGC** is a multi-agent content generation framework that transforms user behavioral data into personalized creative content. Our system seamlessly bridges recommendation systems with AIGC, allowing AI to truly understand and create for individual users.

<div align="center">
  <table align="center" width="100%" style="border: none; border-collapse: collapse;">
    <tr>
      <td colspan="3" align="center" style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; font-weight: bold;">
        🕷️ <strong>DATA COLLECTION LAYER</strong><br/>
        📺 Bilibili Spider • 📕 Xiaohongshu Spider • 🏀 Hupu Spider • 🎬 Douban Spider
      </td>
    </tr>
    <tr><td colspan="3" height="20"></td></tr>
    <tr>
      <td colspan="3" align="center" style="padding: 15px; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); border-radius: 12px; color: white; font-weight: bold;">
        🔮 <strong>RECOMMENDATION ENGINE (SSLRec)</strong><br/>
        LightGCN • SGL • SimGCL • BERT4Rec • DuoRec • 30+ Models
      </td>
    </tr>
    <tr><td colspan="3" height="15"></td></tr>
    <tr>
      <td align="center" style="padding: 12px; background: linear-gradient(135deg, #3742fa 0%, #2f3542 100%); border-radius: 10px; color: white; width: 50%;">
        🎥 <strong>VIDEO ANALYST</strong><br/>
        <small>Frame Extraction • Scene Understanding • Content Summarization</small>
      </td>
      <td width="10"></td>
      <td align="center" style="padding: 12px; background: linear-gradient(135deg, #8c7ae6 0%, #9c88ff 100%); border-radius: 10px; color: white; width: 50%;">
        🖼️ <strong>IMAGE ANALYST</strong><br/>
        <small>Visual Recognition • Style Analysis • Object Detection</small>
      </td>
    </tr>
    <tr><td colspan="3" height="15"></td></tr>
    <tr>
      <td colspan="3" align="center" style="padding: 15px; background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%); border-radius: 12px; color: white; font-weight: bold;">
        🧠 <strong>PROFILE GENERATION</strong><br/>
        Item Profiles • User Profiles • Preference Ranking • Creative Ideas
      </td>
    </tr>
    <tr><td colspan="3" height="15"></td></tr>
    <tr>
      <td align="center" style="padding: 12px; background: linear-gradient(135deg, #e056fd 0%, #f368e0 100%); border-radius: 10px; color: white; width: 50%;">
        🎬 <strong>VIDEO AGENT</strong><br/>
        <small>Cross Talk • Talk Show • MAD/SVC • Rhythm Edit • News • Commentary</small>
      </td>
      <td width="10"></td>
      <td align="center" style="padding: 12px; background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%); border-radius: 10px; color: white; width: 50%;">
        📱 <strong>TEXT-IMAGE GENERATOR</strong><br/>
        <small>AI Copywriting • Image Generation • Platform Links • HTML Posts</small>
      </td>
    </tr>
    <tr><td colspan="3" height="15"></td></tr>
    <tr>
      <td colspan="3" align="center" style="padding: 20px; background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%); border-radius: 15px; color: white; font-weight: bold;">
        ✨ <strong>OUTPUT LAYER</strong><br/>
        🎬 Personalized Videos • 📱 Social Media Posts • 🖼️ Generated Images • 📊 User Insights
      </td>
    </tr>
  </table>
</div>

---

### 🎬 **VideoAgent: Multi-Agent Video Production**

<table align="center" width="100%">
<tr>
<td align="center" width="14%">

**🎭 Cross Talk**
<br/>
<img src="https://img.shields.io/badge/相声-Generation-ff6b6b?style=flat-square"/>
<br/>
<small>Convert audio into Chinese crosstalk duo performances</small>

</td>
<td align="center" width="14%">

**🎤 Talk Show**
<br/>
<img src="https://img.shields.io/badge/Stand--up-Comedy-4ecdc4?style=flat-square"/>
<br/>
<small>Transform content into stand-up format</small>

</td>
<td align="center" width="14%">

**🎵 MAD/SVC**
<br/>
<img src="https://img.shields.io/badge/Singing-Voice-9b59b6?style=flat-square"/>
<br/>
<small>Voice conversion with custom lyrics</small>

</td>
<td align="center" width="14%">

**🔊 MAD/TTS**
<br/>
<img src="https://img.shields.io/badge/Meme-Dubbing-FFC107?style=flat-square"/>
<br/>
<small>Character dialogue adaptation</small>

</td>
<td align="center" width="14%">

**🎬 Rhythm**
<br/>
<img src="https://img.shields.io/badge/Beat--Sync-Editing-00d2d3?style=flat-square"/>
<br/>
<small>Automatic beat-synced cuts</small>

</td>
<td align="center" width="14%">

**📰 News**
<br/>
<img src="https://img.shields.io/badge/Video-Summary-e056fd?style=flat-square"/>
<br/>
<small>Event summarization videos</small>

</td>
<td align="center" width="14%">

**📖 Commentary**
<br/>
<img src="https://img.shields.io/badge/Novel-Adaptation-26de81?style=flat-square"/>
<br/>
<small>Book to video narration</small>

</td>
</tr>
</table>

**Built with cutting-edge tools:**

```
🗣️ CosyVoice     →  Voice Cloning & TTS
🐟 Fish-Speech   →  Multilingual Speech Synthesis
🔗 ImageBind     →  Multimodal Embedding
🎧 Demucs        →  Audio Source Separation
🎼 Whisper       →  Speech Recognition
```

---

## 🚀 Quick Start

### 🖥️ **Environment**

```
OS: Linux, Windows
Python: 3.10+
CUDA: 12.1+ (recommended for GPU acceleration)
```

### 📥 **Clone and Install**

```bash
git clone https://github.com/your-username/Personalized-AIGC.git
cd Personalized-AIGC

# Create virtual environment
python -m venv paigc_env
source paigc_env/bin/activate  # Linux/Mac
# or
.\paigc_env\Scripts\activate   # Windows

# Install dependencies
pip install -e .
```

### ⚙️ **Configuration**

Create a `.env` file in the project root:

```env
# Dataset: bilibili, redbook, hupu, douban
DATASET=bilibili

# LLM Configuration
CHAT_API_KEY=your_openai_api_key
CHAT_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o

# Image Generation (Optional)
IMAGE_API_KEY=your_image_api_key
IMAGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta
IMAGE_MODEL=gemini-2.0-flash
```

### 🎯 **Usage**

Run the complete pipeline with a single command:

```bash
python main.py
```

This will:
1. 🔮 Run SSL recommendation model
2. 🧠 Analyze user content (video, image, text)
3. 📊 Generate user profiles
4. ✨ Create personalized content

---

## 📦 Modules

### 1️⃣ **Data Collection**

```python
from tools.bilibili_realtime import BilibiliRealTime

# Collect user favorites from Bilibili
bilibili = BilibiliRealTime()
bilibili.process_all_users(save_data=True, download_videos=True)
```

### 2️⃣ **Recommendation Engine**

```python
from tools.sslrec import run_sslrec

# Run self-supervised recommendation
run_sslrec()
```

### 3️⃣ **Multimodal Analysis**

```python
from tools.analyze import analyze

# Analyze all users' content with concurrent processing
analyze(max_workers=5)
```

### 4️⃣ **Content Generation**

```python
from tools.product import product

# Generate personalized video and text-image content
product()
```

---

