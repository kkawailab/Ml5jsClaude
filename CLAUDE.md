# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a comprehensive Japanese-language tutorial for ml5.js, a machine learning library for JavaScript. The repository contains educational content structured as a book with 7 chapters, covering topics from basic setup to advanced projects using computer vision, pose detection, hand recognition, and sound classification.

## Repository Structure

The content is organized as a tutorial book in Markdown format:
- `README.md` - Table of contents and introduction (serves as the book cover)
- `chapter01-introduction.md` through `chapter07-projects.md` - Main tutorial chapters
- `README.txt` - Original project requirements (in Japanese)

## Content Guidelines

When modifying or adding content to this repository:

### Language and Format
- All content must be written in Japanese
- Use Markdown format exclusively for documentation
- Include detailed code comments in Japanese within all sample code

### Tutorial Structure
Each chapter should maintain the following structure:
1. Conceptual explanations with subsections
2. Multiple sample code examples with detailed Japanese comments
3. Practice problems (練習問題) at the end of each chapter
4. Solutions (解答) for all practice problems

### Code Examples
- Sample code should be complete and runnable
- Include both HTML structure and JavaScript implementation
- Use p5.js integration where appropriate
- Reference ml5.js models: MobileNet, PoseNet, HandPose, SoundClassifier, Sentiment, Word2Vec, CharRNN

## Common Development Tasks

### Adding New Content
When adding new sections or chapters:
1. Follow the existing chapter structure and numbering scheme
2. Update the table of contents in `README.md`
3. Include the update in the 更新履歴 (update history) section

### Git Workflow
```bash
# Stage all changes
git add .

# Commit with Japanese message describing changes
git commit -m "章の内容を更新: [具体的な変更内容]"

# Push to GitHub (requires remote setup)
git push origin main
```

### Creating Sample Code
Sample code should follow this pattern:
```javascript
// 機能の説明（日本語コメント）
let variable;

function setup() {
    // セットアップの説明
    createCanvas(640, 480);
    // ml5モデルの初期化
    classifier = ml5.imageClassifier('MobileNet', modelReady);
}

function modelReady() {
    console.log('モデル準備完了');
}
```

## ml5.js Model Coverage

The tutorial covers these ml5.js models and their applications:
- **Image Classification**: MobileNet, DarkNet, DoodleNet, custom models via Teachable Machine
- **Pose Detection**: PoseNet (17 keypoints detection)
- **Hand Recognition**: HandPose (21 landmarks)
- **Sound**: SoundClassifier (Speech Commands 18w)
- **Text Analysis**: Sentiment analysis, Word2Vec, CharRNN

## Chapter-Specific Focus

- **Chapter 1-2**: Foundation and environment setup
- **Chapter 3**: Image classification techniques
- **Chapter 4**: Human pose detection and skeleton tracking
- **Chapter 5**: Hand gesture recognition and virtual painting
- **Chapter 6**: Audio processing and natural language
- **Chapter 7**: Complete project implementations combining multiple models

## Important Notes

- The tutorial is designed for beginners with basic JavaScript/HTML knowledge
- Each chapter builds upon previous knowledge
- All code examples should work in modern browsers without additional setup beyond loading ml5.js and p5.js from CDN
- Sample code emphasizes clarity and educational value over performance optimization