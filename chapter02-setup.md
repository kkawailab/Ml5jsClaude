# 第2章: 環境構築とセットアップ

## 2.1 開発環境の準備

### 必要なツール

ml5.jsを使った開発を始めるために、以下のツールを準備しましょう。

#### 1. Webブラウザ
- **推奨**: Google Chrome（開発者ツールが充実）
- その他: Firefox、Safari、Microsoft Edge（最新版）

#### 2. テキストエディタ / IDE
- **Visual Studio Code**（推奨）
  - 無料で高機能
  - 拡張機能が豊富
  - Live Server拡張機能でローカルサーバーを簡単に起動
- その他の選択肢
  - Sublime Text
  - Atom
  - WebStorm

#### 3. ローカルWebサーバー
ml5.jsの一部機能（Webカメラ、マイク等）は、セキュリティの関係でローカルサーバー経由でアクセスする必要があります。

### Visual Studio Codeのセットアップ

```bash
# VS Codeをインストール後、以下の拡張機能をインストール

# 1. Live Server拡張機能
# VS Code内で Ctrl+Shift+X（Mac: Cmd+Shift+X）を押して拡張機能パネルを開く
# "Live Server"を検索してインストール

# 2. その他便利な拡張機能
# - JavaScript (ES6) code snippets
# - Prettier - Code formatter
# - Bracket Pair Colorizer
```

### プロジェクトフォルダの構成

```
my-ml5-project/
│
├── index.html          # メインのHTMLファイル
├── sketch.js          # JavaScriptコード（p5.js + ml5.js）
├── style.css          # スタイルシート（任意）
├── assets/            # 画像や音声ファイル用フォルダ
│   ├── images/
│   └── sounds/
└── models/            # カスタムモデル用フォルダ（必要に応じて）
```

## 2.2 ml5.jsの読み込み方法

### 方法1: CDN（Content Delivery Network）を使用

最も簡単な方法は、CDNから直接ライブラリを読み込むことです。

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ml5.js セットアップ</title>

    <!-- p5.jsの読み込み（ml5.jsと一緒に使う場合） -->
    <script src="https://cdn.jsdelivr.net/npm/p5@1.7.0/lib/p5.min.js"></script>

    <!-- ml5.jsの読み込み（最新版） -->
    <script src="https://unpkg.com/ml5@latest/dist/ml5.min.js"></script>

    <!-- 特定バージョンを指定する場合 -->
    <!-- <script src="https://unpkg.com/ml5@0.12.2/dist/ml5.min.js"></script> -->
</head>
<body>
    <h1>ml5.js プロジェクト</h1>

    <!-- あなたのJavaScriptファイル -->
    <script src="sketch.js"></script>
</body>
</html>
```

### 方法2: NPMを使用（Node.js環境）

```bash
# プロジェクトフォルダで実行
npm init -y

# ml5.jsをインストール
npm install ml5

# p5.jsも使用する場合
npm install p5
```

```javascript
// JavaScriptファイルでの読み込み
const ml5 = require('ml5');
// または ES6 モジュール構文
import * as ml5 from 'ml5';
```

### 方法3: ローカルファイルとして配置

```html
<!-- ダウンロードしたml5.jsファイルを読み込み -->
<script src="./lib/ml5.min.js"></script>
```

### ml5.jsのみを使用する場合（p5.jsなし）

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ml5.js（p5.jsなし）</title>
    <script src="https://unpkg.com/ml5@latest/dist/ml5.min.js"></script>
</head>
<body>
    <h1>画像分類デモ</h1>
    <img id="targetImage" src="./assets/cat.jpg" width="400">
    <p id="result">分類結果がここに表示されます</p>

    <script>
        // ml5.jsのみを使用したコード
        const image = document.getElementById('targetImage');
        const resultText = document.getElementById('result');

        // MobileNetモデルを初期化
        const classifier = ml5.imageClassifier('MobileNet', () => {
            console.log('モデル読み込み完了');

            // 画像を分類
            classifier.classify(image, (err, results) => {
                if (err) {
                    console.error(err);
                    return;
                }

                // 結果を表示
                resultText.innerHTML = `
                    <strong>結果:</strong> ${results[0].label}<br>
                    <strong>確信度:</strong> ${(results[0].confidence * 100).toFixed(2)}%
                `;
            });
        });
    </script>
</body>
</html>
```

## 2.3 最初のプログラム「Hello ML5」

### 基本的なHTMLテンプレート

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello ML5</title>

    <!-- ライブラリの読み込み -->
    <script src="https://cdn.jsdelivr.net/npm/p5@1.7.0/lib/p5.min.js"></script>
    <script src="https://unpkg.com/ml5@latest/dist/ml5.min.js"></script>

    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        main {
            text-align: center;
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        h1 {
            color: #333;
            margin-bottom: 1rem;
        }

        #status {
            margin: 1rem 0;
            padding: 0.5rem;
            background: #f0f0f0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <main>
        <h1>🤖 Hello ML5!</h1>
        <div id="status">モデルを読み込み中...</div>
        <div id="canvas-container"></div>
    </main>

    <!-- メインのJavaScriptファイル -->
    <script src="sketch.js"></script>
</body>
</html>
```

### Hello ML5プログラム

```javascript
// sketch.js
let mobilenet;
let img;
let label = '';
let confidence = 0;

// 画像とモデルを事前に読み込む
function preload() {
    // サンプル画像を読み込み（Public Domainの画像を使用）
    img = loadImage('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg');

    // MobileNetモデルを読み込み
    mobilenet = ml5.imageClassifier('MobileNet', modelLoaded);
}

// モデルの読み込みが完了したときに呼ばれる
function modelLoaded() {
    console.log('ml5.js: MobileNetモデルの読み込みが完了しました！');
    document.getElementById('status').textContent = 'モデル準備完了！画像をクリックして分類を実行';
}

function setup() {
    // キャンバスを作成して、指定したコンテナに配置
    let canvas = createCanvas(640, 480);
    canvas.parent('canvas-container');

    // 画像を中央に配置するための計算
    let imgX = (width - img.width) / 2;
    let imgY = (height - img.height - 100) / 2;

    // 初期描画
    background(240);
    image(img, imgX, imgY);

    // テキストの設定
    textAlign(CENTER, CENTER);
    textSize(20);
}

function draw() {
    // 背景を再描画
    background(240);

    // 画像を描画
    let imgX = (width - img.width) / 2;
    let imgY = (height - img.height - 100) / 2;
    image(img, imgX, imgY);

    // 結果が存在する場合は表示
    if (label !== '') {
        // 背景ボックス
        fill(0, 0, 0, 200);
        rect(0, height - 80, width, 80);

        // 結果テキスト
        fill(255);
        textSize(24);
        text(`検出: ${label}`, width / 2, height - 45);

        textSize(16);
        text(`確信度: ${(confidence * 100).toFixed(2)}%`, width / 2, height - 20);

        // 確信度バー
        fill(0, 255, 0);
        let barWidth = confidence * (width - 40);
        rect(20, height - 10, barWidth, 5);
    }
}

// マウスがクリックされたときに実行
function mousePressed() {
    // キャンバス内でクリックされた場合のみ実行
    if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
        classifyImage();
    }
}

// 画像を分類する関数
function classifyImage() {
    mobilenet.classify(img, gotResult);

    // 分類中の表示
    document.getElementById('status').textContent = '分類中...';
}

// 分類結果を受け取るコールバック関数
function gotResult(error, results) {
    if (error) {
        console.error('エラーが発生しました:', error);
        document.getElementById('status').textContent = 'エラーが発生しました';
        return;
    }

    // 結果を保存
    label = results[0].label;
    confidence = results[0].confidence;

    // ステータスを更新
    document.getElementById('status').textContent =
        `完了！ "${label}" (${(confidence * 100).toFixed(1)}%)`;

    // コンソールにも結果を表示
    console.log('分類結果:');
    results.forEach((result, index) => {
        console.log(`${index + 1}. ${result.label}: ${(result.confidence * 100).toFixed(2)}%`);
    });
}
```

### Webカメラを使った「Hello ML5」

```javascript
// webcam_hello.js
let video;
let mobilenet;
let label = 'モデル読み込み中...';
let confidence = 0;

function setup() {
    createCanvas(640, 520);

    // Webカメラの映像を取得
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();  // HTMLのvideo要素を非表示

    // MobileNetモデルを初期化
    mobilenet = ml5.imageClassifier('MobileNet', video, modelReady);
}

function modelReady() {
    console.log('モデル準備完了！');
    label = 'モデル準備完了';

    // 継続的な分類を開始
    classifyVideo();
}

function classifyVideo() {
    mobilenet.classify(gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // 結果を更新
    label = results[0].label;
    confidence = results[0].confidence;

    // 次のフレームを分類（継続的な分類）
    classifyVideo();
}

function draw() {
    // ビデオフレームを描画
    image(video, 0, 0);

    // 半透明の背景
    fill(0, 0, 0, 180);
    rect(0, height - 40, width, 40);

    // 結果を表示
    fill(255);
    textSize(20);
    textAlign(CENTER);
    text(`${label} (${(confidence * 100).toFixed(1)}%)`, width / 2, height - 15);

    // FPS表示（パフォーマンス確認用）
    fill(0, 255, 0);
    textSize(12);
    textAlign(LEFT);
    text(`FPS: ${frameRate().toFixed(1)}`, 10, 20);
}
```

## 2.4 デバッグ方法とトラブルシューティング

### ブラウザの開発者ツールの使い方

```javascript
// デバッグ用のコード例
function setup() {
    createCanvas(400, 400);

    // コンソールに情報を出力
    console.log('セットアップ開始');
    console.log('ml5のバージョン:', ml5.version);

    // 変数の内容を確認
    let testData = {x: 100, y: 200};
    console.table(testData);  // テーブル形式で表示

    // エラーの確認
    try {
        // エラーが発生する可能性のあるコード
        let classifier = ml5.imageClassifier('MobileNet', modelReady);
    } catch (error) {
        console.error('エラーが発生:', error);
        // エラー時の処理
        document.getElementById('status').textContent = 'エラーが発生しました';
    }
}

// パフォーマンス測定
function measurePerformance() {
    // 処理時間の測定開始
    console.time('classification');

    classifier.classify(image, (err, results) => {
        // 処理時間の測定終了
        console.timeEnd('classification');

        if (!err) {
            console.log('分類完了:', results);
        }
    });
}

// デバッグ用の詳細ログ
function verboseLog(message, data) {
    if (debugMode) {  // debugModeフラグで制御
        console.group(message);
        console.log('タイムスタンプ:', new Date().toISOString());
        console.log('データ:', data);
        console.trace();  // スタックトレースを表示
        console.groupEnd();
    }
}
```

### よくあるエラーと解決方法

#### 1. CORSエラー

```javascript
// エラー例:
// "Access to image at 'file:///...' from origin 'null' has been blocked by CORS policy"

// 解決方法1: ローカルサーバーを使用
// VS CodeのLive Server拡張機能を使用するか、以下のコマンドでサーバーを起動

// Python 3の場合
// python -m http.server 8000

// Node.jsの場合（http-serverをインストール後）
// npx http-server

// 解決方法2: 画像をBase64に変換
function loadImageAsBase64() {
    // inputタグを使用してローカル画像を読み込み
    let input = createFileInput(handleFile);
    input.position(10, 10);
}

function handleFile(file) {
    if (file.type === 'image') {
        // 画像をBase64として読み込み
        createImg(file.data, 'uploaded image')
            .hide()
            .id('uploadedImage');

        // 分類を実行
        let img = document.getElementById('uploadedImage');
        classifier.classify(img, gotResult);
    }
}
```

#### 2. HTTPSエラー（Webカメラ・マイク使用時）

```javascript
// エラー例:
// "getUserMedia() no longer works on insecure origins"

// 解決方法:
// 1. HTTPSでアクセスする
// 2. localhostを使用する（localhostはHTTPでもOK）
// 3. ngrokなどのトンネリングツールを使用

// セキュアコンテキストのチェック
if (window.isSecureContext) {
    console.log('セキュアコンテキストです - Webカメラ使用可能');
    video = createCapture(VIDEO);
} else {
    console.warn('HTTPSまたはlocalhostでアクセスしてください');
    document.body.innerHTML = `
        <h2>セキュリティエラー</h2>
        <p>Webカメラを使用するには、HTTPSまたはlocalhostでアクセスしてください。</p>
    `;
}
```

#### 3. モデル読み込みエラー

```javascript
// エラーハンドリングの実装
let modelLoadTimeout;

function preload() {
    // タイムアウトを設定（30秒）
    modelLoadTimeout = setTimeout(() => {
        console.error('モデルの読み込みがタイムアウトしました');
        handleModelLoadError();
    }, 30000);

    // モデルを読み込み
    mobilenet = ml5.imageClassifier('MobileNet', modelLoaded, handleModelLoadError);
}

function modelLoaded() {
    clearTimeout(modelLoadTimeout);  // タイムアウトをクリア
    console.log('モデル読み込み成功');
    document.getElementById('status').textContent = '準備完了';
}

function handleModelLoadError(error) {
    clearTimeout(modelLoadTimeout);
    console.error('モデル読み込みエラー:', error);

    // ユーザーに通知
    document.getElementById('status').innerHTML = `
        <span style="color: red;">
            モデルの読み込みに失敗しました。<br>
            ネットワーク接続を確認してください。
        </span>
    `;

    // リトライボタンを表示
    let retryButton = createButton('再試行');
    retryButton.position(width / 2 - 40, height / 2);
    retryButton.mousePressed(() => {
        location.reload();  // ページをリロード
    });
}
```

### パフォーマンス最適化

```javascript
// パフォーマンス監視クラス
class PerformanceMonitor {
    constructor() {
        this.fps = 0;
        this.lastTime = 0;
        this.classificationTime = 0;
        this.frameCount = 0;
    }

    // FPSを更新
    updateFPS() {
        this.frameCount++;
        let currentTime = millis();

        if (currentTime - this.lastTime > 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    }

    // 分類時間を測定
    measureClassification(callback) {
        let startTime = performance.now();

        callback(() => {
            this.classificationTime = performance.now() - startTime;
        });
    }

    // 統計情報を表示
    display() {
        fill(0, 255, 0);
        textAlign(LEFT);
        textSize(12);
        text(`FPS: ${this.fps}`, 10, 20);
        text(`分類時間: ${this.classificationTime.toFixed(1)}ms`, 10, 35);
        text(`メモリ使用量: ${this.getMemoryUsage()}`, 10, 50);
    }

    // メモリ使用量を取得（Chrome限定）
    getMemoryUsage() {
        if (performance.memory) {
            let used = performance.memory.usedJSHeapSize / 1048576;  // MB変換
            let total = performance.memory.totalJSHeapSize / 1048576;
            return `${used.toFixed(1)} / ${total.toFixed(1)} MB`;
        }
        return 'N/A';
    }
}

// 使用例
let perfMonitor;

function setup() {
    createCanvas(640, 480);
    perfMonitor = new PerformanceMonitor();
}

function draw() {
    // FPSを更新
    perfMonitor.updateFPS();

    // パフォーマンス情報を表示
    perfMonitor.display();
}
```

## 練習問題

### 問題1: 環境構築
ml5.jsを読み込む3つの方法を挙げ、それぞれのメリット・デメリットを説明してください。

### 問題2: エラー処理
以下のコードにエラー処理を追加して、モデルの読み込み失敗時にユーザーに通知する機能を実装してください。

```javascript
let classifier;

function setup() {
    createCanvas(400, 400);
    // ここにエラー処理を追加
    classifier = ml5.imageClassifier('MobileNet');
}
```

### 問題3: デバッグ
以下のコードが動作しない理由を特定し、修正してください。

```javascript
function setup() {
    let img = loadImage('cat.jpg');
    let classifier = ml5.imageClassifier('MobileNet');
    classifier.classify(img, (err, results) => {
        console.log(results);
    });
}
```

### 問題4: 実装課題
以下の機能を持つプログラムを作成してください：
- 画像をドラッグ&ドロップで読み込める
- 読み込んだ画像を自動的に分類
- 分類結果をグラフィカルに表示（確信度のバーグラフなど）
- エラーハンドリングを適切に実装

## 解答

### 問題1の解答

**ml5.jsを読み込む3つの方法：**

1. **CDN（Content Delivery Network）**
   - メリット：
     - セットアップが簡単
     - 常に最新版または指定版を利用可能
     - キャッシュによる高速読み込み
   - デメリット：
     - インターネット接続が必要
     - 外部サービスに依存
     - オフライン環境で動作しない

2. **NPM（Node Package Manager）**
   - メリット：
     - バージョン管理が容易
     - 他のNPMパッケージとの統合が簡単
     - ビルドツールとの連携が可能
   - デメリット：
     - Node.js環境が必要
     - 初期設定が複雑
     - ビルドプロセスが必要な場合がある

3. **ローカルファイル**
   - メリット：
     - オフラインで動作可能
     - 読み込み速度が安定
     - 外部サービスに依存しない
   - デメリット：
     - 手動でファイルをダウンロード・更新する必要がある
     - バージョン管理が面倒
     - ファイルサイズが大きい

### 問題2の解答

```javascript
let classifier;
let statusText = '初期化中...';

function setup() {
    createCanvas(400, 400);

    // エラー処理を追加
    try {
        // モデル読み込みにコールバックを追加
        classifier = ml5.imageClassifier('MobileNet', modelLoaded, modelError);
    } catch (error) {
        // 同期的なエラーをキャッチ
        console.error('初期化エラー:', error);
        statusText = 'エラー: ml5.jsの初期化に失敗しました';
    }
}

function modelLoaded() {
    console.log('モデルの読み込み成功！');
    statusText = 'モデル準備完了';
}

function modelError(error) {
    console.error('モデル読み込みエラー:', error);
    statusText = `エラー: ${error.message || 'モデルの読み込みに失敗しました'}`;

    // エラー内容に応じた対処法を表示
    if (error.message && error.message.includes('network')) {
        statusText += '\n→ ネットワーク接続を確認してください';
    } else if (error.message && error.message.includes('model')) {
        statusText += '\n→ モデル名を確認してください';
    }
}

function draw() {
    background(240);

    // ステータスを表示
    fill(0);
    textAlign(CENTER, CENTER);
    textSize(16);
    text(statusText, width / 2, height / 2);
}
```

### 問題3の解答

**問題の原因：**
`loadImage()`は非同期関数のため、画像の読み込みが完了する前に分類が実行されてしまう。

**修正版：**

```javascript
let img;
let classifier;

// preloadを使用して、setup前に画像とモデルを読み込む
function preload() {
    img = loadImage('cat.jpg');
    classifier = ml5.imageClassifier('MobileNet');
}

function setup() {
    createCanvas(400, 400);

    // 画像とモデルの準備ができてから分類を実行
    classifier.classify(img, (err, results) => {
        if (err) {
            console.error('分類エラー:', err);
            return;
        }
        console.log('分類結果:', results);
    });

    // 画像を表示
    image(img, 0, 0, width, height);
}

// または、コールバックを使用する方法：
function setup() {
    createCanvas(400, 400);

    // モデルを先に読み込み
    classifier = ml5.imageClassifier('MobileNet', () => {
        // モデルの準備ができてから画像を読み込み
        loadImage('cat.jpg', (img) => {
            // 両方の準備ができてから分類
            classifier.classify(img, (err, results) => {
                if (err) {
                    console.error('分類エラー:', err);
                    return;
                }
                console.log('分類結果:', results);

                // 画像を表示
                image(img, 0, 0, width, height);
            });
        });
    });
}
```

### 問題4の解答

```javascript
// ドラッグ&ドロップ画像分類プログラム
let classifier;
let img = null;
let results = [];
let isClassifying = false;
let statusMessage = 'モデルを読み込み中...';

function setup() {
    createCanvas(800, 600);

    // ドラッグ&ドロップエリアの設定
    setupDragAndDrop();

    // MobileNetモデルを読み込み
    classifier = ml5.imageClassifier('MobileNet', modelReady);
}

function modelReady() {
    console.log('モデル準備完了');
    statusMessage = '画像をドラッグ&ドロップしてください';
}

function setupDragAndDrop() {
    // キャンバス要素を取得
    let canvas = document.querySelector('canvas');

    // ドラッグオーバーイベント
    canvas.addEventListener('dragover', (e) => {
        e.preventDefault();
        canvas.style.border = '3px dashed #00ff00';
    });

    // ドラッグリーブイベント
    canvas.addEventListener('dragleave', (e) => {
        e.preventDefault();
        canvas.style.border = 'none';
    });

    // ドロップイベント
    canvas.addEventListener('drop', (e) => {
        e.preventDefault();
        canvas.style.border = 'none';

        // ファイルを取得
        let file = e.dataTransfer.files[0];

        // 画像ファイルかチェック
        if (file && file.type.startsWith('image/')) {
            handleImageFile(file);
        } else {
            statusMessage = 'エラー: 画像ファイルを選択してください';
        }
    });
}

function handleImageFile(file) {
    statusMessage = '画像を読み込み中...';

    // FileReaderを使用して画像を読み込み
    let reader = new FileReader();

    reader.onload = (event) => {
        // 画像をロード
        loadImage(event.target.result, (loadedImg) => {
            img = loadedImg;
            statusMessage = '分類中...';
            isClassifying = true;

            // 画像を分類
            classifier.classify(img, gotResult);
        }, (error) => {
            console.error('画像読み込みエラー:', error);
            statusMessage = 'エラー: 画像の読み込みに失敗しました';
        });
    };

    reader.onerror = (error) => {
        console.error('ファイル読み込みエラー:', error);
        statusMessage = 'エラー: ファイルの読み込みに失敗しました';
    };

    // ファイルをData URLとして読み込み
    reader.readAsDataURL(file);
}

function gotResult(error, res) {
    isClassifying = false;

    if (error) {
        console.error('分類エラー:', error);
        statusMessage = `エラー: ${error.message}`;
        results = [];
        return;
    }

    results = res;
    statusMessage = '分類完了！';

    // コンソールに詳細を出力
    console.log('分類結果:');
    results.forEach((r, i) => {
        console.log(`${i + 1}. ${r.label}: ${(r.confidence * 100).toFixed(2)}%`);
    });
}

function draw() {
    background(240);

    // ドロップエリアの枠を描画
    if (!img) {
        stroke(150);
        strokeWeight(2);
        fill(250);
        rect(50, 50, width - 100, height - 200);

        // インストラクション
        noStroke();
        fill(100);
        textAlign(CENTER, CENTER);
        textSize(20);
        text('画像をここにドラッグ&ドロップ', width / 2, height / 2 - 100);

        // アイコン
        textSize(50);
        text('📁', width / 2, height / 2 - 50);
    }

    // 画像が存在する場合
    if (img) {
        // 画像のアスペクト比を保持しながら表示
        let imgWidth = img.width;
        let imgHeight = img.height;
        let maxWidth = width - 100;
        let maxHeight = height - 250;

        // スケールを計算
        let scale = min(maxWidth / imgWidth, maxHeight / imgHeight);
        imgWidth *= scale;
        imgHeight *= scale;

        // 画像を中央に配置
        let x = (width - imgWidth) / 2;
        let y = 50;

        image(img, x, y, imgWidth, imgHeight);

        // 画像の枠
        noFill();
        stroke(0);
        strokeWeight(2);
        rect(x, y, imgWidth, imgHeight);
    }

    // 結果の表示エリア
    if (results.length > 0) {
        // 背景
        fill(0, 0, 0, 230);
        rect(0, height - 150, width, 150);

        // タイトル
        fill(255);
        textAlign(LEFT);
        textSize(14);
        text('分類結果:', 20, height - 130);

        // Top 3の結果を表示
        for (let i = 0; i < min(3, results.length); i++) {
            let y = height - 100 + (i * 30);

            // ラベル
            fill(255);
            textSize(16);
            text(`${i + 1}. ${results[i].label}`, 20, y);

            // 確信度
            let confidence = results[i].confidence;
            textSize(12);
            text(`${(confidence * 100).toFixed(1)}%`, 300, y);

            // バーグラフ
            fill(0, 255, 0, 200);
            rect(350, y - 12, confidence * 400, 20);

            // バーの枠
            noFill();
            stroke(255);
            strokeWeight(1);
            rect(350, y - 12, 400, 20);
        }
    }

    // ステータスメッセージ
    fill(0);
    textAlign(CENTER);
    textSize(16);
    text(statusMessage, width / 2, 30);

    // 処理中インジケータ
    if (isClassifying) {
        push();
        translate(width / 2 - 100, 25);
        rotate(frameCount * 0.1);
        noFill();
        stroke(0, 150, 255);
        strokeWeight(3);
        arc(0, 0, 20, 20, 0, PI + HALF_PI);
        pop();
    }

    // リセットボタン
    if (img) {
        fill(255, 100, 100);
        rect(width - 100, 10, 80, 30);
        fill(255);
        textAlign(CENTER, CENTER);
        textSize(14);
        text('リセット', width - 60, 25);
    }
}

// マウスクリック処理
function mousePressed() {
    // リセットボタンのクリック判定
    if (img && mouseX > width - 100 && mouseX < width - 20 &&
        mouseY > 10 && mouseY < 40) {
        // リセット
        img = null;
        results = [];
        statusMessage = '画像をドラッグ&ドロップしてください';
    }
}

// ファイル選択による画像読み込み（代替手段）
function keyPressed() {
    if (key === 'o' || key === 'O') {
        // ファイル選択ダイアログを開く
        let input = createFileInput(handleFile);
        input.position(-1000, -1000);  // 画面外に配置
        input.elt.click();  // クリックイベントを発火
        input.remove();  // 使用後に削除
    }
}

function handleFile(file) {
    if (file.type === 'image') {
        handleImageFile(file.file);
    }
}
```

## まとめ

第2章では、ml5.jsの開発環境の構築からセットアップ、最初のプログラム作成、そしてデバッグ方法まで学習しました。適切な環境設定とエラーハンドリングは、安定したアプリケーション開発の基礎となります。

次章では、ml5.jsの中核機能である画像分類について、より詳しく学習していきます。