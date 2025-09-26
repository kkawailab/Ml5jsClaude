# 第3章: 画像分類の基礎

## 3.1 画像分類とは

### 画像分類の概要

画像分類（Image Classification）は、画像を入力として受け取り、事前に定義されたカテゴリー（クラス）のいずれかに分類する機械学習タスクです。

### 画像分類の仕組み

1. **特徴抽出**: 画像から重要な特徴（エッジ、形状、テクスチャなど）を抽出
2. **パターン認識**: 抽出した特徴をもとに、学習済みのパターンと照合
3. **確率計算**: 各カテゴリーに属する確率を計算
4. **分類決定**: 最も確率の高いカテゴリーを選択

### ml5.jsで使用できる画像分類モデル

```javascript
// 1. MobileNet - 1000カテゴリーの一般物体認識
const mobileNet = ml5.imageClassifier('MobileNet', callback);

// 2. DarkNet - より高精度な分類（Tiny DarkNet）
const darkNet = ml5.imageClassifier('DarkNet', callback);

// 3. DoodleNet - 手描きスケッチの分類（345カテゴリー）
const doodleNet = ml5.imageClassifier('DoodleNet', callback);

// 4. カスタムモデル（Teachable Machineで作成）
const customModel = ml5.imageClassifier('./model/model.json', callback);
```

## 3.2 MobileNetを使った画像認識

### MobileNetとは

MobileNetは、モバイルデバイスでも効率的に動作するよう設計された軽量な畳み込みニューラルネットワーク（CNN）です。ImageNetデータセット（1000カテゴリー）で学習されています。

### 基本的な画像分類

```javascript
// 静止画像の分類
let classifier;
let img;
let results = [];

function preload() {
    // 画像を読み込み
    img = loadImage('assets/dog.jpg');

    // MobileNetモデルを読み込み
    classifier = ml5.imageClassifier('MobileNet', modelLoaded);
}

function modelLoaded() {
    console.log('MobileNetモデル読み込み完了');

    // 画像を分類（デフォルトでTop 3の結果を返す）
    classifier.classify(img, gotResult);
}

function setup() {
    createCanvas(640, 480);

    // 画像を表示
    image(img, 0, 0, width, height);
}

function gotResult(error, res) {
    if (error) {
        console.error(error);
        return;
    }

    results = res;
    displayResults();
}

function displayResults() {
    // 結果表示用の背景
    fill(0, 0, 0, 200);
    rect(0, height - 120, width, 120);

    // 結果をテキストで表示
    fill(255);
    textSize(16);
    let y = height - 95;

    results.forEach((result, i) => {
        // ラベルと確信度を表示
        text(`${i + 1}. ${result.label}`, 20, y + i * 30);

        // 確信度のバー
        fill(0, 255, 0);
        let barWidth = result.confidence * 200;
        rect(250, y + i * 30 - 15, barWidth, 20);

        // パーセンテージ
        fill(255);
        text(`${(result.confidence * 100).toFixed(1)}%`, 460, y + i * 30);
    });
}
```

### 複数画像のバッチ分類

```javascript
// 複数の画像を順番に分類
let classifier;
let images = [];
let currentIndex = 0;
let classificationResults = [];

function preload() {
    // 複数の画像を読み込み
    for (let i = 1; i <= 5; i++) {
        images.push(loadImage(`assets/image${i}.jpg`));
    }

    // MobileNetモデルを読み込み
    classifier = ml5.imageClassifier('MobileNet', modelLoaded);
}

function modelLoaded() {
    console.log('モデル準備完了');
    classifyNextImage();
}

function classifyNextImage() {
    if (currentIndex < images.length) {
        // 現在の画像を分類
        console.log(`画像 ${currentIndex + 1}/${images.length} を分類中...`);

        classifier.classify(images[currentIndex], (error, results) => {
            if (error) {
                console.error(error);
                return;
            }

            // 結果を保存
            classificationResults.push({
                imageIndex: currentIndex,
                image: images[currentIndex],
                results: results
            });

            // 次の画像へ
            currentIndex++;
            classifyNextImage();
        });
    } else {
        // すべての分類が完了
        console.log('すべての画像の分類が完了しました');
        displayAllResults();
    }
}

function displayAllResults() {
    // 結果をグリッド表示
    let cols = 3;
    let rows = Math.ceil(images.length / cols);
    let cellWidth = width / cols;
    let cellHeight = height / rows;

    classificationResults.forEach((item, index) => {
        let col = index % cols;
        let row = Math.floor(index / cols);
        let x = col * cellWidth;
        let y = row * cellHeight;

        // 画像を表示
        image(item.image, x, y, cellWidth - 10, cellHeight - 30);

        // ラベルを表示
        fill(0);
        textSize(12);
        text(item.results[0].label, x + 5, y + cellHeight - 10);
    });
}
```

### 分類オプションのカスタマイズ

```javascript
// カスタム設定での分類
let classifier;

function setup() {
    createCanvas(640, 480);

    // オプションを指定してモデルを作成
    const options = {
        version: 1,  // MobileNetのバージョン（1または2）
        alpha: 0.25, // モデルの幅の乗数（0.25, 0.50, 0.75, 1.0）
        topk: 5      // 返す結果の数
    };

    classifier = ml5.imageClassifier('MobileNet', options, modelReady);
}

function modelReady() {
    console.log('カスタム設定のMobileNet準備完了');
}

// 分類時にも結果数を指定可能
function classifyWithOptions(img) {
    // Top 10の結果を取得
    classifier.classify(img, 10, (error, results) => {
        if (error) {
            console.error(error);
            return;
        }

        // 特定の信頼度以上の結果のみをフィルタ
        let filteredResults = results.filter(r => r.confidence > 0.01);

        console.log('フィルタされた結果:', filteredResults);
    });
}
```

## 3.3 Webカメラからのリアルタイム画像分類

### 基本的なリアルタイム分類

```javascript
// Webカメラのリアルタイム画像分類
let video;
let classifier;
let label = '読み込み中...';
let confidence = 0;

function setup() {
    createCanvas(640, 520);

    // Webカメラの映像を取得
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // MobileNetを動画用に初期化
    classifier = ml5.imageClassifier('MobileNet', video, modelReady);
}

function modelReady() {
    console.log('モデル準備完了！');
    // 分類を開始
    classifyVideo();
}

function classifyVideo() {
    classifier.classify(gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // 結果を更新
    label = results[0].label;
    confidence = results[0].confidence;

    // 継続的に分類（再帰的に呼び出し）
    classifyVideo();
}

function draw() {
    // ビデオを描画
    image(video, 0, 0, width, 480);

    // 結果表示エリア
    fill(0);
    rect(0, 480, width, 40);

    // 結果テキスト
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(20);
    text(`${label} (${(confidence * 100).toFixed(1)}%)`, width / 2, 500);

    // 信頼度インジケーター
    drawConfidenceBar(10, 10, confidence);
}

function drawConfidenceBar(x, y, conf) {
    // 背景
    fill(0, 0, 0, 100);
    rect(x, y, 200, 30);

    // 信頼度に応じて色を変更
    if (conf > 0.7) {
        fill(0, 255, 0);
    } else if (conf > 0.4) {
        fill(255, 255, 0);
    } else {
        fill(255, 0, 0);
    }

    // バー
    rect(x, y, conf * 200, 30);

    // テキスト
    fill(255);
    textAlign(LEFT, CENTER);
    textSize(14);
    text('信頼度', x + 5, y + 15);
}
```

### パフォーマンス最適化されたリアルタイム分類

```javascript
// フレームレート制御とパフォーマンス監視
let video;
let classifier;
let label = '';
let confidence = 0;
let classificationInterval = 500; // 500ms毎に分類
let lastClassificationTime = 0;
let fps = 0;
let avgClassificationTime = 0;
let classificationTimes = [];

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // パフォーマンス設定付きでモデルを初期化
    const options = {
        version: 2,    // MobileNet v2を使用
        alpha: 0.5,    // 中程度の精度とスピード
        topk: 3        // Top 3のみ取得
    };

    classifier = ml5.imageClassifier('MobileNet', video, options, modelReady);
}

function modelReady() {
    console.log('最適化されたモデル準備完了');
}

function draw() {
    // FPS計算
    fps = frameRate();

    // ビデオ描画
    image(video, 0, 0);

    // 指定間隔で分類を実行
    if (millis() - lastClassificationTime > classificationInterval) {
        lastClassificationTime = millis();

        let startTime = performance.now();

        classifier.classify((error, results) => {
            if (!error) {
                label = results[0].label;
                confidence = results[0].confidence;

                // 分類時間を記録
                let classTime = performance.now() - startTime;
                classificationTimes.push(classTime);

                // 最新10回の平均を計算
                if (classificationTimes.length > 10) {
                    classificationTimes.shift();
                }
                avgClassificationTime = classificationTimes.reduce((a, b) => a + b, 0) / classificationTimes.length;
            }
        });
    }

    // UI描画
    drawUI();
}

function drawUI() {
    // 半透明の背景
    fill(0, 0, 0, 150);
    rect(0, height - 80, width, 80);

    // 結果表示
    fill(255);
    textAlign(LEFT);
    textSize(16);
    text(`検出: ${label}`, 10, height - 55);
    text(`信頼度: ${(confidence * 100).toFixed(1)}%`, 10, height - 35);

    // パフォーマンス情報
    textAlign(RIGHT);
    textSize(12);
    fill(0, 255, 0);
    text(`FPS: ${fps.toFixed(1)}`, width - 10, height - 55);
    text(`分類時間: ${avgClassificationTime.toFixed(1)}ms`, width - 10, height - 35);
    text(`間隔: ${classificationInterval}ms`, width - 10, height - 15);

    // 間隔調整コントロール
    drawIntervalControl();
}

function drawIntervalControl() {
    // スライダー風のUI
    let sliderX = 10;
    let sliderY = 10;
    let sliderWidth = 200;
    let sliderHeight = 30;

    // 背景
    fill(0, 0, 0, 150);
    rect(sliderX, sliderY, sliderWidth, sliderHeight);

    // スライダー位置
    let sliderPos = map(classificationInterval, 100, 2000, sliderX, sliderX + sliderWidth);

    // スライダーバー
    stroke(255);
    strokeWeight(2);
    line(sliderX + 10, sliderY + 15, sliderX + sliderWidth - 10, sliderY + 15);

    // スライダーハンドル
    noStroke();
    fill(255, 100, 100);
    ellipse(sliderPos, sliderY + 15, 15, 15);

    // ラベル
    fill(255);
    textAlign(CENTER);
    textSize(10);
    text('分類間隔', sliderX + sliderWidth / 2, sliderY - 5);
}

function mousePressed() {
    // スライダーのインタラクション
    let sliderX = 10;
    let sliderY = 10;
    let sliderWidth = 200;
    let sliderHeight = 30;

    if (mouseX > sliderX && mouseX < sliderX + sliderWidth &&
        mouseY > sliderY && mouseY < sliderY + sliderHeight) {
        classificationInterval = map(mouseX, sliderX, sliderX + sliderWidth, 100, 2000);
    }
}
```

## 3.4 カスタム画像分類器の作成

### Teachable Machineとの連携

```javascript
// Teachable Machineで作成したモデルの使用
let classifier;
let video;
let label = '';
let confidence = 0;

// Teachable MachineのモデルURL
const modelURL = 'https://teachablemachine.withgoogle.com/models/[YOUR-MODEL-ID]/';

function preload() {
    // カスタムモデルを読み込み
    classifier = ml5.imageClassifier(modelURL + 'model.json');
}

function setup() {
    createCanvas(640, 480);

    // Webカメラを設定
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // 分類開始
    classifyVideo();
}

function classifyVideo() {
    classifier.classify(video, gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // カスタムクラスの結果を取得
    label = results[0].label;
    confidence = results[0].confidence;

    // 継続的に分類
    classifyVideo();
}

function draw() {
    // ビデオを描画
    image(video, 0, 0);

    // 結果を表示（カスタムクラス名）
    fill(255);
    textSize(32);
    textAlign(CENTER);
    text(label, width / 2, height - 20);

    // 信頼度メーター
    drawConfidenceMeter();
}

function drawConfidenceMeter() {
    push();
    translate(width - 100, 50);

    // メーターの背景
    stroke(255);
    strokeWeight(2);
    noFill();
    arc(0, 0, 80, 80, PI, TWO_PI);

    // 信頼度の表示
    let angle = map(confidence, 0, 1, PI, TWO_PI);
    stroke(0, 255, 0);
    strokeWeight(8);
    arc(0, 0, 80, 80, PI, angle);

    // パーセンテージ
    fill(255);
    noStroke();
    textAlign(CENTER);
    textSize(20);
    text((confidence * 100).toFixed(0) + '%', 0, 40);

    pop();
}
```

### 転移学習でカスタムモデルを作成

```javascript
// ml5.jsのfeatureExtractorを使った転移学習
let featureExtractor;
let classifier;
let video;
let label = 'トレーニング待機中';
let isTraining = false;

// 各クラスのサンプル数
let samples = {
    'クラス1': 0,
    'クラス2': 0,
    'クラス3': 0
};

function setup() {
    createCanvas(640, 580);

    // ビデオ設定
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // MobileNetを特徴抽出器として使用
    featureExtractor = ml5.featureExtractor('MobileNet', modelReady);

    // 分類器を作成
    classifier = featureExtractor.classification(video);

    // UIボタンを作成
    createTrainingButtons();
}

function modelReady() {
    console.log('特徴抽出器準備完了');
}

function createTrainingButtons() {
    // クラス1のボタン
    let button1 = createButton('クラス1を追加');
    button1.position(10, height - 60);
    button1.mousePressed(() => addExample('クラス1'));

    // クラス2のボタン
    let button2 = createButton('クラス2を追加');
    button2.position(120, height - 60);
    button2.mousePressed(() => addExample('クラス2'));

    // クラス3のボタン
    let button3 = createButton('クラス3を追加');
    button3.position(230, height - 60);
    button3.mousePressed(() => addExample('クラス3'));

    // トレーニング開始ボタン
    let trainButton = createButton('トレーニング開始');
    trainButton.position(340, height - 60);
    trainButton.mousePressed(trainModel);

    // 保存ボタン
    let saveButton = createButton('モデル保存');
    saveButton.position(470, height - 60);
    saveButton.mousePressed(saveModel);
}

function addExample(label) {
    // 現在のビデオフレームをサンプルとして追加
    classifier.addImage(label);
    samples[label]++;

    console.log(`${label}: ${samples[label]}サンプル`);
}

function trainModel() {
    if (!isTraining) {
        isTraining = true;
        label = 'トレーニング中...';

        // トレーニングを実行
        classifier.train((lossValue) => {
            if (lossValue) {
                label = `Loss: ${lossValue.toFixed(4)}`;
            } else {
                label = 'トレーニング完了！';
                isTraining = false;

                // 分類を開始
                classifyVideo();
            }
        });
    }
}

function classifyVideo() {
    if (!isTraining) {
        classifier.classify(gotResult);
    }
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // 結果を更新
    if (results && results[0]) {
        label = `${results[0].label} (${(results[0].confidence * 100).toFixed(1)}%)`;
    }

    // 継続的に分類
    classifyVideo();
}

function saveModel() {
    // モデルを保存（ダウンロード）
    classifier.save('my-model');
}

function draw() {
    // ビデオを描画
    image(video, 0, 0);

    // ステータス表示エリア
    fill(0, 0, 0, 180);
    rect(0, height - 100, width, 40);

    // ラベル表示
    fill(255);
    textAlign(CENTER);
    textSize(20);
    text(label, width / 2, height - 75);

    // サンプル数の表示
    textAlign(LEFT);
    textSize(14);
    let x = 10;
    for (let key in samples) {
        text(`${key}: ${samples[key]}`, x, height - 35);
        x += 120;
    }
}
```

### データ拡張テクニック

```javascript
// データ拡張を使った堅牢なモデル作成
let featureExtractor;
let classifier;
let video;

// データ拡張の設定
let augmentationOptions = {
    flipHorizontal: true,  // 水平反転
    rotation: 15,          // ランダム回転（度）
    zoom: 0.1,            // ランダムズーム
    brightness: 0.2,      // 明度調整
    noise: 0.05          // ノイズ追加
};

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // 特徴抽出器を初期化
    featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
    classifier = featureExtractor.classification(video);
}

function addAugmentedExamples(label, count) {
    for (let i = 0; i < count; i++) {
        // キャンバスにビデオを描画
        push();

        // ランダムな変換を適用
        translate(width / 2, height / 2);

        // 水平反転
        if (augmentationOptions.flipHorizontal && random() > 0.5) {
            scale(-1, 1);
        }

        // 回転
        if (augmentationOptions.rotation > 0) {
            rotate(radians(random(-augmentationOptions.rotation, augmentationOptions.rotation)));
        }

        // ズーム
        if (augmentationOptions.zoom > 0) {
            let zoomFactor = 1 + random(-augmentationOptions.zoom, augmentationOptions.zoom);
            scale(zoomFactor);
        }

        // ビデオを描画
        imageMode(CENTER);
        image(video, 0, 0);

        pop();

        // 明度調整
        if (augmentationOptions.brightness > 0) {
            let brightnessFactor = random(-augmentationOptions.brightness, augmentationOptions.brightness);
            push();
            blendMode(brightnessFactor > 0 ? ADD : MULTIPLY);
            fill(255, abs(brightnessFactor * 255));
            rect(0, 0, width, height);
            pop();
        }

        // ノイズ追加
        if (augmentationOptions.noise > 0) {
            addNoise(augmentationOptions.noise);
        }

        // 拡張された画像をキャプチャして追加
        let augmentedImage = get();
        classifier.addImage(augmentedImage, label);
    }

    console.log(`${count}個の拡張サンプルを${label}に追加`);
}

function addNoise(amount) {
    loadPixels();
    for (let i = 0; i < pixels.length; i += 4) {
        let noise = random(-amount * 255, amount * 255);
        pixels[i] += noise;     // R
        pixels[i + 1] += noise; // G
        pixels[i + 2] += noise; // B
    }
    updatePixels();
}
```

## 練習問題

### 問題1: 基本理解
MobileNetが分類できるカテゴリー数はいくつですか？また、なぜその数なのか説明してください。

### 問題2: コード実装
画像分類の結果をCSVファイルとして保存する機能を実装してください。

### 問題3: リアルタイム分類
Webカメラの映像から特定の物体（例：「cell phone」）が検出されたときだけ、画面にエフェクトを表示するプログラムを作成してください。

### 問題4: カスタム分類器
じゃんけん（グー・チョキ・パー）を認識する分類器を作成してください。データ収集、トレーニング、テストの全工程を実装してください。

## 解答

### 問題1の解答
**答え: 1000カテゴリー**

MobileNetは、ImageNetデータセットで学習されています。ImageNet Large Scale Visual Recognition Challenge（ILSVRC）では1000個のオブジェクトカテゴリーが使用されており、これには日常的な物体、動物、乗り物などが含まれています。この数は、一般的な画像認識タスクに必要十分なカバレッジを提供しながら、モデルのサイズと計算効率のバランスを保つために選ばれました。

### 問題2の解答

```javascript
// 画像分類結果をCSVとして保存
let classifier;
let classificationHistory = [];
let saveButton;

function setup() {
    createCanvas(640, 480);

    // MobileNetモデルを初期化
    classifier = ml5.imageClassifier('MobileNet', modelReady);

    // 保存ボタンを作成
    saveButton = createButton('結果をCSVで保存');
    saveButton.position(10, height + 10);
    saveButton.mousePressed(saveResultsAsCSV);
}

function classifyImage(img, imageName) {
    classifier.classify(img, (error, results) => {
        if (error) {
            console.error(error);
            return;
        }

        // タイムスタンプを追加
        let timestamp = new Date().toISOString();

        // 結果を履歴に保存
        let entry = {
            timestamp: timestamp,
            imageName: imageName || 'unknown',
            predictions: results.map(r => ({
                label: r.label,
                confidence: r.confidence
            }))
        };

        classificationHistory.push(entry);
        console.log('分類完了:', entry);
    });
}

function saveResultsAsCSV() {
    if (classificationHistory.length === 0) {
        console.warn('保存する結果がありません');
        return;
    }

    // CSVヘッダー
    let csv = 'Timestamp,Image Name,Label 1,Confidence 1,Label 2,Confidence 2,Label 3,Confidence 3\n';

    // データ行を追加
    classificationHistory.forEach(entry => {
        let row = [
            entry.timestamp,
            entry.imageName
        ];

        // Top 3の予測を追加
        for (let i = 0; i < 3; i++) {
            if (entry.predictions[i]) {
                row.push(entry.predictions[i].label);
                row.push(entry.predictions[i].confidence.toFixed(4));
            } else {
                row.push('');
                row.push('');
            }
        }

        csv += row.join(',') + '\n';
    });

    // CSVファイルとしてダウンロード
    let blob = new Blob([csv], { type: 'text/csv' });
    let url = URL.createObjectURL(blob);
    let a = document.createElement('a');
    a.href = url;
    a.download = `classification_results_${Date.now()}.csv`;
    a.click();

    console.log('CSVファイルを保存しました');
}

// 複数画像のバッチ処理
function processMultipleImages(imageFiles) {
    imageFiles.forEach((file, index) => {
        loadImage(file.data, (img) => {
            classifyImage(img, file.name);
        });
    });
}
```

### 問題3の解答

```javascript
// 特定物体検出時のエフェクト表示
let video;
let classifier;
let targetObject = 'cell phone';  // 検出対象
let isDetected = false;
let detectionConfidence = 0;
let effectIntensity = 0;
let particles = [];

function setup() {
    createCanvas(640, 480);

    // ビデオ設定
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // MobileNet初期化
    classifier = ml5.imageClassifier('MobileNet', video, modelReady);
}

function modelReady() {
    console.log('モデル準備完了');
    classifyVideo();
}

function classifyVideo() {
    classifier.classify(gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // 対象物体が検出されたか確認
    isDetected = false;
    for (let result of results) {
        if (result.label.toLowerCase().includes(targetObject.toLowerCase())) {
            isDetected = true;
            detectionConfidence = result.confidence;

            // エフェクト用のパーティクルを生成
            if (detectionConfidence > 0.5) {
                for (let i = 0; i < 5; i++) {
                    particles.push(new Particle(random(width), random(height)));
                }
            }
            break;
        }
    }

    // エフェクトの強度を更新（スムーズな遷移）
    if (isDetected) {
        effectIntensity = lerp(effectIntensity, detectionConfidence, 0.1);
    } else {
        effectIntensity = lerp(effectIntensity, 0, 0.1);
    }

    classifyVideo();
}

function draw() {
    // ビデオを描画
    image(video, 0, 0);

    // 検出時のエフェクト
    if (effectIntensity > 0.01) {
        // グローエフェクト
        drawGlowEffect();

        // パーティクルエフェクト
        updateAndDrawParticles();

        // 検出インジケーター
        drawDetectionIndicator();
    }

    // ステータス表示
    drawStatus();
}

function drawGlowEffect() {
    // 画面全体に色付きのグロー
    push();
    blendMode(ADD);
    fill(0, 255, 255, effectIntensity * 30);
    rect(0, 0, width, height);

    // 枠のグロー
    strokeWeight(5);
    stroke(0, 255, 255, effectIntensity * 200);
    noFill();
    rect(5, 5, width - 10, height - 10);
    pop();
}

function drawDetectionIndicator() {
    push();

    // レーダー風の円
    translate(width - 100, 100);
    rotate(frameCount * 0.02);

    strokeWeight(2);
    stroke(0, 255, 0, effectIntensity * 255);
    noFill();

    for (let i = 0; i < 3; i++) {
        let size = 30 + i * 20;
        ellipse(0, 0, size, size);
    }

    // スキャンライン
    stroke(0, 255, 0, effectIntensity * 255);
    line(0, 0, 40, 0);

    pop();

    // 検出ラベル
    fill(0, 255, 0, effectIntensity * 255);
    textAlign(CENTER);
    textSize(24);
    text('DETECTED!', width / 2, 50);
}

function drawStatus() {
    // ステータスバー
    fill(0, 0, 0, 180);
    rect(0, height - 40, width, 40);

    fill(255);
    textAlign(LEFT);
    textSize(16);
    text(`検出対象: ${targetObject}`, 10, height - 15);

    if (isDetected) {
        fill(0, 255, 0);
        text(`検出中 (${(detectionConfidence * 100).toFixed(1)}%)`, 200, height - 15);
    } else {
        fill(255, 100, 100);
        text('未検出', 200, height - 15);
    }
}

// パーティクルクラス
class Particle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.vx = random(-2, 2);
        this.vy = random(-2, 2);
        this.life = 255;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life -= 5;
    }

    display() {
        push();
        fill(0, 255, 255, this.life * effectIntensity);
        noStroke();
        ellipse(this.x, this.y, 10, 10);
        pop();
    }

    isDead() {
        return this.life <= 0;
    }
}

function updateAndDrawParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        let p = particles[i];
        p.update();
        p.display();

        if (p.isDead()) {
            particles.splice(i, 1);
        }
    }
}

// キーボードで検出対象を変更
function keyPressed() {
    if (key === '1') {
        targetObject = 'cell phone';
    } else if (key === '2') {
        targetObject = 'laptop';
    } else if (key === '3') {
        targetObject = 'person';
    }
}
```

### 問題4の解答

```javascript
// じゃんけん認識システム
let featureExtractor;
let classifier;
let video;
let label = 'トレーニング前';
let confidence = 0;

// ゲームの状態
let gameState = 'collecting'; // 'collecting', 'training', 'playing'
let samples = {
    'グー': 0,
    'チョキ': 0,
    'パー': 0
};

// ゲームのスコア
let playerScore = 0;
let computerScore = 0;
let lastComputerMove = '';

function setup() {
    createCanvas(640, 580);

    // ビデオ設定
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // 特徴抽出器と分類器を初期化
    featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
    classifier = featureExtractor.classification(video);

    // UIを作成
    createUI();
}

function modelReady() {
    console.log('じゃんけん認識システム準備完了');
}

function createUI() {
    // データ収集ボタン
    let rockBtn = createButton('グー (R)');
    rockBtn.position(10, height - 60);
    rockBtn.mousePressed(() => addTrainingData('グー'));

    let scissorsBtn = createButton('チョキ (S)');
    scissorsBtn.position(100, height - 60);
    scissorsBtn.mousePressed(() => addTrainingData('チョキ'));

    let paperBtn = createButton('パー (P)');
    paperBtn.position(200, height - 60);
    paperBtn.mousePressed(() => addTrainingData('パー'));

    // トレーニングボタン
    let trainBtn = createButton('トレーニング開始');
    trainBtn.position(300, height - 60);
    trainBtn.mousePressed(startTraining);

    // ゲーム開始ボタン
    let playBtn = createButton('じゃんけん開始');
    playBtn.position(440, height - 60);
    playBtn.mousePressed(startGame);

    // リセットボタン
    let resetBtn = createButton('リセット');
    resetBtn.position(560, height - 60);
    resetBtn.mousePressed(resetAll);
}

function addTrainingData(gesture) {
    if (gameState === 'collecting') {
        classifier.addImage(gesture);
        samples[gesture]++;

        // 効果音的な視覚フィードバック
        flashEffect();
    }
}

function flashEffect() {
    push();
    fill(255, 255, 255, 100);
    rect(0, 0, width, height);
    pop();
}

function startTraining() {
    if (gameState === 'collecting') {
        // 最小サンプル数をチェック
        let minSamples = Math.min(samples['グー'], samples['チョキ'], samples['パー']);

        if (minSamples < 10) {
            label = '各ジェスチャー最低10サンプル必要';
            return;
        }

        gameState = 'training';
        label = 'トレーニング中...';

        classifier.train((lossValue) => {
            if (lossValue) {
                label = `トレーニング中... Loss: ${lossValue.toFixed(4)}`;
            } else {
                label = 'トレーニング完了！';
                gameState = 'ready';
            }
        });
    }
}

function startGame() {
    if (gameState === 'ready' || gameState === 'playing') {
        gameState = 'playing';
        classifyGesture();
    }
}

function classifyGesture() {
    if (gameState === 'playing') {
        classifier.classify(gotResult);
    }
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    if (results && results[0]) {
        label = results[0].label;
        confidence = results[0].confidence;

        // 高い信頼度の場合のみゲーム判定
        if (confidence > 0.8 && frameCount % 60 === 0) {
            playRound(label);
        }
    }

    // 継続的に分類
    classifyGesture();
}

function playRound(playerMove) {
    // コンピューターの手を決定
    let moves = ['グー', 'チョキ', 'パー'];
    lastComputerMove = random(moves);

    // 勝敗判定
    let result = judgeWinner(playerMove, lastComputerMove);

    if (result === 'win') {
        playerScore++;
    } else if (result === 'lose') {
        computerScore++;
    }
}

function judgeWinner(player, computer) {
    if (player === computer) {
        return 'draw';
    }

    if ((player === 'グー' && computer === 'チョキ') ||
        (player === 'チョキ' && computer === 'パー') ||
        (player === 'パー' && computer === 'グー')) {
        return 'win';
    }

    return 'lose';
}

function resetAll() {
    gameState = 'collecting';
    samples = { 'グー': 0, 'チョキ': 0, 'パー': 0 };
    playerScore = 0;
    computerScore = 0;
    label = 'リセット完了';
}

function draw() {
    // ビデオを描画
    image(video, 0, 0);

    // ゲーム状態に応じたUI表示
    drawGameUI();

    // サンプル数表示
    drawSampleCount();

    // ゲームスコア表示
    if (gameState === 'playing') {
        drawGameScore();
    }
}

function drawGameUI() {
    // 状態表示エリア
    fill(0, 0, 0, 180);
    rect(0, 0, width, 80);

    // タイトル
    fill(255);
    textAlign(CENTER);
    textSize(24);
    text('じゃんけん認識ゲーム', width / 2, 30);

    // 現在の状態
    textSize(16);
    text(`状態: ${gameState} | ${label}`, width / 2, 55);

    // 信頼度バー
    if (gameState === 'playing' && confidence > 0) {
        fill(0, 255, 0);
        rect(width / 2 - 100, 65, confidence * 200, 10);
    }
}

function drawSampleCount() {
    // サンプル数表示
    fill(0, 0, 0, 180);
    rect(0, height - 100, 200, 40);

    fill(255);
    textAlign(LEFT);
    textSize(12);
    text(`グー: ${samples['グー']} | チョキ: ${samples['チョキ']} | パー: ${samples['パー']}`, 10, height - 80);
}

function drawGameScore() {
    // スコアボード
    fill(0, 0, 0, 180);
    rect(width - 200, height - 150, 200, 150);

    fill(255);
    textAlign(CENTER);
    textSize(18);
    text('スコア', width - 100, height - 120);

    textSize(14);
    text(`あなた: ${playerScore}`, width - 100, height - 90);
    text(`コンピュータ: ${computerScore}`, width - 100, height - 65);

    if (lastComputerMove) {
        text(`相手の手: ${lastComputerMove}`, width - 100, height - 35);
    }
}

// キーボードショートカット
function keyPressed() {
    if (gameState === 'collecting') {
        if (key === 'r' || key === 'R') {
            addTrainingData('グー');
        } else if (key === 's' || key === 'S') {
            addTrainingData('チョキ');
        } else if (key === 'p' || key === 'P') {
            addTrainingData('パー');
        }
    }
}
```

## まとめ

第3章では、ml5.jsを使った画像分類の基礎から応用まで学習しました。MobileNetを使った基本的な分類から、Teachable Machineとの連携、転移学習を使ったカスタムモデルの作成まで、幅広い技術を習得しました。

次章では、人のポーズを検出するPoseNetについて学習します。