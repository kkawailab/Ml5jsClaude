# 第1章: ml5.jsとは - 導入編

## 1.1 ml5.jsの概要と特徴

### ml5.jsとは？

ml5.jsは、機械学習をより身近で使いやすくすることを目的として開発されたJavaScriptライブラリです。「ml」は Machine Learning（機械学習）、「5」は HTML5 を意味しています。

### 主な特徴

1. **使いやすさ**: 複雑な数学やアルゴリズムの知識なしに機械学習を利用できます
2. **ブラウザで動作**: 特別なサーバーやインストール不要で、Webブラウザ上で動作します
3. **豊富な事前学習済みモデル**: すぐに使える様々なAIモデルが用意されています
4. **p5.jsとの親和性**: クリエイティブコーディングライブラリp5.jsと組み合わせて使いやすい設計です
5. **オープンソース**: 無料で使用でき、コミュニティによって開発されています

### TensorFlow.jsとの関係

ml5.jsは内部でTensorFlow.jsを使用していますが、より高レベルなAPIを提供することで、初心者でも機械学習を扱いやすくしています。

```javascript
// TensorFlow.jsの場合（低レベル）
const model = await tf.loadLayersModel('path/to/model.json');
const prediction = model.predict(tf.tensor2d([[...]]));

// ml5.jsの場合（高レベル）
const classifier = ml5.imageClassifier('MobileNet', modelReady);
function modelReady() {
    classifier.classify(image, gotResult);
}
```

## 1.2 機械学習の基本概念

### 機械学習とは？

機械学習は、コンピュータがデータから自動的にパターンを学習し、予測や分類を行う技術です。

### 主要な用語

- **モデル（Model）**: 学習済みのパターンを持つプログラム
- **トレーニング（Training）**: モデルにデータからパターンを学習させるプロセス
- **推論（Inference）**: 学習済みモデルを使って新しいデータに対して予測を行うこと
- **分類（Classification）**: データをカテゴリーに分ける（例：犬か猫か）
- **回帰（Regression）**: 連続値を予測する（例：気温の予測）

### ml5.jsでの機械学習の流れ

```javascript
// 1. モデルの読み込み
let classifier;

// 2. モデルの初期化
function preload() {
    // MobileNetという事前学習済みモデルを読み込む
    classifier = ml5.imageClassifier('MobileNet');
}

// 3. 推論の実行
function setup() {
    // 画像に対して分類を実行
    classifier.classify(img, gotResult);
}

// 4. 結果の処理
function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }
    // 結果を表示
    console.log(results[0].label);  // 最も可能性の高いラベル
    console.log(results[0].confidence);  // 信頼度（0〜1）
}
```

## 1.3 ml5.jsでできること

### 1. 画像認識・分類

```javascript
// 画像分類の例
const classifier = ml5.imageClassifier('MobileNet', () => {
    console.log('モデル読み込み完了！');
});

// 画像を分類
classifier.classify(document.getElementById('myImage'), (err, results) => {
    if (err) {
        console.error(err);
        return;
    }
    // 結果を表示（例: [{label: "dog", confidence: 0.95}]）
    console.log('これは ' + results[0].label + ' です');
    console.log('確信度: ' + (results[0].confidence * 100).toFixed(2) + '%');
});
```

### 2. ポーズ検出

```javascript
// PoseNetで人のポーズを検出
let poseNet;
let poses = [];

function setup() {
    // Webカメラの映像を取得
    const video = createCapture(VIDEO);
    video.size(640, 480);

    // PoseNetモデルを初期化
    poseNet = ml5.poseNet(video, modelReady);

    // ポーズが検出されたときのコールバック
    poseNet.on('pose', function(results) {
        poses = results;
    });
}

function modelReady() {
    console.log('PoseNet準備完了！');
}

function draw() {
    // 検出された各ポーズのキーポイントを描画
    for (let i = 0; i < poses.length; i++) {
        let pose = poses[i].pose;
        // 鼻の位置に円を描画
        if (pose.nose) {
            fill(255, 0, 0);
            ellipse(pose.nose.x, pose.nose.y, 20, 20);
        }
    }
}
```

### 3. 物体検出

```javascript
// COCOSSD（物体検出モデル）の使用例
let objectDetector;
let detections = [];

function preload() {
    // COCOSSDモデルを読み込み
    objectDetector = ml5.objectDetector('cocossd');
}

function setup() {
    // 画像に対して物体検出を実行
    objectDetector.detect(img, gotDetections);
}

function gotDetections(error, results) {
    if (error) {
        console.error(error);
        return;
    }
    detections = results;

    // 検出された物体を表示
    detections.forEach(detection => {
        console.log(detection.label + ': ' + detection.confidence);
        // バウンディングボックスの座標
        console.log('位置:', detection.x, detection.y, detection.width, detection.height);
    });
}
```

### 4. 手の認識

```javascript
// HandPoseで手の関節を検出
let handpose;
let predictions = [];

function setup() {
    const video = createCapture(VIDEO);
    video.size(640, 480);

    // HandPoseモデルを初期化
    handpose = ml5.handpose(video, modelReady);

    // 手が検出されたときのコールバック
    handpose.on('predict', results => {
        predictions = results;
    });
}

function draw() {
    // 各手のランドマーク（関節点）を描画
    for (let i = 0; i < predictions.length; i++) {
        const keypoints = predictions[i].landmarks;

        // 21個の関節点を描画
        for (let j = 0; j < keypoints.length; j++) {
            const [x, y, z] = keypoints[j];
            fill(0, 255, 0);
            ellipse(x, y, 10, 10);
        }
    }
}
```

### 5. 音声認識

```javascript
// SoundClassifierで音声を分類
let soundClassifier;

function preload() {
    // 音声分類モデル（Speech Commands）を読み込み
    soundClassifier = ml5.soundClassifier('SpeechCommands18w', modelReady);
}

function modelReady() {
    // 音声の分類を開始
    soundClassifier.classify(gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }
    // 認識された音声コマンドを表示
    console.log('聞こえた言葉: ' + results[0].label);
    console.log('確信度: ' + results[0].confidence);
}
```

## 1.4 p5.jsとの連携

### p5.jsとは？

p5.jsは、クリエイティブコーディングのためのJavaScriptライブラリです。Processing言語をベースにしており、ビジュアルアートやインタラクティブなグラフィックスを簡単に作成できます。

### ml5.jsとp5.jsの組み合わせ

```html
<!DOCTYPE html>
<html>
<head>
    <title>ml5.js + p5.js デモ</title>
    <!-- p5.jsライブラリ -->
    <script src="https://cdn.jsdelivr.net/npm/p5@1.7.0/lib/p5.min.js"></script>
    <!-- ml5.jsライブラリ -->
    <script src="https://unpkg.com/ml5@latest/dist/ml5.min.js"></script>
</head>
<body>
    <script>
        let video;
        let poseNet;
        let poses = [];

        function setup() {
            // キャンバスを作成
            createCanvas(640, 480);

            // ビデオキャプチャを開始
            video = createCapture(VIDEO);
            video.size(width, height);
            video.hide(); // HTML要素としてのビデオを非表示

            // PoseNetモデルを初期化
            poseNet = ml5.poseNet(video, modelReady);

            // ポーズ検出のイベントリスナー
            poseNet.on('pose', function(results) {
                poses = results;
            });
        }

        function modelReady() {
            console.log('モデル準備完了！');
        }

        function draw() {
            // ビデオフレームを描画
            image(video, 0, 0, width, height);

            // 検出されたポーズを描画
            drawKeypoints();
            drawSkeleton();
        }

        // キーポイント（関節）を描画
        function drawKeypoints() {
            for (let i = 0; i < poses.length; i++) {
                let pose = poses[i].pose;

                for (let j = 0; j < pose.keypoints.length; j++) {
                    let keypoint = pose.keypoints[j];
                    // 信頼度が0.2以上の場合のみ描画
                    if (keypoint.score > 0.2) {
                        fill(255, 0, 0);
                        noStroke();
                        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
                    }
                }
            }
        }

        // スケルトン（骨格）を描画
        function drawSkeleton() {
            for (let i = 0; i < poses.length; i++) {
                let skeleton = poses[i].skeleton;

                for (let j = 0; j < skeleton.length; j++) {
                    let partA = skeleton[j][0];
                    let partB = skeleton[j][1];
                    stroke(255, 0, 0);
                    strokeWeight(2);
                    line(partA.position.x, partA.position.y,
                         partB.position.x, partB.position.y);
                }
            }
        }
    </script>
</body>
</html>
```

### p5.jsの基本関数

- **setup()**: プログラム開始時に1回だけ実行される
- **draw()**: 継続的に繰り返し実行される（デフォルトで60fps）
- **preload()**: setup()の前に実行され、リソースの読み込みに使用

### インタラクティブな要素の追加

```javascript
let classifier;
let imageModelURL = 'https://teachablemachine.withgoogle.com/models/[YOUR-MODEL]/';
let label = "待機中...";
let confidence = 0;

function preload() {
    // Teachable Machineで作成したカスタムモデルを読み込み
    classifier = ml5.imageClassifier(imageModelURL + 'model.json');
}

function setup() {
    createCanvas(640, 520);

    // ビデオ入力を開始
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // 分類を開始
    classifyVideo();
}

function classifyVideo() {
    classifier.classify(video, gotResults);
}

function gotResults(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // 結果を更新
    label = results[0].label;
    confidence = results[0].confidence;

    // 継続的に分類
    classifyVideo();
}

function draw() {
    background(0);

    // ビデオを表示
    image(video, 0, 0);

    // 半透明の背景
    fill(0, 0, 0, 150);
    rect(0, height - 40, width, 40);

    // 結果を表示
    fill(255);
    textSize(20);
    textAlign(CENTER);
    text(label + ' (' + nf(confidence * 100, 0, 2) + '%)', width / 2, height - 10);

    // 信頼度に応じて色を変更
    if (confidence > 0.8) {
        fill(0, 255, 0);  // 緑：高信頼度
    } else if (confidence > 0.5) {
        fill(255, 255, 0);  // 黄：中信頼度
    } else {
        fill(255, 0, 0);  // 赤：低信頼度
    }

    // 信頼度バーを描画
    rect(10, 10, confidence * 200, 20);
}
```

## 練習問題

### 問題1: 基本理解
ml5.jsの特徴として正しくないものはどれですか？
1. ブラウザ上で動作する
2. サーバーサイドでのみ動作する
3. 事前学習済みモデルが利用できる
4. TensorFlow.jsをベースにしている

### 問題2: コード理解
次のコードの空欄を埋めて、画像分類を実行するプログラムを完成させてください。

```javascript
let classifier;
let img;

function preload() {
    // 画像を読み込み
    img = loadImage('cat.jpg');
    // MobileNetモデルを読み込み
    classifier = ml5.________('MobileNet');
}

function setup() {
    createCanvas(400, 400);
    // 画像を分類
    classifier.________(img, gotResult);
    image(img, 0, 0, width, height);
}

function ________(error, results) {
    if (error) {
        console.error(error);
        return;
    }
    console.log(results);
}
```

### 問題3: 実装課題
PoseNetを使用して、鼻の位置を追跡し、その軌跡を描画するプログラムを作成してください。

### 問題4: 応用問題
ml5.jsのimageClassifierを使って、Webカメラの映像をリアルタイムで分類し、最も可能性の高い3つの結果を画面に表示するプログラムを作成してください。

## 解答

### 問題1の解答
**答え: 2. サーバーサイドでのみ動作する**

ml5.jsはブラウザ上で動作するライブラリです。Node.jsでも使用可能ですが、主にクライアントサイド（ブラウザ）での使用を想定して設計されています。

### 問題2の解答

```javascript
let classifier;
let img;

function preload() {
    // 画像を読み込み
    img = loadImage('cat.jpg');
    // MobileNetモデルを読み込み
    classifier = ml5.imageClassifier('MobileNet');  // 空欄1: imageClassifier
}

function setup() {
    createCanvas(400, 400);
    // 画像を分類
    classifier.classify(img, gotResult);  // 空欄2: classify
    image(img, 0, 0, width, height);
}

function gotResult(error, results) {  // 空欄3: gotResult
    if (error) {
        console.error(error);
        return;
    }
    console.log(results);
    // 結果を画面に表示
    fill(255);
    textSize(16);
    text(results[0].label + ': ' + nf(results[0].confidence, 0, 2), 10, height - 10);
}
```

### 問題3の解答

```javascript
let video;
let poseNet;
let poses = [];
let prevNoseX = 0;
let prevNoseY = 0;
let trail = [];  // 軌跡を保存する配列

function setup() {
    createCanvas(640, 480);

    // ビデオキャプチャを開始
    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    // PoseNetモデルを初期化
    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', function(results) {
        poses = results;
    });
}

function modelReady() {
    console.log('PoseNet準備完了！');
}

function draw() {
    // ビデオを描画
    image(video, 0, 0, width, height);

    // 軌跡を描画
    stroke(255, 0, 255);
    strokeWeight(3);
    noFill();
    beginShape();
    for (let i = 0; i < trail.length; i++) {
        vertex(trail[i].x, trail[i].y);
    }
    endShape();

    // 鼻の位置を取得して軌跡に追加
    if (poses.length > 0) {
        let pose = poses[0].pose;
        if (pose.nose) {
            let noseX = pose.nose.x;
            let noseY = pose.nose.y;

            // 軌跡に追加（移動があった場合のみ）
            if (dist(noseX, noseY, prevNoseX, prevNoseY) > 2) {
                trail.push({x: noseX, y: noseY});

                // 軌跡の長さを制限（メモリ節約）
                if (trail.length > 100) {
                    trail.shift();
                }

                prevNoseX = noseX;
                prevNoseY = noseY;
            }

            // 現在の鼻の位置に円を描画
            fill(255, 0, 0);
            noStroke();
            ellipse(noseX, noseY, 20, 20);
        }
    }

    // 軌跡をクリアするインストラクション
    fill(255);
    textSize(16);
    text('スペースキーで軌跡をクリア', 10, 30);
}

function keyPressed() {
    if (key === ' ') {
        trail = [];  // スペースキーで軌跡をクリア
    }
}
```

### 問題4の解答

```javascript
let video;
let classifier;
let results = [];

function setup() {
    createCanvas(640, 520);

    // ビデオキャプチャを開始
    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // MobileNetモデルを初期化
    classifier = ml5.imageClassifier('MobileNet', video, modelReady);
}

function modelReady() {
    console.log('モデル準備完了！');
    // 分類を開始
    classifyVideo();
}

function classifyVideo() {
    classifier.classify(gotResults);
}

function gotResults(error, res) {
    if (error) {
        console.error(error);
        return;
    }

    // 結果を保存
    results = res;

    // 継続的に分類
    classifyVideo();
}

function draw() {
    // ビデオを描画
    image(video, 0, 0, width, 480);

    // 半透明の背景を描画
    fill(0, 0, 0, 200);
    rect(0, height - 40, width, 40);

    // Top 3の結果を表示
    if (results.length > 0) {
        // タイトル
        fill(255);
        textSize(14);
        textAlign(LEFT);
        text('検出結果 Top 3:', 10, 20);

        // 各結果を表示
        for (let i = 0; i < min(3, results.length); i++) {
            let label = results[i].label;
            let confidence = results[i].confidence;

            // 信頼度に応じて色を設定
            if (confidence > 0.7) {
                fill(0, 255, 0);  // 緑
            } else if (confidence > 0.4) {
                fill(255, 255, 0);  // 黄
            } else {
                fill(255, 100, 100);  // 薄い赤
            }

            // 順位、ラベル、信頼度を表示
            textSize(16);
            let yPos = 50 + (i * 30);
            text((i + 1) + '. ' + label, 10, yPos);

            // 信頼度バー
            let barWidth = confidence * 200;
            rect(10, yPos + 5, barWidth, 10);

            // パーセンテージ表示
            fill(255);
            textSize(12);
            text(nf(confidence * 100, 0, 1) + '%', barWidth + 20, yPos + 13);
        }

        // 最も可能性の高い結果を下部に大きく表示
        fill(255);
        textSize(20);
        textAlign(CENTER);
        text(results[0].label + ' (' + nf(results[0].confidence * 100, 0, 1) + '%)',
             width / 2, height - 10);
    }
}
```

## まとめ

第1章では、ml5.jsの基本概念と主要な機能について学びました。ml5.jsは機械学習を簡単に扱えるようにする強力なツールであり、画像認識、ポーズ検出、音声認識など、様々な用途に活用できます。

次章では、実際にml5.jsを使用するための環境構築とセットアップについて詳しく学習します。