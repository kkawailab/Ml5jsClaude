# 第4章: ポーズ検出入門

## 4.1 PoseNetの基本

### PoseNetとは

PoseNetは、画像や動画から人の姿勢（ポーズ）を検出するディープラーニングモデルです。人体の主要な関節点（キーポイント）の位置を推定し、骨格構造を認識します。

### キーポイントの種類

PoseNetは17個のキーポイントを検出します：

```javascript
// PoseNetが検出する17のキーポイント
const keypoints = [
    'nose',           // 鼻
    'leftEye',        // 左目
    'rightEye',       // 右目
    'leftEar',        // 左耳
    'rightEar',       // 右耳
    'leftShoulder',   // 左肩
    'rightShoulder',  // 右肩
    'leftElbow',      // 左肘
    'rightElbow',     // 右肘
    'leftWrist',      // 左手首
    'rightWrist',     // 右手首
    'leftHip',        // 左腰
    'rightHip',       // 右腰
    'leftKnee',       // 左膝
    'rightKnee',      // 右膝
    'leftAnkle',      // 左足首
    'rightAnkle'      // 右足首
];
```

### 基本的なPoseNetの使用

```javascript
// PoseNetの基本実装
let video;
let poseNet;
let poses = [];

function setup() {
    createCanvas(640, 480);

    // ビデオキャプチャを開始
    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    // PoseNetモデルを初期化
    poseNet = ml5.poseNet(video, modelReady);

    // ポーズ検出のイベントリスナー
    poseNet.on('pose', function(results) {
        poses = results;
        // 検出されたポーズ数をログ出力
        console.log(`${poses.length}人を検出`);
    });
}

function modelReady() {
    console.log('PoseNetモデル準備完了');
}

function draw() {
    // ビデオを描画
    image(video, 0, 0, width, height);

    // すべての検出されたポーズを描画
    drawKeypoints();
    drawSkeleton();
}

// キーポイントを描画
function drawKeypoints() {
    // すべてのポーズをループ
    for (let i = 0; i < poses.length; i++) {
        let pose = poses[i].pose;

        // 各キーポイントを描画
        for (let j = 0; j < pose.keypoints.length; j++) {
            let keypoint = pose.keypoints[j];

            // 信頼度が0.2以上の場合のみ描画
            if (keypoint.score > 0.2) {
                fill(255, 0, 0);
                noStroke();
                ellipse(keypoint.position.x, keypoint.position.y, 10, 10);

                // キーポイント名を表示
                fill(255);
                textSize(10);
                text(keypoint.part, keypoint.position.x + 5, keypoint.position.y - 5);
            }
        }
    }
}

// スケルトン（骨格）を描画
function drawSkeleton() {
    // すべてのポーズをループ
    for (let i = 0; i < poses.length; i++) {
        let skeleton = poses[i].skeleton;

        // 各骨格ラインを描画
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
```

## 4.2 人体のキーポイント検出

### 詳細なキーポイント情報の取得

```javascript
// キーポイントの詳細情報を取得・表示
let video;
let poseNet;
let currentPose = null;

function setup() {
    createCanvas(800, 600);

    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    // PoseNetの設定オプション
    let options = {
        architecture: 'MobileNetV1',
        imageScaleFactor: 0.3,  // 画像のスケール（精度と速度のトレードオフ）
        outputStride: 16,        // 出力ストライド（8, 16, 32）
        flipHorizontal: false,   // 水平反転
        minConfidence: 0.5,      // 最小信頼度
        maxPoseDetections: 5,    // 最大検出人数
        scoreThreshold: 0.5,     // スコアしきい値
        nmsRadius: 30,           // 非最大抑制の半径
        detectionType: 'multiple', // 'single' または 'multiple'
        multiplier: 0.75         // モデルの深さ（0.50, 0.75, 1.0）
    };

    poseNet = ml5.poseNet(video, options, modelReady);
    poseNet.on('pose', gotPoses);
}

function modelReady() {
    console.log('PoseNet準備完了（詳細設定）');
}

function gotPoses(results) {
    if (results.length > 0) {
        currentPose = results[0].pose;
    }
}

function draw() {
    // ビデオを左側に描画
    image(video, 0, 0, 640, 480);

    // キーポイント情報パネル
    drawInfoPanel();

    if (currentPose) {
        // キーポイントを描画
        drawDetailedKeypoints();

        // 体の中心線を描画
        drawCenterLine();

        // 角度を計算して表示
        calculateAndDisplayAngles();
    }
}

function drawInfoPanel() {
    // 情報パネルの背景
    fill(0, 0, 0, 200);
    rect(640, 0, 160, height);

    // タイトル
    fill(255);
    textAlign(LEFT);
    textSize(14);
    text('キーポイント情報', 650, 20);

    if (currentPose) {
        // 信頼度の高い上位5つのキーポイントを表示
        let sortedKeypoints = currentPose.keypoints.sort((a, b) => b.score - a.score);

        for (let i = 0; i < min(5, sortedKeypoints.length); i++) {
            let kp = sortedKeypoints[i];
            let y = 50 + i * 30;

            fill(255);
            textSize(12);
            text(kp.part, 650, y);

            // 信頼度バー
            fill(0, 255, 0);
            rect(650, y + 5, kp.score * 100, 10);

            // 座標
            fill(200);
            textSize(10);
            text(`(${Math.round(kp.position.x)}, ${Math.round(kp.position.y)})`, 650, y + 25);
        }
    }
}

function drawDetailedKeypoints() {
    for (let keypoint of currentPose.keypoints) {
        if (keypoint.score > 0.2) {
            // 信頼度に応じて色を変更
            let r = map(keypoint.score, 0, 1, 255, 0);
            let g = map(keypoint.score, 0, 1, 0, 255);

            fill(r, g, 0);
            noStroke();

            // キーポイントのサイズも信頼度に応じて変更
            let size = map(keypoint.score, 0, 1, 5, 15);
            ellipse(keypoint.position.x, keypoint.position.y, size, size);
        }
    }
}

function drawCenterLine() {
    // 体の中心線を計算（鼻から腰の中心まで）
    let nose = currentPose.nose;
    let leftHip = currentPose.leftHip;
    let rightHip = currentPose.rightHip;

    if (nose.score > 0.2 && leftHip.score > 0.2 && rightHip.score > 0.2) {
        // 腰の中心点を計算
        let hipCenterX = (leftHip.x + rightHip.x) / 2;
        let hipCenterY = (leftHip.y + rightHip.y) / 2;

        // 中心線を描画
        stroke(0, 255, 255);
        strokeWeight(3);
        line(nose.x, nose.y, hipCenterX, hipCenterY);
    }
}

function calculateAndDisplayAngles() {
    // 肘の角度を計算
    let leftShoulder = currentPose.leftShoulder;
    let leftElbow = currentPose.leftElbow;
    let leftWrist = currentPose.leftWrist;

    if (leftShoulder.score > 0.2 && leftElbow.score > 0.2 && leftWrist.score > 0.2) {
        let angle = calculateAngle(
            leftShoulder, leftElbow, leftWrist
        );

        // 角度を表示
        fill(255, 255, 0);
        textSize(16);
        text(`左肘: ${Math.round(angle)}°`, leftElbow.x + 10, leftElbow.y);
    }

    // 膝の角度を計算
    let leftHip = currentPose.leftHip;
    let leftKnee = currentPose.leftKnee;
    let leftAnkle = currentPose.leftAnkle;

    if (leftHip.score > 0.2 && leftKnee.score > 0.2 && leftAnkle.score > 0.2) {
        let angle = calculateAngle(
            leftHip, leftKnee, leftAnkle
        );

        // 角度を表示
        fill(255, 255, 0);
        textSize(16);
        text(`左膝: ${Math.round(angle)}°`, leftKnee.x + 10, leftKnee.y);
    }
}

// 3点から角度を計算
function calculateAngle(pointA, pointB, pointC) {
    let AB = createVector(pointA.x - pointB.x, pointA.y - pointB.y);
    let BC = createVector(pointC.x - pointB.x, pointC.y - pointB.y);

    let angle = degrees(AB.angleBetween(BC));
    return abs(angle);
}
```

### 体の向きと姿勢の判定

```javascript
// 体の向きと姿勢を判定
let video;
let poseNet;
let pose;
let bodyOrientation = '';
let postureStatus = '';

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', gotPoses);
}

function gotPoses(results) {
    if (results.length > 0) {
        pose = results[0].pose;
        analyzeBodyOrientation();
        analyzePosture();
    }
}

function analyzeBodyOrientation() {
    if (!pose) return;

    let leftShoulder = pose.leftShoulder;
    let rightShoulder = pose.rightShoulder;

    if (leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        // 肩の幅から体の向きを判定
        let shoulderWidth = abs(leftShoulder.x - rightShoulder.x);
        let maxWidth = width * 0.25;  // 正面を向いている時の想定幅

        let ratio = shoulderWidth / maxWidth;

        if (ratio > 0.8) {
            bodyOrientation = '正面';
        } else if (leftShoulder.x < rightShoulder.x && ratio < 0.5) {
            bodyOrientation = '右向き';
        } else if (leftShoulder.x > rightShoulder.x && ratio < 0.5) {
            bodyOrientation = '左向き';
        } else {
            bodyOrientation = '斜め';
        }
    }
}

function analyzePosture() {
    if (!pose) return;

    let nose = pose.nose;
    let leftShoulder = pose.leftShoulder;
    let rightShoulder = pose.rightShoulder;
    let leftHip = pose.leftHip;
    let rightHip = pose.rightHip;

    // 姿勢の良し悪しを判定
    if (nose.score > 0.3 && leftShoulder.score > 0.3 &&
        rightShoulder.score > 0.3 && leftHip.score > 0.3) {

        // 肩の中心
        let shoulderCenterY = (leftShoulder.y + rightShoulder.y) / 2;

        // 腰の中心
        let hipCenterY = (leftHip.y + rightHip.y) / 2;

        // 垂直アライメントチェック
        let noseX = nose.x;
        let shoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2;

        let alignment = abs(noseX - shoulderCenterX);

        if (alignment < 30) {
            postureStatus = '良い姿勢';
        } else {
            postureStatus = '姿勢要改善';
        }

        // 肩の傾き
        let shoulderTilt = abs(leftShoulder.y - rightShoulder.y);
        if (shoulderTilt > 20) {
            postureStatus += ' (肩が傾いています)';
        }
    }
}

function draw() {
    image(video, 0, 0);

    if (pose) {
        // キーポイントとスケルトンを描画
        drawKeypoints();
        drawSkeleton();

        // 分析結果を表示
        displayAnalysis();
    }
}

function displayAnalysis() {
    // 分析結果パネル
    fill(0, 0, 0, 200);
    rect(0, height - 100, width, 100);

    fill(255);
    textAlign(LEFT);
    textSize(16);

    text(`体の向き: ${bodyOrientation}`, 20, height - 70);
    text(`姿勢: ${postureStatus}`, 20, height - 45);

    // 推奨事項
    if (postureStatus.includes('要改善')) {
        fill(255, 255, 0);
        text('💡 背筋を伸ばして、肩の力を抜きましょう', 20, height - 20);
    }
}
```

## 4.3 ポーズを使ったインタラクティブアート

### ポーズで絵を描く

```javascript
// ポーズを使ったペイントアプリ
let video;
let poseNet;
let poses = [];
let paintCanvas;
let brushColor;
let brushSize = 10;
let isDrawing = false;

function setup() {
    createCanvas(640, 480);

    // ペイント用の別キャンバス
    paintCanvas = createGraphics(640, 480);
    paintCanvas.clear();

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', (results) => {
        poses = results;
    });

    // 初期ブラシ色
    brushColor = color(255, 0, 0);
}

function modelReady() {
    console.log('ポーズペイント準備完了');
}

function draw() {
    // ビデオを描画
    image(video, 0, 0);

    // ペイントレイヤーを重ねる
    image(paintCanvas, 0, 0);

    if (poses.length > 0) {
        let pose = poses[0].pose;

        // 右手で描画
        let rightWrist = pose.rightWrist;
        if (rightWrist.score > 0.2) {
            // 手首の位置にカーソル表示
            noFill();
            stroke(255);
            strokeWeight(2);
            ellipse(rightWrist.x, rightWrist.y, brushSize * 2);

            // 右手を上げたら描画開始
            let rightShoulder = pose.rightShoulder;
            if (rightShoulder.score > 0.2) {
                if (rightWrist.y < rightShoulder.y - 50) {
                    if (!isDrawing) {
                        isDrawing = true;
                    }
                    // ペイントキャンバスに描画
                    paintCanvas.fill(brushColor);
                    paintCanvas.noStroke();
                    paintCanvas.ellipse(rightWrist.x, rightWrist.y, brushSize);
                } else {
                    isDrawing = false;
                }
            }
        }

        // 左手で色を選択
        let leftWrist = pose.leftWrist;
        if (leftWrist.score > 0.2) {
            selectColorByPosition(leftWrist.x, leftWrist.y);
        }
    }

    // UI表示
    drawUI();
}

function selectColorByPosition(x, y) {
    // 画面の左側の高さで色を選択
    if (x < 100) {
        let hue = map(y, 0, height, 0, 360);
        brushColor = color(`hsb(${hue}, 100%, 100%)`);
    }
}

function drawUI() {
    // カラーパレット
    for (let i = 0; i < height; i += 5) {
        let hue = map(i, 0, height, 0, 360);
        push();
        colorMode(HSB);
        fill(hue, 100, 100);
        noStroke();
        rect(0, i, 30, 5);
        pop();
    }

    // 現在の色
    fill(brushColor);
    rect(width - 50, 10, 40, 40);

    // インストラクション
    fill(255);
    textAlign(RIGHT);
    textSize(12);
    text('右手を上げて描画', width - 60, 70);
    text('左手を左端で色選択', width - 60, 85);
    text('Cキーでクリア', width - 60, 100);

    // 描画状態
    if (isDrawing) {
        fill(0, 255, 0);
        text('描画中', width - 60, 120);
    }
}

function keyPressed() {
    if (key === 'c' || key === 'C') {
        paintCanvas.clear();
    }
    if (key === 's' || key === 'S') {
        saveCanvas(paintCanvas, 'pose-painting', 'png');
    }
}
```

### ポーズでゲームコントロール

```javascript
// ポーズで操作するゲーム
let video;
let poseNet;
let pose;
let player;
let obstacles = [];
let score = 0;
let gameState = 'playing'; // 'playing', 'gameOver'

class Player {
    constructor() {
        this.x = width / 2;
        this.y = height - 100;
        this.size = 30;
        this.speed = 5;
    }

    update(noseX) {
        // 鼻の位置でプレイヤーを操作
        if (noseX) {
            this.x = lerp(this.x, noseX, 0.2);
        }

        // 画面内に制限
        this.x = constrain(this.x, this.size / 2, width - this.size / 2);
    }

    display() {
        fill(0, 255, 0);
        ellipse(this.x, this.y, this.size);
    }

    hits(obstacle) {
        let d = dist(this.x, this.y, obstacle.x, obstacle.y);
        return d < (this.size + obstacle.size) / 2;
    }
}

class Obstacle {
    constructor() {
        this.x = random(width);
        this.y = 0;
        this.size = random(20, 40);
        this.speed = random(2, 5);
    }

    update() {
        this.y += this.speed;
    }

    display() {
        fill(255, 0, 0);
        ellipse(this.x, this.y, this.size);
    }

    offScreen() {
        return this.y > height + this.size;
    }
}

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', gotPoses);

    player = new Player();
}

function modelReady() {
    console.log('ポーズゲーム準備完了');
}

function gotPoses(results) {
    if (results.length > 0) {
        pose = results[0].pose;
    }
}

function draw() {
    // ビデオを薄く表示
    tint(255, 50);
    image(video, 0, 0);
    noTint();

    if (gameState === 'playing') {
        playGame();
    } else {
        showGameOver();
    }
}

function playGame() {
    // プレイヤーを更新
    if (pose && pose.nose.score > 0.2) {
        player.update(pose.nose.x);
    }
    player.display();

    // 障害物を生成
    if (frameCount % 30 === 0) {
        obstacles.push(new Obstacle());
    }

    // 障害物を更新
    for (let i = obstacles.length - 1; i >= 0; i--) {
        let obs = obstacles[i];
        obs.update();
        obs.display();

        // 衝突判定
        if (player.hits(obs)) {
            gameState = 'gameOver';
        }

        // 画面外の障害物を削除
        if (obs.offScreen()) {
            obstacles.splice(i, 1);
            score++;
        }
    }

    // スコア表示
    fill(255);
    textSize(24);
    textAlign(LEFT);
    text(`スコア: ${score}`, 10, 30);

    // ポーズインジケーター
    if (pose && pose.nose.score > 0.2) {
        // 鼻の位置を表示
        stroke(0, 255, 255);
        strokeWeight(3);
        noFill();
        ellipse(pose.nose.x, pose.nose.y, 30);
    }
}

function showGameOver() {
    fill(0, 0, 0, 200);
    rect(0, 0, width, height);

    fill(255);
    textAlign(CENTER);
    textSize(48);
    text('GAME OVER', width / 2, height / 2 - 50);

    textSize(24);
    text(`最終スコア: ${score}`, width / 2, height / 2);

    textSize(16);
    text('両手を上げてリスタート', width / 2, height / 2 + 50);

    // 両手を上げたらリスタート
    if (pose) {
        let leftWrist = pose.leftWrist;
        let rightWrist = pose.rightWrist;
        let nose = pose.nose;

        if (leftWrist.score > 0.2 && rightWrist.score > 0.2 && nose.score > 0.2) {
            if (leftWrist.y < nose.y && rightWrist.y < nose.y) {
                resetGame();
            }
        }
    }
}

function resetGame() {
    gameState = 'playing';
    score = 0;
    obstacles = [];
    player = new Player();
}
```

## 4.4 複数人のポーズ検出

### 複数人同時検出

```javascript
// 複数人のポーズを同時に検出・追跡
let video;
let poseNet;
let poses = [];
let poseHistory = {};
let colors = {};

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    // 複数人検出の設定
    let options = {
        maxPoseDetections: 5,     // 最大5人まで検出
        minPoseConfidence: 0.15,  // 最小ポーズ信頼度
        minPartConfidence: 0.1    // 最小パーツ信頼度
    };

    poseNet = ml5.poseNet(video, options, modelReady);
    poseNet.on('pose', gotPoses);

    colorMode(HSB);
}

function modelReady() {
    console.log('複数人検出モード準備完了');
}

function gotPoses(results) {
    poses = results;
    updatePoseTracking();
}

function updatePoseTracking() {
    // 各ポーズにIDを割り当て、履歴を追跡
    for (let i = 0; i < poses.length; i++) {
        let pose = poses[i].pose;

        // 簡易的なID割り当て（実際はより高度な追跡が必要）
        let poseId = `person_${i}`;

        // 履歴を保存
        if (!poseHistory[poseId]) {
            poseHistory[poseId] = [];
            // 各人に異なる色を割り当て
            colors[poseId] = color(random(360), 80, 100);
        }

        // 鼻の位置を履歴に追加（軌跡用）
        if (pose.nose.score > 0.2) {
            poseHistory[poseId].push({
                x: pose.nose.x,
                y: pose.nose.y,
                timestamp: millis()
            });

            // 古いデータを削除（最新30フレーム分のみ保持）
            if (poseHistory[poseId].length > 30) {
                poseHistory[poseId].shift();
            }
        }
    }
}

function draw() {
    image(video, 0, 0);

    // 各人のポーズを異なる色で描画
    for (let i = 0; i < poses.length; i++) {
        let poseId = `person_${i}`;
        let pose = poses[i].pose;
        let skeleton = poses[i].skeleton;

        // この人用の色を使用
        let personColor = colors[poseId] || color(0, 80, 100);

        // キーポイントを描画
        fill(personColor);
        noStroke();
        for (let keypoint of pose.keypoints) {
            if (keypoint.score > 0.2) {
                ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
            }
        }

        // スケルトンを描画
        strokeWeight(2);
        stroke(personColor);
        for (let j = 0; j < skeleton.length; j++) {
            let partA = skeleton[j][0];
            let partB = skeleton[j][1];
            line(partA.position.x, partA.position.y,
                 partB.position.x, partB.position.y);
        }

        // 人物ラベル
        fill(personColor);
        textSize(16);
        text(`Person ${i + 1}`, pose.nose.x - 30, pose.nose.y - 20);
    }

    // 移動軌跡を描画
    drawMovementTrails();

    // 統計情報
    displayStats();
}

function drawMovementTrails() {
    // 各人の移動軌跡を描画
    for (let poseId in poseHistory) {
        let trail = poseHistory[poseId];
        let personColor = colors[poseId];

        noFill();
        strokeWeight(3);

        for (let i = 1; i < trail.length; i++) {
            // 古い点ほど薄く
            let alpha = map(i, 0, trail.length, 20, 100);
            stroke(hue(personColor), saturation(personColor), brightness(personColor), alpha);

            line(trail[i - 1].x, trail[i - 1].y,
                 trail[i].x, trail[i].y);
        }
    }
}

function displayStats() {
    // 統計情報パネル
    fill(0, 0, 0, 200);
    rect(0, 0, 200, 100);

    fill(255);
    textAlign(LEFT);
    textSize(14);
    text(`検出人数: ${poses.length}`, 10, 25);

    // 各人の信頼度を表示
    textSize(12);
    for (let i = 0; i < poses.length; i++) {
        let avgScore = calculateAverageScore(poses[i].pose);
        text(`Person ${i + 1}: ${(avgScore * 100).toFixed(1)}%`, 10, 45 + i * 15);
    }
}

function calculateAverageScore(pose) {
    let totalScore = 0;
    let count = 0;

    for (let keypoint of pose.keypoints) {
        totalScore += keypoint.score;
        count++;
    }

    return count > 0 ? totalScore / count : 0;
}
```

### グループインタラクション

```javascript
// 複数人でのインタラクティブゲーム
let video;
let poseNet;
let poses = [];
let bubbles = [];
let teamScores = { left: 0, right: 0 };

class Bubble {
    constructor() {
        this.x = random(width);
        this.y = height + 50;
        this.size = random(30, 60);
        this.speed = random(1, 3);
        this.color = color(random(360), 80, 100);
        this.popped = false;
    }

    update() {
        if (!this.popped) {
            this.y -= this.speed;
        }
    }

    display() {
        if (!this.popped) {
            fill(this.color);
            noStroke();
            ellipse(this.x, this.y, this.size);
        }
    }

    checkPop(x, y) {
        if (!this.popped) {
            let d = dist(x, y, this.x, this.y);
            if (d < this.size / 2 + 20) {
                this.popped = true;
                return true;
            }
        }
        return false;
    }

    offScreen() {
        return this.y < -this.size || this.popped;
    }
}

function setup() {
    createCanvas(640, 480);
    colorMode(HSB);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    let options = {
        maxPoseDetections: 4  // 最大4人
    };

    poseNet = ml5.poseNet(video, options, modelReady);
    poseNet.on('pose', gotPoses);
}

function modelReady() {
    console.log('グループゲーム準備完了');
}

function gotPoses(results) {
    poses = results;
}

function draw() {
    image(video, 0, 0);

    // 半透明オーバーレイ
    fill(0, 0, 0, 100);
    rect(0, 0, width, height);

    // チーム分けライン
    stroke(255);
    strokeWeight(2);
    line(width / 2, 0, width / 2, height);

    // バブルを生成
    if (frameCount % 20 === 0) {
        bubbles.push(new Bubble());
    }

    // バブルを更新・表示
    for (let i = bubbles.length - 1; i >= 0; i--) {
        let bubble = bubbles[i];
        bubble.update();
        bubble.display();

        // 各プレイヤーの手でバブルを割る
        for (let j = 0; j < poses.length; j++) {
            let pose = poses[j].pose;

            // 両手でバブルをチェック
            if (pose.leftWrist.score > 0.2) {
                if (bubble.checkPop(pose.leftWrist.x, pose.leftWrist.y)) {
                    addScore(pose.leftWrist.x);
                    createPopEffect(bubble.x, bubble.y);
                }
            }

            if (pose.rightWrist.score > 0.2) {
                if (bubble.checkPop(pose.rightWrist.x, pose.rightWrist.y)) {
                    addScore(pose.rightWrist.x);
                    createPopEffect(bubble.x, bubble.y);
                }
            }
        }

        // 画面外のバブルを削除
        if (bubble.offScreen()) {
            bubbles.splice(i, 1);
        }
    }

    // プレイヤーを描画
    drawPlayers();

    // スコアボード
    drawScoreboard();
}

function drawPlayers() {
    for (let i = 0; i < poses.length; i++) {
        let pose = poses[i].pose;

        // チームカラー（画面の左右で分ける）
        let teamColor;
        if (pose.nose.x < width / 2) {
            teamColor = color(200, 80, 100);  // 青チーム
        } else {
            teamColor = color(0, 80, 100);    // 赤チーム
        }

        // 手の位置を強調表示
        fill(teamColor);
        noStroke();

        if (pose.leftWrist.score > 0.2) {
            ellipse(pose.leftWrist.x, pose.leftWrist.y, 30);
        }

        if (pose.rightWrist.score > 0.2) {
            ellipse(pose.rightWrist.x, pose.rightWrist.y, 30);
        }

        // プレイヤー番号
        fill(255);
        textAlign(CENTER);
        textSize(16);
        if (pose.nose.score > 0.2) {
            text(`P${i + 1}`, pose.nose.x, pose.nose.y - 20);
        }
    }
}

function addScore(x) {
    if (x < width / 2) {
        teamScores.left++;
    } else {
        teamScores.right++;
    }
}

function createPopEffect(x, y) {
    // バブルが割れたエフェクト（簡易版）
    for (let i = 0; i < 10; i++) {
        fill(random(360), 80, 100, 150);
        let offsetX = random(-20, 20);
        let offsetY = random(-20, 20);
        ellipse(x + offsetX, y + offsetY, random(5, 15));
    }
}

function drawScoreboard() {
    // スコアボード背景
    fill(0, 0, 0, 200);
    rect(width / 2 - 150, 10, 300, 60);

    // チームスコア
    fill(200, 80, 100);
    textAlign(RIGHT);
    textSize(32);
    text(teamScores.left, width / 2 - 20, 50);

    fill(0, 80, 100);
    textAlign(LEFT);
    text(teamScores.right, width / 2 + 20, 50);

    // VS
    fill(255);
    textAlign(CENTER);
    textSize(20);
    text('VS', width / 2, 50);
}

function keyPressed() {
    if (key === 'r' || key === 'R') {
        // リセット
        teamScores = { left: 0, right: 0 };
        bubbles = [];
    }
}
```

## 練習問題

### 問題1: キーポイント理解
PoseNetが検出する17個のキーポイントのうち、上半身に関するものをすべて挙げてください。

### 問題2: 角度計算
3つのキーポイント（肩、肘、手首）から腕の曲げ角度を計算し、その角度に応じて異なる色で表示するプログラムを作成してください。

### 問題3: ポーズ分類
立っている、座っている、手を挙げているの3つの状態を検出するプログラムを作成してください。

### 問題4: フィットネスアプリ
スクワットの回数をカウントするフィットネスアプリを作成してください。正しいフォームかどうかもチェックする機能を含めてください。

## 解答

### 問題1の解答
上半身のキーポイント（11個）：
1. nose（鼻）
2. leftEye（左目）
3. rightEye（右目）
4. leftEar（左耳）
5. rightEar（右耳）
6. leftShoulder（左肩）
7. rightShoulder（右肩）
8. leftElbow（左肘）
9. rightElbow（右肘）
10. leftWrist（左手首）
11. rightWrist（右手首）

### 問題2の解答

```javascript
// 腕の角度を計算して色分け表示
let video;
let poseNet;
let pose;

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', gotPoses);
}

function modelReady() {
    console.log('角度計算システム準備完了');
}

function gotPoses(results) {
    if (results.length > 0) {
        pose = results[0].pose;
    }
}

function draw() {
    image(video, 0, 0);

    if (pose) {
        // 左腕の角度を計算・表示
        calculateArmAngle('left');

        // 右腕の角度を計算・表示
        calculateArmAngle('right');
    }
}

function calculateArmAngle(side) {
    let shoulder = pose[side + 'Shoulder'];
    let elbow = pose[side + 'Elbow'];
    let wrist = pose[side + 'Wrist'];

    if (shoulder.score > 0.2 && elbow.score > 0.2 && wrist.score > 0.2) {
        // ベクトルを作成
        let v1 = createVector(shoulder.x - elbow.x, shoulder.y - elbow.y);
        let v2 = createVector(wrist.x - elbow.x, wrist.y - elbow.y);

        // 角度を計算
        let angle = degrees(v1.angleBetween(v2));
        angle = abs(angle);

        // 角度に応じて色を決定
        let angleColor;
        if (angle < 45) {
            angleColor = color(255, 0, 0);  // 赤：鋭角
        } else if (angle < 90) {
            angleColor = color(255, 255, 0);  // 黄：中間
        } else if (angle < 135) {
            angleColor = color(0, 255, 0);  // 緑：鈍角
        } else {
            angleColor = color(0, 0, 255);  // 青：伸展
        }

        // 関節を色付きで描画
        fill(angleColor);
        noStroke();
        ellipse(shoulder.x, shoulder.y, 15);
        ellipse(elbow.x, elbow.y, 20);
        ellipse(wrist.x, wrist.y, 15);

        // ラインを描画
        stroke(angleColor);
        strokeWeight(5);
        line(shoulder.x, shoulder.y, elbow.x, elbow.y);
        line(elbow.x, elbow.y, wrist.x, wrist.y);

        // 角度を表示
        fill(255);
        noStroke();
        textSize(16);
        textAlign(CENTER);
        text(`${Math.round(angle)}°`, elbow.x, elbow.y - 25);

        // 角度の弧を描画
        noFill();
        stroke(angleColor);
        strokeWeight(2);
        push();
        translate(elbow.x, elbow.y);
        let startAngle = atan2(shoulder.y - elbow.y, shoulder.x - elbow.x);
        let endAngle = atan2(wrist.y - elbow.y, wrist.x - elbow.x);
        arc(0, 0, 50, 50, min(startAngle, endAngle), max(startAngle, endAngle));
        pop();
    }
}
```

### 問題3の解答

```javascript
// ポーズ分類システム
let video;
let poseNet;
let pose;
let poseState = '検出中...';

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', gotPoses);
}

function modelReady() {
    console.log('ポーズ分類システム準備完了');
}

function gotPoses(results) {
    if (results.length > 0) {
        pose = results[0].pose;
        classifyPose();
    }
}

function classifyPose() {
    if (!pose) return;

    let nose = pose.nose;
    let leftShoulder = pose.leftShoulder;
    let rightShoulder = pose.rightShoulder;
    let leftHip = pose.leftHip;
    let rightHip = pose.rightHip;
    let leftWrist = pose.leftWrist;
    let rightWrist = pose.rightWrist;
    let leftKnee = pose.leftKnee;
    let rightKnee = pose.rightKnee;

    // 信頼度チェック
    if (nose.score < 0.3 || leftShoulder.score < 0.3) {
        poseState = '検出精度不足';
        return;
    }

    // 手を挙げているかチェック
    let leftHandRaised = false;
    let rightHandRaised = false;

    if (leftWrist.score > 0.3 && leftShoulder.score > 0.3) {
        leftHandRaised = leftWrist.y < leftShoulder.y - 50;
    }

    if (rightWrist.score > 0.3 && rightShoulder.score > 0.3) {
        rightHandRaised = rightWrist.y < rightShoulder.y - 50;
    }

    if (leftHandRaised || rightHandRaised) {
        poseState = '手を挙げている';
        return;
    }

    // 座っているか立っているかを判定
    if (leftHip.score > 0.3 && rightHip.score > 0.3 &&
        leftKnee.score > 0.3 && rightKnee.score > 0.3) {

        // 腰と膝の高さの差
        let hipY = (leftHip.y + rightHip.y) / 2;
        let kneeY = (leftKnee.y + rightKnee.y) / 2;
        let heightDiff = kneeY - hipY;

        // 肩と腰の距離
        let shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        let torsoLength = hipY - shoulderY;

        // 座っている判定（膝と腰の高さが近い）
        if (heightDiff < torsoLength * 0.3) {
            poseState = '座っている';
        } else {
            poseState = '立っている';
        }
    } else {
        // デフォルトは立っていると判定
        poseState = '立っている';
    }
}

function draw() {
    image(video, 0, 0);

    if (pose) {
        // キーポイントを描画
        drawKeypoints();

        // 状態に応じた視覚的フィードバック
        drawStateFeedback();
    }

    // 状態表示
    displayState();
}

function drawStateFeedback() {
    // 状態に応じて異なる色とエフェクト
    push();
    blendMode(ADD);

    if (poseState === '手を挙げている') {
        // 黄色のハイライト
        fill(255, 255, 0, 30);
        rect(0, 0, width, height);
    } else if (poseState === '座っている') {
        // 青のハイライト
        fill(0, 100, 255, 30);
        rect(0, 0, width, height);
    } else if (poseState === '立っている') {
        // 緑のハイライト
        fill(0, 255, 0, 20);
        rect(0, 0, width, height);
    }

    pop();
}

function displayState() {
    // 状態表示パネル
    fill(0, 0, 0, 200);
    rect(0, height - 80, width, 80);

    // 状態テキスト
    fill(255);
    textAlign(CENTER);
    textSize(32);
    text(poseState, width / 2, height - 35);

    // アイコン
    textSize(48);
    if (poseState === '手を挙げている') {
        text('🙌', width / 2 - 100, height - 25);
    } else if (poseState === '座っている') {
        text('🪑', width / 2 - 100, height - 25);
    } else if (poseState === '立っている') {
        text('🚶', width / 2 - 100, height - 25);
    }
}

function drawKeypoints() {
    // 状態に応じた色でキーポイントを描画
    let keypointColor;
    if (poseState === '手を挙げている') {
        keypointColor = color(255, 255, 0);
    } else if (poseState === '座っている') {
        keypointColor = color(0, 100, 255);
    } else {
        keypointColor = color(0, 255, 0);
    }

    fill(keypointColor);
    noStroke();

    for (let keypoint of pose.keypoints) {
        if (keypoint.score > 0.2) {
            ellipse(keypoint.position.x, keypoint.position.y, 10);
        }
    }
}
```

### 問題4の解答

```javascript
// スクワットカウンターアプリ
let video;
let poseNet;
let pose;
let squatCount = 0;
let squatState = 'ready';  // 'ready', 'down', 'up'
let previousKneeAngle = 180;
let formFeedback = '';
let calibrationPhase = true;
let standingKneeAngle = 180;
let isRecording = false;
let workout = [];

function setup() {
    createCanvas(640, 580);

    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    poseNet = ml5.poseNet(video, modelReady);
    poseNet.on('pose', gotPoses);

    // UIボタン
    createButtons();
}

function createButtons() {
    let startBtn = createButton('運動開始');
    startBtn.position(10, height - 50);
    startBtn.mousePressed(() => {
        isRecording = true;
        squatCount = 0;
        workout = [];
    });

    let stopBtn = createButton('運動終了');
    stopBtn.position(100, height - 50);
    stopBtn.mousePressed(() => {
        isRecording = false;
        showWorkoutSummary();
    });

    let calibrateBtn = createButton('キャリブレーション');
    calibrateBtn.position(200, height - 50);
    calibrateBtn.mousePressed(() => {
        calibrationPhase = true;
    });
}

function modelReady() {
    console.log('スクワットカウンター準備完了');
}

function gotPoses(results) {
    if (results.length > 0) {
        pose = results[0].pose;
        if (isRecording) {
            analyzeSquat();
        }
        if (calibrationPhase) {
            calibrate();
        }
    }
}

function calibrate() {
    // 立っている状態での膝の角度を記録
    let leftHip = pose.leftHip;
    let leftKnee = pose.leftKnee;
    let leftAnkle = pose.leftAnkle;

    if (leftHip.score > 0.3 && leftKnee.score > 0.3 && leftAnkle.score > 0.3) {
        standingKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
        calibrationPhase = false;
        console.log('キャリブレーション完了: 立位膝角度 = ' + standingKneeAngle);
    }
}

function analyzeSquat() {
    if (!pose) return;

    // 必要なキーポイント
    let leftHip = pose.leftHip;
    let leftKnee = pose.leftKnee;
    let leftAnkle = pose.leftAnkle;
    let rightHip = pose.rightHip;
    let rightKnee = pose.rightKnee;
    let rightAnkle = pose.rightAnkle;
    let leftShoulder = pose.leftShoulder;
    let rightShoulder = pose.rightShoulder;

    // 信頼度チェック
    if (leftHip.score < 0.3 || leftKnee.score < 0.3 || leftAnkle.score < 0.3) {
        formFeedback = '体が見えません';
        return;
    }

    // 膝の角度を計算
    let leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
    let rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
    let avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;

    // フォームチェック
    checkForm(pose);

    // スクワットの状態を判定
    if (squatState === 'ready' || squatState === 'up') {
        if (avgKneeAngle < 120) {  // 膝が曲がった
            squatState = 'down';
        }
    } else if (squatState === 'down') {
        if (avgKneeAngle > 160) {  // 膝が伸びた
            squatState = 'up';
            squatCount++;

            // ワークアウトデータを記録
            workout.push({
                timestamp: millis(),
                kneeAngle: avgKneeAngle,
                form: formFeedback
            });
        }
    }

    previousKneeAngle = avgKneeAngle;
}

function checkForm(pose) {
    let warnings = [];

    // 膝の位置チェック（つま先より前に出ていないか）
    let leftKnee = pose.leftKnee;
    let leftAnkle = pose.leftAnkle;

    if (leftKnee.score > 0.3 && leftAnkle.score > 0.3) {
        if (leftKnee.x < leftAnkle.x - 30) {
            warnings.push('膝が前に出すぎです');
        }
    }

    // 背中の姿勢チェック
    let nose = pose.nose;
    let leftHip = pose.leftHip;

    if (nose.score > 0.3 && leftHip.score > 0.3) {
        let spineAngle = abs(degrees(atan2(nose.y - leftHip.y, nose.x - leftHip.x)));

        if (spineAngle < 60) {
            warnings.push('背中をまっすぐに');
        }
    }

    // 左右のバランスチェック
    let leftShoulder = pose.leftShoulder;
    let rightShoulder = pose.rightShoulder;

    if (leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        let shoulderTilt = abs(leftShoulder.y - rightShoulder.y);

        if (shoulderTilt > 20) {
            warnings.push('左右のバランスを保って');
        }
    }

    if (warnings.length > 0) {
        formFeedback = warnings.join(', ');
    } else {
        formFeedback = '良いフォームです！';
    }
}

function calculateAngle(pointA, pointB, pointC) {
    let AB = createVector(pointA.x - pointB.x, pointA.y - pointB.y);
    let BC = createVector(pointC.x - pointB.x, pointC.y - pointB.y);
    let angle = degrees(AB.angleBetween(BC));
    return abs(angle);
}

function draw() {
    image(video, 0, 0, width, 480);

    if (pose) {
        drawSkeleton();
        drawAngleIndicators();
    }

    drawUI();
}

function drawSkeleton() {
    // キーポイントを描画
    for (let keypoint of pose.keypoints) {
        if (keypoint.score > 0.2) {
            fill(0, 255, 0);
            noStroke();
            ellipse(keypoint.position.x, keypoint.position.y, 10);
        }
    }
}

function drawAngleIndicators() {
    // 膝の角度を表示
    let leftHip = pose.leftHip;
    let leftKnee = pose.leftKnee;
    let leftAnkle = pose.leftAnkle;

    if (leftHip.score > 0.3 && leftKnee.score > 0.3 && leftAnkle.score > 0.3) {
        let angle = calculateAngle(leftHip, leftKnee, leftAnkle);

        // 角度の色（深いスクワットほど緑）
        let angleColor;
        if (angle < 90) {
            angleColor = color(0, 255, 0);
        } else if (angle < 120) {
            angleColor = color(255, 255, 0);
        } else {
            angleColor = color(255, 0, 0);
        }

        // 角度表示
        fill(angleColor);
        textSize(20);
        text(`${Math.round(angle)}°`, leftKnee.x + 20, leftKnee.y);

        // 角度の弧を描画
        noFill();
        stroke(angleColor);
        strokeWeight(3);
        push();
        translate(leftKnee.x, leftKnee.y);
        let startAngle = atan2(leftHip.y - leftKnee.y, leftHip.x - leftKnee.x);
        let endAngle = atan2(leftAnkle.y - leftKnee.y, leftAnkle.x - leftKnee.x);
        arc(0, 0, 50, 50, min(startAngle, endAngle), max(startAngle, endAngle));
        pop();
    }
}

function drawUI() {
    // UIパネル
    fill(0, 0, 0, 200);
    rect(0, 480, width, 100);

    // カウント表示
    fill(255);
    textAlign(LEFT);
    textSize(36);
    text(`スクワット: ${squatCount}回`, 20, 520);

    // 状態表示
    textSize(20);
    let stateColor;
    if (squatState === 'down') {
        stateColor = color(0, 255, 0);
    } else {
        stateColor = color(255, 255, 0);
    }
    fill(stateColor);
    text(`状態: ${squatState}`, 20, 550);

    // フォーム評価
    fill(255);
    textSize(16);
    text(`フォーム: ${formFeedback}`, 300, 520);

    // 記録中インジケーター
    if (isRecording) {
        fill(255, 0, 0);
        ellipse(width - 30, 510, 20);
        fill(255);
        textAlign(RIGHT);
        text('記録中', width - 45, 515);
    }
}

function showWorkoutSummary() {
    console.log('ワークアウト完了！');
    console.log(`合計スクワット: ${squatCount}回`);
    console.log(`運動時間: ${Math.round(millis() / 1000)}秒`);

    // データをCSVで保存することも可能
    if (workout.length > 0) {
        let csv = 'Time,KneeAngle,Form\n';
        workout.forEach(w => {
            csv += `${w.timestamp},${w.kneeAngle},${w.form}\n`;
        });
        // saveStrings(csv.split('\n'), 'workout.csv');
    }
}
```

## まとめ

第4章では、PoseNetを使用した人体ポーズ検出の基礎から応用まで学習しました。17個のキーポイント検出、角度計算、ポーズ分類、そして複数人同時検出まで、幅広い技術を習得しました。

次章では、より詳細な手の認識とジェスチャー検出について学習します。