# 第5章: 手の認識とジェスチャー

## 5.1 HandPoseの使い方

### HandPoseとは

HandPoseは、手の形状を詳細に検出するディープラーニングモデルです。手の21個の関節点（ランドマーク）を3D座標で検出し、指の動きを正確に追跡できます。

### 21個のランドマーク

```javascript
// HandPoseが検出する21個のランドマーク
const handLandmarks = {
    0: 'wrist',           // 手首
    1: 'thumb_cmc',       // 親指の付け根
    2: 'thumb_mcp',       // 親指の第2関節
    3: 'thumb_ip',        // 親指の第1関節
    4: 'thumb_tip',       // 親指の先端
    5: 'index_mcp',       // 人差し指の付け根
    6: 'index_pip',       // 人差し指の第2関節
    7: 'index_dip',       // 人差し指の第1関節
    8: 'index_tip',       // 人差し指の先端
    9: 'middle_mcp',      // 中指の付け根
    10: 'middle_pip',     // 中指の第2関節
    11: 'middle_dip',     // 中指の第1関節
    12: 'middle_tip',     // 中指の先端
    13: 'ring_mcp',       // 薬指の付け根
    14: 'ring_pip',       // 薬指の第2関節
    15: 'ring_dip',       // 薬指の第1関節
    16: 'ring_tip',       // 薬指の先端
    17: 'pinky_mcp',      // 小指の付け根
    18: 'pinky_pip',      // 小指の第2関節
    19: 'pinky_dip',      // 小指の第1関節
    20: 'pinky_tip'       // 小指の先端
};
```

### 基本的なHandPoseの実装

```javascript
// HandPoseの基本実装
let handpose;
let video;
let predictions = [];

function setup() {
    createCanvas(640, 480);

    // ビデオキャプチャを開始
    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    // HandPoseモデルを初期化
    handpose = ml5.handpose(video, modelReady);

    // 手が検出されたときのイベントリスナー
    handpose.on('predict', results => {
        predictions = results;
    });
}

function modelReady() {
    console.log('HandPoseモデル準備完了');
}

function draw() {
    // ビデオを描画
    image(video, 0, 0, width, height);

    // 検出された手を描画
    drawKeypoints();
    drawSkeleton();
}

// キーポイントを描画
function drawKeypoints() {
    for (let i = 0; i < predictions.length; i++) {
        const prediction = predictions[i];

        // 21個のランドマークを描画
        for (let j = 0; j < prediction.landmarks.length; j++) {
            const keypoint = prediction.landmarks[j];

            // 各ランドマークを円で描画
            fill(0, 255, 0);
            noStroke();
            ellipse(keypoint[0], keypoint[1], 10, 10);

            // ランドマーク番号を表示
            fill(255);
            textSize(8);
            text(j, keypoint[0] + 5, keypoint[1] - 5);
        }
    }
}

// 手の骨格を描画
function drawSkeleton() {
    for (let i = 0; i < predictions.length; i++) {
        const landmarks = predictions[i].landmarks;

        // 指の接続を定義
        const connections = [
            // 親指
            [0, 1], [1, 2], [2, 3], [3, 4],
            // 人差し指
            [0, 5], [5, 6], [6, 7], [7, 8],
            // 中指
            [0, 9], [9, 10], [10, 11], [11, 12],
            // 薬指
            [0, 13], [13, 14], [14, 15], [15, 16],
            // 小指
            [0, 17], [17, 18], [18, 19], [19, 20],
            // 手のひら
            [5, 9], [9, 13], [13, 17]
        ];

        stroke(255, 0, 0);
        strokeWeight(3);

        // 接続線を描画
        for (let connection of connections) {
            const [a, b] = connection;
            const pointA = landmarks[a];
            const pointB = landmarks[b];
            line(pointA[0], pointA[1], pointB[0], pointB[1]);
        }
    }
}
```

## 5.2 手の関節点の取得と活用

### 詳細な関節情報の取得

```javascript
// 手の詳細情報を取得・分析
let handpose;
let video;
let predictions = [];
let handInfo = {
    isOpen: false,
    fingerStates: [],
    gesture: '',
    palmSize: 0,
    orientation: 0
};

function setup() {
    createCanvas(800, 600);

    video = createCapture(VIDEO);
    video.size(640, 480);
    video.hide();

    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', results => {
        predictions = results;
        if (predictions.length > 0) {
            analyzeHand(predictions[0]);
        }
    });
}

function modelReady() {
    console.log('手の分析システム準備完了');
}

function analyzeHand(prediction) {
    const landmarks = prediction.landmarks;

    // 手のひらのサイズを計算
    const wrist = landmarks[0];
    const middleMCP = landmarks[9];
    handInfo.palmSize = dist(wrist[0], wrist[1], middleMCP[0], middleMCP[1]);

    // 手の向きを計算
    const indexMCP = landmarks[5];
    const pinkyMCP = landmarks[17];
    handInfo.orientation = degrees(atan2(
        pinkyMCP[1] - indexMCP[1],
        pinkyMCP[0] - indexMCP[0]
    ));

    // 各指の状態を分析
    handInfo.fingerStates = analyzeFingers(landmarks);

    // 手が開いているか判定
    const openFingers = handInfo.fingerStates.filter(f => f.isExtended).length;
    handInfo.isOpen = openFingers >= 3;

    // ジェスチャーを認識
    handInfo.gesture = recognizeGesture(handInfo.fingerStates);
}

function analyzeFingers(landmarks) {
    const fingerStates = [];
    const fingerNames = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    const fingerIndices = [
        [1, 2, 3, 4],      // 親指
        [5, 6, 7, 8],      // 人差し指
        [9, 10, 11, 12],   // 中指
        [13, 14, 15, 16],  // 薬指
        [17, 18, 19, 20]   // 小指
    ];

    for (let i = 0; i < 5; i++) {
        const indices = fingerIndices[i];
        const mcp = landmarks[indices[0]];
        const pip = landmarks[indices[1]];
        const dip = landmarks[indices[2]];
        const tip = landmarks[indices[3]];

        // 指が伸びているか判定
        let isExtended;
        if (i === 0) {  // 親指は特別な判定
            isExtended = tip[0] < mcp[0] - 30 || tip[0] > mcp[0] + 30;
        } else {
            // 他の指は垂直方向で判定
            isExtended = tip[1] < pip[1];
        }

        // 指の曲がり角度を計算
        const angle = calculateFingerAngle(mcp, pip, dip, tip);

        fingerStates.push({
            name: fingerNames[i],
            isExtended: isExtended,
            angle: angle,
            tipPosition: tip
        });
    }

    return fingerStates;
}

function calculateFingerAngle(mcp, pip, dip, tip) {
    // PIP関節の角度を計算
    const v1 = createVector(mcp[0] - pip[0], mcp[1] - pip[1]);
    const v2 = createVector(dip[0] - pip[0], dip[1] - pip[1]);
    const angle = degrees(v1.angleBetween(v2));
    return abs(angle);
}

function recognizeGesture(fingerStates) {
    const extended = fingerStates.map(f => f.isExtended);

    // 基本的なジェスチャー認識
    if (extended.every(e => !e)) {
        return 'グー';
    } else if (extended.every(e => e)) {
        return 'パー';
    } else if (extended[1] && extended[2] && !extended[0] && !extended[3] && !extended[4]) {
        return 'チョキ';
    } else if (extended[1] && !extended[2] && !extended[3] && !extended[4]) {
        return '指差し';
    } else if (extended[0] && !extended[1] && !extended[2] && !extended[3] && !extended[4]) {
        return 'いいね';
    } else if (extended[1] && extended[4] && !extended[2] && !extended[3]) {
        return 'ロック';
    } else if (!extended[0] && !extended[1] && !extended[2] && !extended[3] && extended[4]) {
        return '小指';
    } else {
        return 'その他';
    }
}

function draw() {
    // ビデオを左側に表示
    image(video, 0, 0, 640, 480);

    // 情報パネルを右側に表示
    drawInfoPanel();

    if (predictions.length > 0) {
        // 手を描画
        drawDetailedHand();

        // 3D表示（簡易版）
        draw3DHand();
    }
}

function drawInfoPanel() {
    // 背景
    fill(20);
    rect(640, 0, 160, height);

    // 情報表示
    fill(255);
    textAlign(LEFT);
    textSize(14);

    let y = 30;
    text('手の分析', 650, y);

    y += 30;
    textSize(12);
    text(`ジェスチャー: ${handInfo.gesture}`, 650, y);

    y += 20;
    text(`手の状態: ${handInfo.isOpen ? '開いている' : '閉じている'}`, 650, y);

    y += 20;
    text(`向き: ${handInfo.orientation.toFixed(0)}°`, 650, y);

    y += 30;
    text('指の状態:', 650, y);

    for (let finger of handInfo.fingerStates) {
        y += 20;
        let status = finger.isExtended ? '伸' : '曲';
        let color = finger.isExtended ? color(0, 255, 0) : color(255, 0, 0);
        fill(color);
        text(`${finger.name}: ${status} (${finger.angle.toFixed(0)}°)`, 660, y);
    }
}

function drawDetailedHand() {
    if (predictions.length === 0) return;

    const landmarks = predictions[0].landmarks;

    // 各指を異なる色で描画
    const fingerColors = [
        color(255, 0, 0),    // 親指：赤
        color(255, 255, 0),  // 人差し指：黄
        color(0, 255, 0),    // 中指：緑
        color(0, 255, 255),  // 薬指：シアン
        color(255, 0, 255)   // 小指：マゼンタ
    ];

    // 指ごとに描画
    const fingerIndices = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ];

    for (let f = 0; f < 5; f++) {
        const indices = fingerIndices[f];
        stroke(fingerColors[f]);
        strokeWeight(4);

        for (let i = 1; i < indices.length; i++) {
            const a = landmarks[indices[i - 1]];
            const b = landmarks[indices[i]];
            line(a[0], a[1], b[0], b[1]);
        }

        // 関節点
        fill(fingerColors[f]);
        noStroke();
        for (let i of indices) {
            ellipse(landmarks[i][0], landmarks[i][1], 8);
        }
    }
}

function draw3DHand() {
    // 簡易3D表示（Z座標を使用）
    push();
    translate(720, 500);

    if (predictions.length > 0) {
        const landmarks = predictions[0].landmarks;

        // Z座標を使って奥行きを表現
        for (let i = 0; i < landmarks.length; i++) {
            const [x, y, z] = landmarks[i];

            // Z座標でサイズと明度を調整
            const size = map(z, -100, 100, 15, 5);
            const brightness = map(z, -100, 100, 255, 100);

            fill(brightness);
            noStroke();

            // 簡易的な投影変換
            const projX = (x - 320) * 0.2;
            const projY = (y - 240) * 0.2;

            ellipse(projX, projY, size);
        }
    }

    pop();
}
```

## 5.3 ジェスチャー認識の実装

### カスタムジェスチャー認識システム

```javascript
// 高度なジェスチャー認識システム
let handpose;
let video;
let predictions = [];
let gestureHistory = [];
let currentGesture = '';
let gestureConfidence = 0;

// カスタムジェスチャーの定義
const customGestures = {
    'peace': {
        fingers: [false, true, true, false, false],
        name: 'ピース'
    },
    'ok': {
        fingers: [true, false, false, false, false],
        special: 'thumb_index_touch',
        name: 'OK'
    },
    'rock': {
        fingers: [false, true, false, false, true],
        name: 'ロック'
    },
    'thumbsUp': {
        fingers: [true, false, false, false, false],
        name: 'いいね'
    },
    'pointUp': {
        fingers: [false, true, false, false, false],
        name: '上を指す'
    },
    'fist': {
        fingers: [false, false, false, false, false],
        name: 'グー'
    },
    'open': {
        fingers: [true, true, true, true, true],
        name: 'パー'
    }
};

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', handlePredictions);
}

function modelReady() {
    console.log('ジェスチャー認識システム準備完了');
}

function handlePredictions(results) {
    predictions = results;
    if (predictions.length > 0) {
        const gesture = detectGesture(predictions[0]);
        updateGestureHistory(gesture);
    }
}

function detectGesture(prediction) {
    const landmarks = prediction.landmarks;
    const fingerStates = getFingerStates(landmarks);

    // カスタムジェスチャーをチェック
    for (let key in customGestures) {
        const gesture = customGestures[key];

        if (matchesGesture(fingerStates, gesture)) {
            // 特別な条件をチェック
            if (gesture.special === 'thumb_index_touch') {
                if (checkThumbIndexTouch(landmarks)) {
                    return key;
                }
            } else {
                return key;
            }
        }
    }

    return 'unknown';
}

function getFingerStates(landmarks) {
    const states = [];

    // 親指（特別な判定）
    const thumbTip = landmarks[4];
    const thumbBase = landmarks[2];
    states.push(thumbTip[0] < thumbBase[0] - 30 || thumbTip[0] > thumbBase[0] + 30);

    // 他の4本の指
    const fingerTips = [8, 12, 16, 20];
    const fingerPIPs = [6, 10, 14, 18];

    for (let i = 0; i < 4; i++) {
        const tip = landmarks[fingerTips[i]];
        const pip = landmarks[fingerPIPs[i]];
        states.push(tip[1] < pip[1] - 20);
    }

    return states;
}

function matchesGesture(fingerStates, gesture) {
    if (!gesture.fingers) return false;

    for (let i = 0; i < 5; i++) {
        if (fingerStates[i] !== gesture.fingers[i]) {
            return false;
        }
    }

    return true;
}

function checkThumbIndexTouch(landmarks) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const distance = dist(thumbTip[0], thumbTip[1], indexTip[0], indexTip[1]);
    return distance < 30;
}

function updateGestureHistory(gesture) {
    gestureHistory.push(gesture);

    // 履歴を最新10フレームに制限
    if (gestureHistory.length > 10) {
        gestureHistory.shift();
    }

    // 最も頻繁なジェスチャーを現在のジェスチャーとする
    const counts = {};
    for (let g of gestureHistory) {
        counts[g] = (counts[g] || 0) + 1;
    }

    let maxCount = 0;
    let mostFrequent = 'unknown';

    for (let g in counts) {
        if (counts[g] > maxCount) {
            maxCount = counts[g];
            mostFrequent = g;
        }
    }

    currentGesture = mostFrequent;
    gestureConfidence = maxCount / gestureHistory.length;
}

function draw() {
    image(video, 0, 0);

    // 手を描画
    drawHand();

    // ジェスチャー情報を表示
    drawGestureInfo();

    // ジェスチャーに応じたエフェクト
    drawGestureEffect();
}

function drawHand() {
    for (let prediction of predictions) {
        const landmarks = prediction.landmarks;

        // 骨格を描画
        stroke(0, 255, 0);
        strokeWeight(3);

        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20]
        ];

        for (let conn of connections) {
            const [a, b] = conn;
            line(landmarks[a][0], landmarks[a][1],
                 landmarks[b][0], landmarks[b][1]);
        }

        // 関節点を描画
        fill(255, 0, 0);
        noStroke();
        for (let landmark of landmarks) {
            ellipse(landmark[0], landmark[1], 8);
        }
    }
}

function drawGestureInfo() {
    // 背景パネル
    fill(0, 0, 0, 200);
    rect(0, height - 100, width, 100);

    // ジェスチャー名
    fill(255);
    textAlign(CENTER);
    textSize(32);

    if (currentGesture !== 'unknown' && customGestures[currentGesture]) {
        text(customGestures[currentGesture].name, width / 2, height - 50);

        // 信頼度バー
        fill(0, 255, 0, 100);
        rect(width / 2 - 100, height - 30, gestureConfidence * 200, 20);

        fill(255);
        textSize(14);
        text(`信頼度: ${(gestureConfidence * 100).toFixed(0)}%`, width / 2, height - 10);
    } else {
        text('ジェスチャーを検出中...', width / 2, height - 50);
    }
}

function drawGestureEffect() {
    if (currentGesture === 'thumbsUp' && gestureConfidence > 0.8) {
        // いいねエフェクト
        push();
        translate(width / 2, height / 2);
        rotate(frameCount * 0.05);
        noFill();
        stroke(255, 255, 0);
        strokeWeight(3);
        star(0, 0, 50, 100, 5);
        pop();
    } else if (currentGesture === 'peace' && gestureConfidence > 0.8) {
        // ピースエフェクト
        fill(255, 255, 0, 50);
        noStroke();
        for (let i = 0; i < 5; i++) {
            let x = random(width);
            let y = random(height);
            let size = random(10, 30);
            ellipse(x, y, size);
        }
    }
}

function star(x, y, radius1, radius2, npoints) {
    let angle = TWO_PI / npoints;
    let halfAngle = angle / 2.0;
    beginShape();
    for (let a = 0; a < TWO_PI; a += angle) {
        let sx = x + cos(a) * radius2;
        let sy = y + sin(a) * radius2;
        vertex(sx, sy);
        sx = x + cos(a + halfAngle) * radius1;
        sy = y + sin(a + halfAngle) * radius1;
        vertex(sx, sy);
    }
    endShape(CLOSE);
}
```

### 動的ジェスチャー認識

```javascript
// 動きを含むジェスチャー認識
let handpose;
let video;
let predictions = [];
let motionTracker = {
    positions: [],
    velocities: [],
    gesture: '',
    phase: 'idle'
};

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', results => {
        predictions = results;
        if (predictions.length > 0) {
            trackMotion(predictions[0]);
        }
    });
}

function trackMotion(prediction) {
    const wrist = prediction.landmarks[0];
    const currentPos = createVector(wrist[0], wrist[1]);

    // 位置を記録
    motionTracker.positions.push(currentPos);

    // 速度を計算
    if (motionTracker.positions.length > 1) {
        const prevPos = motionTracker.positions[motionTracker.positions.length - 2];
        const velocity = p5.Vector.sub(currentPos, prevPos);
        motionTracker.velocities.push(velocity);
    }

    // 履歴を制限
    if (motionTracker.positions.length > 30) {
        motionTracker.positions.shift();
    }
    if (motionTracker.velocities.length > 30) {
        motionTracker.velocities.shift();
    }

    // モーションジェスチャーを検出
    detectMotionGesture();
}

function detectMotionGesture() {
    if (motionTracker.velocities.length < 10) return;

    // 平均速度を計算
    let avgVelocity = createVector(0, 0);
    for (let v of motionTracker.velocities) {
        avgVelocity.add(v);
    }
    avgVelocity.div(motionTracker.velocities.length);

    const speed = avgVelocity.mag();

    // スワイプ検出
    if (speed > 10) {
        const angle = degrees(avgVelocity.heading());

        if (angle > -45 && angle < 45) {
            motionTracker.gesture = 'スワイプ右';
        } else if (angle > 45 && angle < 135) {
            motionTracker.gesture = 'スワイプ下';
        } else if (angle > 135 || angle < -135) {
            motionTracker.gesture = 'スワイプ左';
        } else {
            motionTracker.gesture = 'スワイプ上';
        }
    } else if (speed < 2) {
        // 円を描く動きを検出
        if (detectCircularMotion()) {
            motionTracker.gesture = '円を描く';
        } else {
            motionTracker.gesture = '静止';
        }
    } else {
        motionTracker.gesture = '移動中';
    }
}

function detectCircularMotion() {
    if (motionTracker.positions.length < 20) return false;

    // 最近の位置から中心を計算
    let center = createVector(0, 0);
    for (let pos of motionTracker.positions) {
        center.add(pos);
    }
    center.div(motionTracker.positions.length);

    // 各点の中心からの距離を計算
    let distances = [];
    for (let pos of motionTracker.positions) {
        distances.push(p5.Vector.dist(pos, center));
    }

    // 距離の標準偏差が小さければ円形
    let avgDist = distances.reduce((a, b) => a + b) / distances.length;
    let variance = distances.reduce((sum, d) => sum + pow(d - avgDist, 2), 0) / distances.length;
    let stdDev = sqrt(variance);

    return stdDev < 20 && avgDist > 30;
}

function draw() {
    image(video, 0, 0);

    // 手を描画
    if (predictions.length > 0) {
        drawHand(predictions[0]);
    }

    // モーション軌跡を描画
    drawMotionTrail();

    // ジェスチャー情報を表示
    displayMotionInfo();
}

function drawMotionTrail() {
    noFill();
    strokeWeight(3);

    for (let i = 1; i < motionTracker.positions.length; i++) {
        // 古い位置ほど薄く
        let alpha = map(i, 0, motionTracker.positions.length, 50, 255);
        stroke(0, 255, 255, alpha);

        line(motionTracker.positions[i - 1].x, motionTracker.positions[i - 1].y,
             motionTracker.positions[i].x, motionTracker.positions[i].y);
    }
}

function displayMotionInfo() {
    fill(0, 0, 0, 200);
    rect(0, 0, 200, 80);

    fill(255);
    textAlign(LEFT);
    textSize(16);
    text('モーション: ' + motionTracker.gesture, 10, 30);

    if (motionTracker.velocities.length > 0) {
        const currentSpeed = motionTracker.velocities[motionTracker.velocities.length - 1].mag();
        text('速度: ' + currentSpeed.toFixed(1), 10, 50);
    }
}

function drawHand(prediction) {
    const landmarks = prediction.landmarks;

    // シンプルな手の描画
    stroke(0, 255, 0);
    strokeWeight(2);

    for (let i = 0; i < landmarks.length; i++) {
        fill(0, 255, 0);
        noStroke();
        ellipse(landmarks[i][0], landmarks[i][1], 6);
    }
}
```

## 5.4 バーチャルペイントアプリの作成

### 完全なバーチャルペイントアプリ

```javascript
// バーチャルペイントアプリケーション
let handpose;
let video;
let predictions = [];
let paintCanvas;
let currentColor;
let brushSize = 10;
let isDrawing = false;
let previousPosition = null;

// UIコントロール
let colorPalette = [];
let tools = ['pen', 'eraser', 'spray', 'marker'];
let currentTool = 'pen';
let uiElements = [];

class UIElement {
    constructor(x, y, w, h, label, action) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.label = label;
        this.action = action;
        this.isHovered = false;
        this.isSelected = false;
    }

    checkHover(x, y) {
        this.isHovered = x > this.x && x < this.x + this.w &&
                        y > this.y && y < this.y + this.h;
        return this.isHovered;
    }

    display() {
        push();

        if (this.isSelected) {
            fill(100, 255, 100);
        } else if (this.isHovered) {
            fill(200);
        } else {
            fill(150);
        }

        rect(this.x, this.y, this.w, this.h, 5);

        fill(0);
        textAlign(CENTER, CENTER);
        textSize(12);
        text(this.label, this.x + this.w / 2, this.y + this.h / 2);

        pop();
    }
}

function setup() {
    createCanvas(800, 600);

    // ペイントキャンバスを作成
    paintCanvas = createGraphics(800, 600);
    paintCanvas.clear();

    // ビデオキャプチャ
    video = createCapture(VIDEO);
    video.size(320, 240);
    video.hide();

    // HandPose初期化
    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', results => {
        predictions = results;
    });

    // カラーパレット作成
    setupColorPalette();

    // UIエレメント作成
    setupUIElements();

    currentColor = color(0);
}

function setupColorPalette() {
    // 基本色
    colorPalette = [
        color(0),         // 黒
        color(255),       // 白
        color(255, 0, 0), // 赤
        color(0, 255, 0), // 緑
        color(0, 0, 255), // 青
        color(255, 255, 0), // 黄
        color(255, 0, 255), // マゼンタ
        color(0, 255, 255), // シアン
        color(255, 128, 0), // オレンジ
        color(128, 0, 255)  // 紫
    ];
}

function setupUIElements() {
    // ツールボタン
    let toolY = 10;
    for (let i = 0; i < tools.length; i++) {
        let btn = new UIElement(
            width - 100,
            toolY + i * 40,
            80,
            30,
            tools[i],
            () => { currentTool = tools[i]; }
        );
        uiElements.push(btn);
    }

    // クリアボタン
    let clearBtn = new UIElement(
        width - 100,
        height - 100,
        80,
        30,
        'クリア',
        () => { paintCanvas.clear(); }
    );
    uiElements.push(clearBtn);

    // 保存ボタン
    let saveBtn = new UIElement(
        width - 100,
        height - 60,
        80,
        30,
        '保存',
        () => { saveCanvas(paintCanvas, 'myArtwork', 'png'); }
    );
    uiElements.push(saveBtn);
}

function modelReady() {
    console.log('バーチャルペイント準備完了');
}

function draw() {
    // 背景
    background(240);

    // ペイントキャンバスを表示
    image(paintCanvas, 0, 0);

    // ビデオを小さく表示
    push();
    translate(width - 320, height - 240);
    scale(-1, 1);
    image(video, 0, 0, 320, 240);
    pop();

    // 手の検出と描画
    if (predictions.length > 0) {
        const prediction = predictions[0];
        const landmarks = prediction.landmarks;

        // 人差し指の先端を取得
        const indexTip = landmarks[8];
        const indexPIP = landmarks[6];
        const thumbTip = landmarks[4];

        // 画面座標に変換（ビデオは反転している）
        const x = map(indexTip[0], 0, 640, width, 0);
        const y = map(indexTip[1], 0, 480, 0, height);

        // ジェスチャーで描画制御
        const indexExtended = indexTip[1] < indexPIP[1] - 20;
        const thumbIndexDistance = dist(thumbTip[0], thumbTip[1], indexTip[0], indexTip[1]);

        // つまむジェスチャーで描画
        if (thumbIndexDistance < 40 && indexExtended) {
            isDrawing = true;

            if (previousPosition) {
                drawOnCanvas(previousPosition.x, previousPosition.y, x, y);
            }

            previousPosition = createVector(x, y);
        } else {
            isDrawing = false;
            previousPosition = null;
        }

        // カーソル表示
        drawCursor(x, y);

        // 色選択（薬指を伸ばす）
        const ringTip = landmarks[16];
        const ringPIP = landmarks[14];
        if (ringTip[1] < ringPIP[1] - 20) {
            selectColorByPosition(x, y);
        }

        // UI操作（小指を伸ばす）
        const pinkyTip = landmarks[20];
        const pinkyPIP = landmarks[18];
        if (pinkyTip[1] < pinkyPIP[1] - 20) {
            checkUIInteraction(x, y);
        }
    }

    // UI表示
    drawUI();
}

function drawOnCanvas(x1, y1, x2, y2) {
    paintCanvas.push();

    switch (currentTool) {
        case 'pen':
            paintCanvas.stroke(currentColor);
            paintCanvas.strokeWeight(brushSize);
            paintCanvas.line(x1, y1, x2, y2);
            break;

        case 'eraser':
            paintCanvas.erase();
            paintCanvas.strokeWeight(brushSize * 2);
            paintCanvas.line(x1, y1, x2, y2);
            paintCanvas.noErase();
            break;

        case 'spray':
            paintCanvas.noStroke();
            paintCanvas.fill(currentColor);
            for (let i = 0; i < 10; i++) {
                let offsetX = random(-brushSize, brushSize);
                let offsetY = random(-brushSize, brushSize);
                let size = random(1, 3);
                paintCanvas.ellipse(x2 + offsetX, y2 + offsetY, size);
            }
            break;

        case 'marker':
            paintCanvas.stroke(red(currentColor), green(currentColor), blue(currentColor), 100);
            paintCanvas.strokeWeight(brushSize * 3);
            paintCanvas.strokeCap(SQUARE);
            paintCanvas.line(x1, y1, x2, y2);
            break;
    }

    paintCanvas.pop();
}

function drawCursor(x, y) {
    push();

    // カーソル表示
    noFill();
    stroke(0);
    strokeWeight(2);

    if (isDrawing) {
        stroke(currentColor);
        strokeWeight(3);
    }

    ellipse(x, y, brushSize * 2);

    // ツール表示
    fill(0);
    noStroke();
    textAlign(CENTER);
    textSize(10);
    text(currentTool, x, y - brushSize - 10);

    pop();
}

function selectColorByPosition(x, y) {
    // カラーパレットから色を選択
    let paletteY = 50;
    for (let i = 0; i < colorPalette.length; i++) {
        let colorX = 10;
        let colorY = paletteY + i * 30;

        if (dist(x, y, colorX + 15, colorY + 15) < 20) {
            currentColor = colorPalette[i];
        }
    }
}

function checkUIInteraction(x, y) {
    for (let element of uiElements) {
        if (element.checkHover(x, y)) {
            element.action();

            // 選択状態を更新
            uiElements.forEach(e => e.isSelected = false);
            element.isSelected = true;
        }
    }
}

function drawUI() {
    // カラーパレット
    let paletteY = 50;
    for (let i = 0; i < colorPalette.length; i++) {
        fill(colorPalette[i]);
        stroke(0);
        strokeWeight(2);
        rect(10, paletteY + i * 30, 30, 25);

        // 選択中の色にマーク
        if (colorPalette[i].toString() === currentColor.toString()) {
            noFill();
            stroke(255, 0, 0);
            strokeWeight(3);
            rect(8, paletteY + i * 30 - 2, 34, 29);
        }
    }

    // ブラシサイズ調整
    fill(0);
    textAlign(LEFT);
    textSize(12);
    text('ブラシサイズ', 10, 380);

    // スライダー
    stroke(0);
    strokeWeight(2);
    line(10, 400, 110, 400);

    let sliderX = map(brushSize, 1, 50, 10, 110);
    fill(100);
    ellipse(sliderX, 400, 15);

    // UIエレメント
    for (let element of uiElements) {
        element.display();
    }

    // 情報表示
    fill(0);
    textAlign(LEFT);
    textSize(14);
    text(`ツール: ${currentTool}`, 10, height - 80);
    text(`サイズ: ${brushSize}`, 10, height - 60);
    text(isDrawing ? '描画中' : '待機中', 10, height - 40);
}

function keyPressed() {
    // キーボードショートカット
    if (key === 'c' || key === 'C') {
        paintCanvas.clear();
    } else if (key === 's' || key === 'S') {
        saveCanvas(paintCanvas, 'artwork', 'png');
    } else if (key === '+') {
        brushSize = min(brushSize + 5, 50);
    } else if (key === '-') {
        brushSize = max(brushSize - 5, 1);
    } else if (key >= '1' && key <= '4') {
        currentTool = tools[int(key) - 1];
    }
}
```

## 練習問題

### 問題1: ランドマーク理解
HandPoseが検出する21個のランドマークのうち、人差し指に関連するランドマークの番号をすべて挙げてください。

### 問題2: 指カウント
伸びている指の本数をカウントして表示するプログラムを作成してください。

### 問題3: 手話認識
簡単な手話（数字の1〜5）を認識するプログラムを作成してください。

### 問題4: エアピアノ
手の動きで音を鳴らすエアピアノアプリを作成してください。各指が仮想的な鍵盤に触れると音が鳴るようにしてください。

## 解答

### 問題1の解答
人差し指に関連するランドマーク番号：
- 5: index_mcp（人差し指の付け根）
- 6: index_pip（人差し指の第2関節）
- 7: index_dip（人差し指の第1関節）
- 8: index_tip（人差し指の先端）

### 問題2の解答

```javascript
// 指カウントプログラム
let handpose;
let video;
let predictions = [];
let fingerCount = 0;

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', results => {
        predictions = results;
        if (predictions.length > 0) {
            fingerCount = countFingers(predictions[0].landmarks);
        }
    });
}

function modelReady() {
    console.log('指カウントシステム準備完了');
}

function countFingers(landmarks) {
    let count = 0;

    // 親指（横方向で判定）
    const thumbTip = landmarks[4];
    const thumbMCP = landmarks[2];
    const wrist = landmarks[0];

    // 手の向きを判定
    const isRightHand = landmarks[17][0] > landmarks[5][0];

    if (isRightHand) {
        if (thumbTip[0] > thumbMCP[0] + 30) count++;
    } else {
        if (thumbTip[0] < thumbMCP[0] - 30) count++;
    }

    // 他の4本の指（縦方向で判定）
    const fingerTips = [8, 12, 16, 20];
    const fingerPIPs = [6, 10, 14, 18];

    for (let i = 0; i < 4; i++) {
        const tip = landmarks[fingerTips[i]];
        const pip = landmarks[fingerPIPs[i]];

        // 指が伸びているか判定
        if (tip[1] < pip[1] - 10) {
            count++;
        }
    }

    return count;
}

function draw() {
    image(video, 0, 0);

    // 手を描画
    if (predictions.length > 0) {
        drawHand(predictions[0].landmarks);
    }

    // 指の本数を表示
    displayFingerCount();
}

function drawHand(landmarks) {
    // 各指を伸びているかどうかで色分け
    const fingerGroups = [
        [0, 1, 2, 3, 4],      // 親指
        [0, 5, 6, 7, 8],      // 人差し指
        [0, 9, 10, 11, 12],   // 中指
        [0, 13, 14, 15, 16],  // 薬指
        [0, 17, 18, 19, 20]   // 小指
    ];

    const extendedFingers = getExtendedFingers(landmarks);

    for (let f = 0; f < 5; f++) {
        const indices = fingerGroups[f];
        const isExtended = extendedFingers[f];

        // 色を設定
        if (isExtended) {
            stroke(0, 255, 0);  // 緑：伸びている
            fill(0, 255, 0);
        } else {
            stroke(255, 0, 0);  // 赤：曲がっている
            fill(255, 0, 0);
        }

        strokeWeight(3);

        // 骨格を描画
        for (let i = 1; i < indices.length; i++) {
            const a = landmarks[indices[i - 1]];
            const b = landmarks[indices[i]];
            line(a[0], a[1], b[0], b[1]);
        }

        // 関節点を描画
        noStroke();
        for (let i of indices) {
            ellipse(landmarks[i][0], landmarks[i][1], 8);
        }
    }
}

function getExtendedFingers(landmarks) {
    const extended = [];

    // 親指
    const thumbTip = landmarks[4];
    const thumbMCP = landmarks[2];
    const isRightHand = landmarks[17][0] > landmarks[5][0];

    if (isRightHand) {
        extended.push(thumbTip[0] > thumbMCP[0] + 30);
    } else {
        extended.push(thumbTip[0] < thumbMCP[0] - 30);
    }

    // 他の指
    const fingerTips = [8, 12, 16, 20];
    const fingerPIPs = [6, 10, 14, 18];

    for (let i = 0; i < 4; i++) {
        const tip = landmarks[fingerTips[i]];
        const pip = landmarks[fingerPIPs[i]];
        extended.push(tip[1] < pip[1] - 10);
    }

    return extended;
}

function displayFingerCount() {
    // 背景パネル
    fill(0, 0, 0, 200);
    rect(width / 2 - 150, 20, 300, 100, 10);

    // 数字を大きく表示
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(64);
    text(fingerCount, width / 2, 70);

    // ラベル
    textSize(16);
    text('指の本数', width / 2, 105);

    // 視覚的な表示
    for (let i = 0; i < 5; i++) {
        if (i < fingerCount) {
            fill(0, 255, 0);
        } else {
            fill(100);
        }
        rect(width / 2 - 60 + i * 25, 130, 20, 40);
    }
}
```

### 問題3の解答

```javascript
// 手話数字認識システム（1〜5）
let handpose;
let video;
let predictions = [];
let recognizedNumber = 0;
let confidence = 0;

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', results => {
        predictions = results;
        if (predictions.length > 0) {
            recognizeSignLanguageNumber(predictions[0].landmarks);
        }
    });
}

function modelReady() {
    console.log('手話認識システム準備完了');
}

function recognizeSignLanguageNumber(landmarks) {
    const fingers = getFingerStates(landmarks);

    // 各数字のパターンを定義
    // [親指, 人差し指, 中指, 薬指, 小指]
    const patterns = {
        1: [false, true, false, false, false],
        2: [false, true, true, false, false],
        3: [true, true, true, false, false],
        4: [false, true, true, true, true],
        5: [true, true, true, true, true]
    };

    // パターンマッチング
    let bestMatch = 0;
    let bestScore = 0;

    for (let num in patterns) {
        let score = 0;
        for (let i = 0; i < 5; i++) {
            if (fingers[i] === patterns[num][i]) {
                score++;
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestMatch = parseInt(num);
        }
    }

    // 信頼度計算
    confidence = bestScore / 5;

    if (confidence > 0.8) {
        recognizedNumber = bestMatch;
    } else {
        recognizedNumber = 0;
    }
}

function getFingerStates(landmarks) {
    const states = [];

    // 親指（特別な判定）
    const thumbTip = landmarks[4];
    const thumbIP = landmarks[3];
    const thumbMCP = landmarks[2];
    const indexMCP = landmarks[5];

    // 親指が横に開いているか判定
    const thumbDist = dist(thumbTip[0], thumbTip[1], indexMCP[0], indexMCP[1]);
    states.push(thumbDist > 60);

    // 他の指
    const fingerTips = [8, 12, 16, 20];
    const fingerPIPs = [6, 10, 14, 18];

    for (let i = 0; i < 4; i++) {
        const tip = landmarks[fingerTips[i]];
        const pip = landmarks[fingerPIPs[i]];
        states.push(tip[1] < pip[1] - 20);
    }

    return states;
}

function draw() {
    image(video, 0, 0);

    if (predictions.length > 0) {
        drawHandWithHighlight(predictions[0].landmarks);
    }

    displayRecognizedNumber();
}

function drawHandWithHighlight(landmarks) {
    // 基本の手を描画
    stroke(100);
    strokeWeight(2);

    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
        [5, 9], [9, 13], [13, 17]
    ];

    for (let conn of connections) {
        line(landmarks[conn[0]][0], landmarks[conn[0]][1],
             landmarks[conn[1]][0], landmarks[conn[1]][1]);
    }

    // ランドマークを描画
    fill(0, 200, 0);
    noStroke();
    for (let landmark of landmarks) {
        ellipse(landmark[0], landmark[1], 6);
    }
}

function displayRecognizedNumber() {
    // 背景パネル
    fill(0, 0, 0, 200);
    rect(20, 20, 200, 150, 10);

    // 認識された数字
    fill(255);
    textAlign(CENTER);

    if (recognizedNumber > 0) {
        textSize(72);
        text(recognizedNumber, 120, 100);

        // 信頼度バー
        fill(0, 255, 0, 100);
        rect(30, 140, confidence * 180, 20);

        fill(255);
        textSize(12);
        text(`信頼度: ${(confidence * 100).toFixed(0)}%`, 120, 155);
    } else {
        textSize(24);
        text('手話を検出中...', 120, 95);
    }

    // 説明
    textSize(14);
    text('1〜5の数字を', 120, 40);
    text('手話で表してください', 120, 60);
}
```

### 問題4の解答

```javascript
// エアピアノアプリ
let handpose;
let video;
let predictions = [];
let piano;
let isPlaying = [];

class VirtualPiano {
    constructor() {
        this.keys = [];
        this.whiteKeyWidth = 40;
        this.blackKeyWidth = 25;

        // 白鍵の作成（C〜B）
        const whiteNotes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'];
        for (let i = 0; i < whiteNotes.length; i++) {
            this.keys.push({
                x: 150 + i * this.whiteKeyWidth,
                y: 300,
                width: this.whiteKeyWidth,
                height: 150,
                note: whiteNotes[i],
                isWhite: true,
                isPressed: false,
                oscillator: null
            });
        }

        // 黒鍵の作成
        const blackNotes = ['C#4', 'D#4', 'F#4', 'G#4', 'A#4'];
        const blackPositions = [0.75, 1.75, 3.75, 4.75, 5.75];
        for (let i = 0; i < blackNotes.length; i++) {
            this.keys.push({
                x: 150 + blackPositions[i] * this.whiteKeyWidth,
                y: 300,
                width: this.blackKeyWidth,
                height: 90,
                note: blackNotes[i],
                isWhite: false,
                isPressed: false,
                oscillator: null
            });
        }
    }

    checkKey(x, y, fingerName) {
        // 黒鍵を先にチェック（上にあるため）
        for (let key of this.keys.filter(k => !k.isWhite)) {
            if (x > key.x && x < key.x + key.width &&
                y > key.y && y < key.y + key.height) {
                this.pressKey(key, fingerName);
                return;
            }
        }

        // 白鍵をチェック
        for (let key of this.keys.filter(k => k.isWhite)) {
            if (x > key.x && x < key.x + key.width &&
                y > key.y && y < key.y + key.height) {
                this.pressKey(key, fingerName);
                return;
            }
        }
    }

    pressKey(key, fingerName) {
        if (!key.isPressed) {
            key.isPressed = true;
            key.pressedBy = fingerName;
            this.playNote(key.note);
        }
    }

    releaseKey(fingerName) {
        for (let key of this.keys) {
            if (key.pressedBy === fingerName) {
                key.isPressed = false;
                key.pressedBy = null;
                this.stopNote(key.note);
            }
        }
    }

    playNote(note) {
        // 音階の周波数
        const frequencies = {
            'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
            'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
            'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
            'C5': 523.25
        };

        // Web Audio APIで音を生成
        if (!window.audioContext) {
            window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        const osc = window.audioContext.createOscillator();
        const gainNode = window.audioContext.createGain();

        osc.frequency.value = frequencies[note];
        osc.type = 'sine';

        gainNode.gain.value = 0.3;
        gainNode.gain.exponentialRampToValueAtTime(0.01,
            window.audioContext.currentTime + 0.5);

        osc.connect(gainNode);
        gainNode.connect(window.audioContext.destination);

        osc.start();

        // 音を保存（後で止めるため）
        const key = this.keys.find(k => k.note === note);
        if (key) key.oscillator = { osc, gainNode };
    }

    stopNote(note) {
        const key = this.keys.find(k => k.note === note);
        if (key && key.oscillator) {
            key.oscillator.osc.stop();
            key.oscillator = null;
        }
    }

    display() {
        // 白鍵を描画
        for (let key of this.keys.filter(k => k.isWhite)) {
            if (key.isPressed) {
                fill(200);
            } else {
                fill(255);
            }
            stroke(0);
            strokeWeight(2);
            rect(key.x, key.y, key.width, key.height);

            // 音名表示
            fill(0);
            noStroke();
            textAlign(CENTER);
            textSize(12);
            text(key.note, key.x + key.width / 2, key.y + key.height - 10);
        }

        // 黒鍵を描画
        for (let key of this.keys.filter(k => !k.isWhite)) {
            if (key.isPressed) {
                fill(100);
            } else {
                fill(0);
            }
            stroke(0);
            strokeWeight(2);
            rect(key.x, key.y, key.width, key.height);
        }
    }
}

function setup() {
    createCanvas(640, 480);

    video = createCapture(VIDEO);
    video.size(320, 240);
    video.hide();

    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', results => {
        predictions = results;
    });

    piano = new VirtualPiano();

    // 各指の演奏状態を初期化
    isPlaying = {
        thumb: false,
        index: false,
        middle: false,
        ring: false,
        pinky: false
    };
}

function modelReady() {
    console.log('エアピアノ準備完了');
}

function draw() {
    background(240);

    // ピアノを描画
    piano.display();

    // ビデオを小さく表示
    push();
    translate(width - 160, 10);
    scale(-0.5, 0.5);
    image(video, 0, 0);
    pop();

    // 手の検出と演奏
    if (predictions.length > 0) {
        playWithHand(predictions[0].landmarks);
        drawHandOnPiano(predictions[0].landmarks);
    }

    // 説明
    displayInstructions();
}

function playWithHand(landmarks) {
    // 各指先の位置を取得
    const fingerTips = {
        thumb: landmarks[4],
        index: landmarks[8],
        middle: landmarks[12],
        ring: landmarks[16],
        pinky: landmarks[20]
    };

    const fingerPIPs = {
        thumb: landmarks[2],
        index: landmarks[6],
        middle: landmarks[10],
        ring: landmarks[14],
        pinky: landmarks[18]
    };

    // 各指で鍵盤をチェック
    for (let finger in fingerTips) {
        const tip = fingerTips[finger];
        const pip = fingerPIPs[finger];

        // 画面座標に変換
        const x = map(tip[0], 0, 640, width, 0);
        const y = map(tip[1], 0, 480, 0, height);

        // 指が下がっているか判定（演奏動作）
        if (tip[1] > pip[1] + 20) {
            if (!isPlaying[finger]) {
                piano.checkKey(x, y, finger);
                isPlaying[finger] = true;
            }
        } else {
            if (isPlaying[finger]) {
                piano.releaseKey(finger);
                isPlaying[finger] = false;
            }
        }
    }
}

function drawHandOnPiano(landmarks) {
    // 指先の位置を表示
    const fingerTips = [4, 8, 12, 16, 20];
    const fingerColors = [
        color(255, 0, 0),
        color(255, 255, 0),
        color(0, 255, 0),
        color(0, 255, 255),
        color(255, 0, 255)
    ];

    for (let i = 0; i < fingerTips.length; i++) {
        const tip = landmarks[fingerTips[i]];
        const x = map(tip[0], 0, 640, width, 0);
        const y = map(tip[1], 0, 480, 0, height);

        fill(fingerColors[i]);
        noStroke();
        ellipse(x, y, 15);
    }
}

function displayInstructions() {
    fill(0);
    textAlign(LEFT);
    textSize(14);
    text('エアピアノ', 10, 30);
    text('指を下げて演奏', 10, 50);
    text('指を上げて音を止める', 10, 70);
}
```

## まとめ

第5章では、HandPoseを使用した手の認識とジェスチャー検出について学習しました。21個のランドマークの検出、ジェスチャー認識、動的ジェスチャーの実装、そしてバーチャルペイントアプリの作成まで、実践的な技術を習得しました。

次章では、音声認識とテキスト分析について学習します。