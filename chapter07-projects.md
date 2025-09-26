# 第7章: 実践プロジェクト

## 7.1 インタラクティブなWebカメラフィルター

### ARフィルターアプリケーション

```javascript
// 統合型ARカメラフィルター
let video;
let poseNet;
let handpose;
let faceapi;
let poses = [];
let hands = [];
let faces = [];

// フィルター設定
let currentFilter = 'none';
let filters = {
    'none': 'なし',
    'sunglasses': 'サングラス',
    'animal': '動物',
    'comic': 'コミック',
    'sparkle': 'キラキラ',
    'rainbow': 'レインボー'
};

// エフェクト用の変数
let particles = [];
let faceGraphics;

function preload() {
    // カスタム画像を読み込み
    // sunglassesImg = loadImage('assets/sunglasses.png');
    // animalEars = loadImage('assets/ears.png');
}

function setup() {
    createCanvas(640, 480);

    // ビデオ設定
    video = createCapture(VIDEO);
    video.size(width, height);
    video.hide();

    // 各種AIモデルを初期化
    setupModels();

    // UIを作成
    createFilterUI();

    // オフスクリーンキャンバス
    faceGraphics = createGraphics(width, height);
}

function setupModels() {
    // PoseNet初期化
    poseNet = ml5.poseNet(video, () => {
        console.log('PoseNet準備完了');
    });
    poseNet.on('pose', results => {
        poses = results;
    });

    // HandPose初期化
    handpose = ml5.handpose(video, () => {
        console.log('HandPose準備完了');
    });
    handpose.on('predict', results => {
        hands = results;
    });

    // Face-API初期化（簡易版として顔の位置を検出）
    // faceapi = ml5.faceApi(video, detectionOptions, modelReady);
}

function createFilterUI() {
    // フィルター選択ボタン
    let y = 10;
    for (let key in filters) {
        let btn = createButton(filters[key]);
        btn.position(width + 10, y);
        btn.mousePressed(() => {
            currentFilter = key;
            resetEffects();
        });
        y += 35;
    }

    // スクリーンショットボタン
    let captureBtn = createButton('📸 撮影');
    captureBtn.position(width + 10, height - 40);
    captureBtn.mousePressed(takeSnapshot);
}

function resetEffects() {
    particles = [];
}

function takeSnapshot() {
    saveCanvas('ar_filter_' + Date.now(), 'png');
}

function draw() {
    // ビデオを描画
    push();
    translate(width, 0);
    scale(-1, 1);  // 鏡像
    image(video, 0, 0);
    pop();

    // 選択されたフィルターを適用
    applyFilter();

    // パーティクルエフェクト
    updateParticles();

    // UI情報
    drawInfo();
}

function applyFilter() {
    switch(currentFilter) {
        case 'sunglasses':
            applySunglassesFilter();
            break;
        case 'animal':
            applyAnimalFilter();
            break;
        case 'comic':
            applyComicFilter();
            break;
        case 'sparkle':
            applySparkleFilter();
            break;
        case 'rainbow':
            applyRainbowFilter();
            break;
    }
}

function applySunglassesFilter() {
    if (poses.length > 0) {
        let pose = poses[0].pose;

        // 目の位置を取得
        let leftEye = pose.leftEye;
        let rightEye = pose.rightEye;
        let nose = pose.nose;

        if (leftEye.score > 0.2 && rightEye.score > 0.2) {
            // サングラスを描画
            push();

            // 目の中心と角度を計算
            let eyeCenterX = width - (leftEye.x + rightEye.x) / 2;
            let eyeCenterY = (leftEye.y + rightEye.y) / 2;
            let eyeDistance = dist(leftEye.x, leftEye.y, rightEye.x, rightEye.y);
            let angle = atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

            translate(eyeCenterX, eyeCenterY);
            rotate(-angle);

            // サングラスの形を描画
            fill(0, 0, 0, 200);
            stroke(0);
            strokeWeight(3);

            // 左レンズ
            ellipse(-eyeDistance / 2, 0, eyeDistance * 0.6, eyeDistance * 0.5);
            // 右レンズ
            ellipse(eyeDistance / 2, 0, eyeDistance * 0.6, eyeDistance * 0.5);
            // ブリッジ
            line(-eyeDistance * 0.2, 0, eyeDistance * 0.2, 0);

            // テンプル
            line(-eyeDistance / 2 - eyeDistance * 0.3, 0, -eyeDistance, 0);
            line(eyeDistance / 2 + eyeDistance * 0.3, 0, eyeDistance, 0);

            pop();
        }
    }
}

function applyAnimalFilter() {
    if (poses.length > 0) {
        let pose = poses[0].pose;
        let nose = pose.nose;
        let leftEar = pose.leftEar;
        let rightEar = pose.rightEar;

        if (nose.score > 0.2) {
            // 動物の鼻
            push();
            fill(255, 100, 100);
            noStroke();
            ellipse(width - nose.x, nose.y, 30, 20);

            // ひげ
            stroke(0);
            strokeWeight(2);
            for (let i = -2; i <= 2; i++) {
                if (i !== 0) {
                    line(width - nose.x - 40, nose.y + i * 10,
                         width - nose.x - 10, nose.y + i * 8);
                    line(width - nose.x + 10, nose.y + i * 8,
                         width - nose.x + 40, nose.y + i * 10);
                }
            }
            pop();
        }

        // 動物の耳
        if (leftEar.score > 0.2 && rightEar.score > 0.2) {
            push();
            fill(150, 100, 50);
            stroke(100, 50, 0);
            strokeWeight(2);

            // 左耳
            triangle(width - leftEar.x - 20, leftEar.y - 40,
                    width - leftEar.x, leftEar.y - 80,
                    width - leftEar.x + 20, leftEar.y - 40);

            // 右耳
            triangle(width - rightEar.x - 20, rightEar.y - 40,
                    width - rightEar.x, rightEar.y - 80,
                    width - rightEar.x + 20, rightEar.y - 40);
            pop();
        }
    }
}

function applyComicFilter() {
    // コミック風エフェクト
    push();
    drawingContext.filter = 'contrast(200%) saturate(150%)';
    image(video, 0, 0, width, height);
    pop();

    // スピードライン
    if (poses.length > 0 && random() < 0.1) {
        let pose = poses[0].pose;
        let centerX = width - pose.nose.x;
        let centerY = pose.nose.y;

        stroke(255, 255, 0);
        strokeWeight(2);
        for (let i = 0; i < 20; i++) {
            let angle = random(TWO_PI);
            let startR = 100;
            let endR = 300;
            line(centerX + cos(angle) * startR,
                 centerY + sin(angle) * startR,
                 centerX + cos(angle) * endR,
                 centerY + sin(angle) * endR);
        }
    }

    // 吹き出し効果
    if (hands.length > 0) {
        let hand = hands[0].landmarks;
        let palmX = width - hand[0][0];
        let palmY = hand[0][1];

        // 拳を作っているか判定（簡易版）
        let isFist = true;
        for (let i = 8; i < hand.length; i += 4) {
            if (hand[i][1] < hand[i - 2][1]) {
                isFist = false;
                break;
            }
        }

        if (isFist) {
            drawSpeechBubble(palmX, palmY - 50, 'POW!');
        }
    }
}

function drawSpeechBubble(x, y, text) {
    push();
    fill(255);
    stroke(0);
    strokeWeight(3);

    // 吹き出し本体
    ellipse(x, y, 100, 60);

    // 吹き出しの尾
    triangle(x - 10, y + 20, x + 10, y + 20, x, y + 40);

    // テキスト
    fill(255, 0, 0);
    textAlign(CENTER, CENTER);
    textSize(24);
    textStyle(BOLD);
    text(text, x, y - 5);

    pop();
}

function applySparkleFilter() {
    // 手の位置にキラキラエフェクト
    if (hands.length > 0) {
        for (let hand of hands) {
            for (let landmark of hand.landmarks) {
                if (random() < 0.3) {
                    particles.push(new Sparkle(width - landmark[0], landmark[1]));
                }
            }
        }
    }

    // 顔の周りにもキラキラ
    if (poses.length > 0) {
        let pose = poses[0].pose;
        if (pose.nose.score > 0.2 && random() < 0.1) {
            let angle = random(TWO_PI);
            let r = random(50, 100);
            particles.push(new Sparkle(
                width - pose.nose.x + cos(angle) * r,
                pose.nose.y + sin(angle) * r
            ));
        }
    }
}

function applyRainbowFilter() {
    // レインボーオーバーレイ
    push();
    blendMode(ADD);

    for (let i = 0; i < 7; i++) {
        let hue = map(i, 0, 7, 0, 360);
        fill(`hsla(${hue}, 100%, 50%, 0.1)`);
        noStroke();

        let offset = sin(frameCount * 0.02 + i) * 20;
        rect(0, i * height / 7 + offset, width, height / 7);
    }

    pop();
}

// パーティクルクラス
class Sparkle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.vx = random(-2, 2);
        this.vy = random(-2, 2);
        this.size = random(5, 15);
        this.life = 255;
        this.color = color(random(255), random(255), random(255));
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life -= 5;
        this.size *= 0.98;
    }

    display() {
        push();
        noStroke();
        fill(red(this.color), green(this.color), blue(this.color), this.life);

        // 星形を描画
        star(this.x, this.y, this.size / 2, this.size, 5);

        pop();
    }

    isDead() {
        return this.life <= 0;
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

function updateParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        particles[i].update();
        particles[i].display();

        if (particles[i].isDead()) {
            particles.splice(i, 1);
        }
    }
}

function drawInfo() {
    // 現在のフィルター名
    fill(255);
    stroke(0);
    strokeWeight(2);
    textAlign(LEFT);
    textSize(20);
    text('フィルター: ' + filters[currentFilter], 10, 30);

    // FPS
    textSize(12);
    text('FPS: ' + frameRate().toFixed(1), 10, height - 10);
}
```

## 7.2 ジェスチャーで操作するゲーム

### 手のジェスチャーで遊ぶ音楽ゲーム

```javascript
// リズムゲーム with HandPose
let handpose;
let video;
let predictions = [];

// ゲーム変数
let gameState = 'menu';  // 'menu', 'playing', 'gameOver'
let score = 0;
let combo = 0;
let maxCombo = 0;
let health = 100;

// ノーツ（音符）
let notes = [];
let noteSpeed = 3;
let lastNoteTime = 0;
let noteInterval = 1000;  // ミリ秒

// レーン設定
let lanes = [];
let laneCount = 5;

// エフェクト
let hitEffects = [];

// 音楽とリズム
let bpm = 120;
let beatInterval;

class Note {
    constructor(laneIndex, gesture) {
        this.lane = laneIndex;
        this.x = lanes[laneIndex].x;
        this.y = -50;
        this.gesture = gesture;  // 'open', 'close', 'point', etc.
        this.size = 40;
        this.hit = false;
        this.missed = false;
    }

    update() {
        this.y += noteSpeed;

        // 判定ラインを越えたらミス
        if (this.y > height - 50 && !this.hit && !this.missed) {
            this.missed = true;
            handleMiss();
        }
    }

    display() {
        if (this.hit || this.missed) return;

        // ノートを描画
        push();

        // ジェスチャーに応じた色
        switch(this.gesture) {
            case 'open':
                fill(0, 255, 0);
                break;
            case 'close':
                fill(255, 0, 0);
                break;
            case 'point':
                fill(0, 0, 255);
                break;
            default:
                fill(255, 255, 0);
        }

        noStroke();
        ellipse(this.x, this.y, this.size);

        // ジェスチャーアイコン
        fill(255);
        textAlign(CENTER, CENTER);
        textSize(20);

        switch(this.gesture) {
            case 'open':
                text('✋', this.x, this.y);
                break;
            case 'close':
                text('✊', this.x, this.y);
                break;
            case 'point':
                text('👆', this.x, this.y);
                break;
        }

        pop();
    }

    checkHit(handGesture, handX) {
        // 判定エリア内にあるか
        let inJudgmentArea = this.y > height - 100 && this.y < height - 20;
        let correctLane = abs(handX - this.x) < 50;
        let correctGesture = handGesture === this.gesture;

        if (inJudgmentArea && correctLane && correctGesture) {
            this.hit = true;

            // タイミング精度
            let timing = abs(this.y - (height - 60));
            let accuracy;

            if (timing < 10) {
                accuracy = 'PERFECT';
                score += 100;
            } else if (timing < 25) {
                accuracy = 'GREAT';
                score += 50;
            } else {
                accuracy = 'GOOD';
                score += 25;
            }

            combo++;
            maxCombo = max(maxCombo, combo);

            // エフェクト作成
            createHitEffect(this.x, height - 60, accuracy);

            return true;
        }

        return false;
    }
}

class HitEffect {
    constructor(x, y, text) {
        this.x = x;
        this.y = y;
        this.text = text;
        this.life = 60;
        this.size = 20;
    }

    update() {
        this.y -= 2;
        this.life -= 2;
        this.size += 0.5;
    }

    display() {
        push();

        let alpha = map(this.life, 0, 60, 0, 255);

        if (this.text === 'PERFECT') {
            fill(255, 215, 0, alpha);
        } else if (this.text === 'GREAT') {
            fill(0, 255, 0, alpha);
        } else {
            fill(0, 150, 255, alpha);
        }

        textAlign(CENTER, CENTER);
        textSize(this.size);
        textStyle(BOLD);
        text(this.text, this.x, this.y);

        pop();
    }

    isDead() {
        return this.life <= 0;
    }
}

function setup() {
    createCanvas(640, 480);

    // ビデオ設定
    video = createCapture(VIDEO);
    video.size(320, 240);
    video.hide();

    // HandPose初期化
    handpose = ml5.handpose(video, modelReady);
    handpose.on('predict', results => {
        predictions = results;
    });

    // レーン初期化
    for (let i = 0; i < laneCount; i++) {
        lanes.push({
            x: width / (laneCount + 1) * (i + 1),
            active: false
        });
    }

    // ビート間隔計算
    beatInterval = 60000 / bpm;

    // UI作成
    createGameUI();
}

function modelReady() {
    console.log('リズムゲーム準備完了');
}

function createGameUI() {
    // スタートボタン
    let startBtn = createButton('ゲーム開始');
    startBtn.position(width / 2 - 50, height / 2);
    startBtn.mousePressed(() => {
        gameState = 'playing';
        startBtn.hide();
        resetGame();
    });
}

function resetGame() {
    score = 0;
    combo = 0;
    maxCombo = 0;
    health = 100;
    notes = [];
    hitEffects = [];
    lastNoteTime = millis();
}

function draw() {
    background(20);

    if (gameState === 'menu') {
        drawMenu();
    } else if (gameState === 'playing') {
        playGame();
    } else if (gameState === 'gameOver') {
        drawGameOver();
    }
}

function drawMenu() {
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(48);
    text('リズムゲーム', width / 2, height / 3);

    textSize(20);
    text('手のジェスチャーで音符をキャッチ！', width / 2, height / 2 - 50);

    // ビデオプレビュー
    push();
    translate(width / 2 + 160, height - 120);
    scale(-0.5, 0.5);
    image(video, 0, 0);
    pop();
}

function playGame() {
    // ゲームエリア描画
    drawGameArea();

    // ノーツ生成
    if (millis() - lastNoteTime > noteInterval) {
        generateNote();
        lastNoteTime = millis();
    }

    // ノーツ更新・描画
    for (let i = notes.length - 1; i >= 0; i--) {
        notes[i].update();
        notes[i].display();

        if (notes[i].y > height || notes[i].hit) {
            notes.splice(i, 1);
        }
    }

    // 手の検出と判定
    if (predictions.length > 0) {
        processHandInput(predictions[0]);
    }

    // エフェクト更新
    updateEffects();

    // UI表示
    drawGameUI();

    // ゲームオーバーチェック
    if (health <= 0) {
        gameState = 'gameOver';
    }
}

function drawGameArea() {
    // レーン
    stroke(100);
    strokeWeight(2);
    for (let lane of lanes) {
        line(lane.x, 0, lane.x, height);

        if (lane.active) {
            fill(255, 255, 0, 50);
            noStroke();
            rect(lane.x - 40, 0, 80, height);
            lane.active = false;
        }
    }

    // 判定ライン
    stroke(255, 0, 0);
    strokeWeight(3);
    line(0, height - 60, width, height - 60);

    // 判定エリア
    fill(255, 255, 255, 30);
    rect(0, height - 100, width, 80);
}

function generateNote() {
    // ランダムなレーンとジェスチャー
    let laneIndex = floor(random(laneCount));
    let gestures = ['open', 'close', 'point'];
    let gesture = random(gestures);

    notes.push(new Note(laneIndex, gesture));

    // 難易度調整
    if (score > 500) {
        noteSpeed = 4;
        noteInterval = 800;
    }
    if (score > 1000) {
        noteSpeed = 5;
        noteInterval = 600;
    }
}

function processHandInput(hand) {
    const landmarks = hand.landmarks;

    // 手の位置（手首の位置）
    let handX = map(landmarks[0][0], 0, 640, width, 0);

    // ジェスチャー認識
    let gesture = recognizeGesture(landmarks);

    // アクティブレーンを更新
    for (let i = 0; i < lanes.length; i++) {
        if (abs(handX - lanes[i].x) < 40) {
            lanes[i].active = true;
        }
    }

    // ノーツとの判定
    for (let note of notes) {
        if (!note.hit && !note.missed) {
            if (note.checkHit(gesture, handX)) {
                break;  // 1度に1つのノートのみヒット
            }
        }
    }

    // 手の位置を表示
    drawHandIndicator(handX, gesture);
}

function recognizeGesture(landmarks) {
    // 簡易的なジェスチャー認識
    let fingerTips = [8, 12, 16, 20];  // 人差し指〜小指の先端
    let fingerPIPs = [6, 10, 14, 18];  // 第二関節

    let extendedCount = 0;

    for (let i = 0; i < 4; i++) {
        if (landmarks[fingerTips[i]][1] < landmarks[fingerPIPs[i]][1]) {
            extendedCount++;
        }
    }

    if (extendedCount >= 3) {
        return 'open';
    } else if (extendedCount === 1) {
        return 'point';
    } else {
        return 'close';
    }
}

function drawHandIndicator(x, gesture) {
    push();

    fill(255, 255, 0);
    noStroke();

    // 現在のジェスチャーを表示
    textAlign(CENTER);
    textSize(30);

    let icon = '';
    switch(gesture) {
        case 'open':
            icon = '✋';
            break;
        case 'close':
            icon = '✊';
            break;
        case 'point':
            icon = '👆';
            break;
    }

    text(icon, x, height - 30);

    pop();
}

function handleMiss() {
    combo = 0;
    health -= 10;

    // ミスエフェクト
    createHitEffect(width / 2, height / 2, 'MISS');
}

function createHitEffect(x, y, text) {
    hitEffects.push(new HitEffect(x, y, text));
}

function updateEffects() {
    for (let i = hitEffects.length - 1; i >= 0; i--) {
        hitEffects[i].update();
        hitEffects[i].display();

        if (hitEffects[i].isDead()) {
            hitEffects.splice(i, 1);
        }
    }
}

function drawGameUI() {
    // スコア表示
    fill(255);
    textAlign(LEFT);
    textSize(24);
    text('スコア: ' + score, 10, 30);
    text('コンボ: ' + combo, 10, 60);

    // 体力バー
    fill(100);
    rect(10, 90, 200, 20);

    if (health > 50) {
        fill(0, 255, 0);
    } else if (health > 25) {
        fill(255, 255, 0);
    } else {
        fill(255, 0, 0);
    }

    rect(10, 90, health * 2, 20);

    // ビデオ表示（小さく）
    push();
    translate(width - 160, 10);
    scale(-0.5, 0.5);
    image(video, 0, 0);
    pop();
}

function drawGameOver() {
    fill(255);
    textAlign(CENTER);
    textSize(48);
    text('GAME OVER', width / 2, height / 3);

    textSize(24);
    text('最終スコア: ' + score, width / 2, height / 2);
    text('最大コンボ: ' + maxCombo, width / 2, height / 2 + 40);

    textSize(16);
    text('スペースキーでリトライ', width / 2, height * 2 / 3);
}

function keyPressed() {
    if (key === ' ' && gameState === 'gameOver') {
        gameState = 'playing';
        resetGame();
    }
}
```

## 7.3 感情認識チャットボット

### AIアシスタントアプリケーション

```javascript
// 統合型AIアシスタント
let sentiment;
let soundClassifier;
let video;
let poseNet;
let faceExpression;

// アシスタントの状態
let assistant = {
    mood: 'neutral',
    message: 'こんにちは！今日はどんな気分ですか？',
    avatar: null,
    responses: [],
    userMood: 0.5
};

// ユーザー入力
let userInput;
let chatHistory = [];

// 表情とポーズ
let currentPose = null;
let currentExpression = 'neutral';

function setup() {
    createCanvas(800, 600);

    // AIモデルの初期化
    setupAIModels();

    // UI作成
    createAssistantUI();

    // アバター初期化
    assistant.avatar = new Avatar(width - 150, 100);
}

function setupAIModels() {
    // 感情分析
    sentiment = ml5.sentiment('movieReviews', () => {
        console.log('感情分析準備完了');
    });

    // 音声認識
    soundClassifier = ml5.soundClassifier('SpeechCommands18w', () => {
        console.log('音声認識準備完了');
        soundClassifier.classify(gotSoundCommand);
    });

    // ビデオとポーズ検出
    video = createCapture(VIDEO);
    video.size(320, 240);
    video.hide();

    poseNet = ml5.poseNet(video, () => {
        console.log('ポーズ検出準備完了');
    });
    poseNet.on('pose', results => {
        if (results.length > 0) {
            currentPose = results[0].pose;
            analyzeUserState();
        }
    });
}

function createAssistantUI() {
    // テキスト入力
    userInput = createElement('textarea');
    userInput.position(20, height - 100);
    userInput.size(500, 60);
    userInput.attribute('placeholder', 'メッセージを入力...');

    // 送信ボタン
    let sendBtn = createButton('送信');
    sendBtn.position(530, height - 100);
    sendBtn.mousePressed(sendMessage);
}

function sendMessage() {
    let text = userInput.value();
    if (text.length > 0) {
        // チャット履歴に追加
        chatHistory.push({
            sender: 'user',
            message: text,
            timestamp: new Date().toLocaleTimeString()
        });

        // 感情分析
        let prediction = sentiment.predict(text);
        assistant.userMood = prediction.score;

        // アシスタントの応答生成
        generateResponse(text, prediction.score);

        // 入力をクリア
        userInput.value('');
    }
}

function generateResponse(userText, sentimentScore) {
    let response = '';

    // 感情に応じた応答
    if (sentimentScore < 0.3) {
        // ネガティブ
        assistant.mood = 'concerned';
        response = getEncouragingResponse();
    } else if (sentimentScore > 0.7) {
        // ポジティブ
        assistant.mood = 'happy';
        response = getPositiveResponse();
    } else {
        // ニュートラル
        assistant.mood = 'neutral';
        response = getNeutralResponse();
    }

    // コンテキストを考慮した応答
    if (userText.includes('疲れ')) {
        response += '\n休憩も大切です。深呼吸をしてリラックスしましょう。';
    } else if (userText.includes('嬉しい')) {
        response += '\nそれは素晴らしいですね！その気持ちを大切にしてください。';
    }

    // 応答を追加
    chatHistory.push({
        sender: 'assistant',
        message: response,
        timestamp: new Date().toLocaleTimeString()
    });

    assistant.message = response;
}

function getEncouragingResponse() {
    const responses = [
        '大変そうですね。でも、きっと乗り越えられます！',
        '今は辛いかもしれませんが、必ず良い方向に向かいます。',
        '一歩ずつ進んでいきましょう。私がサポートします。',
        '深呼吸をして、少しリラックスしてみませんか？'
    ];
    return random(responses);
}

function getPositiveResponse() {
    const responses = [
        '素晴らしいですね！その調子です！',
        'とても前向きで良いですね！',
        'あなたの元気が伝わってきます！',
        'その気持ちを維持していきましょう！'
    ];
    return random(responses);
}

function getNeutralResponse() {
    const responses = [
        'なるほど、理解しました。',
        'お話を聞かせていただき、ありがとうございます。',
        'それについてもう少し詳しく教えていただけますか？',
        '興味深いお話ですね。'
    ];
    return random(responses);
}

function analyzeUserState() {
    if (!currentPose) return;

    // 姿勢分析
    let leftShoulder = currentPose.leftShoulder;
    let rightShoulder = currentPose.rightShoulder;

    if (leftShoulder.score > 0.3 && rightShoulder.score > 0.3) {
        // 肩の高さで緊張度を判定
        let shoulderDiff = abs(leftShoulder.y - rightShoulder.y);

        if (shoulderDiff > 20) {
            currentExpression = 'tense';
        } else {
            currentExpression = 'relaxed';
        }
    }
}

function gotSoundCommand(error, results) {
    if (!error && results.length > 0) {
        let command = results[0].label;

        // 音声コマンドに応答
        if (command === 'yes') {
            assistant.message = 'はい、分かりました！';
        } else if (command === 'no') {
            assistant.message = 'いいえ、違いますね。';
        } else if (command === 'stop') {
            assistant.message = '了解です。停止します。';
        }
    }
}

function draw() {
    // 背景グラデーション
    drawBackground();

    // チャット表示
    drawChatHistory();

    // アシスタントアバター
    assistant.avatar.update(assistant.mood);
    assistant.avatar.display();

    // ユーザー状態表示
    drawUserStatus();

    // ビデオプレビュー
    drawVideoPreview();
}

function drawBackground() {
    // ムードに応じた背景色
    let c1, c2;

    switch(assistant.mood) {
        case 'happy':
            c1 = color(255, 200, 100);
            c2 = color(255, 150, 200);
            break;
        case 'concerned':
            c1 = color(100, 150, 200);
            c2 = color(150, 100, 200);
            break;
        default:
            c1 = color(200, 220, 240);
            c2 = color(240, 200, 220);
    }

    for (let i = 0; i <= height; i++) {
        let inter = map(i, 0, height, 0, 1);
        let c = lerpColor(c1, c2, inter);
        stroke(c);
        line(0, i, width, i);
    }
}

function drawChatHistory() {
    // チャットエリア背景
    fill(255, 255, 255, 200);
    rect(20, 100, 500, height - 220);

    // メッセージ表示
    let y = 120;

    for (let i = max(0, chatHistory.length - 5); i < chatHistory.length; i++) {
        let msg = chatHistory[i];

        if (msg.sender === 'user') {
            // ユーザーメッセージ
            fill(100, 150, 255);
            textAlign(RIGHT);
            text(msg.message, 500, y);
        } else {
            // アシスタントメッセージ
            fill(50);
            textAlign(LEFT);
            text(msg.message, 40, y);
        }

        // タイムスタンプ
        fill(150);
        textSize(10);
        text(msg.timestamp, msg.sender === 'user' ? 450 : 40, y + 15);

        y += 60;
    }
}

function drawUserStatus() {
    push();
    translate(550, 300);

    // ステータスパネル
    fill(255, 255, 255, 180);
    rect(0, 0, 200, 150);

    fill(0);
    textAlign(LEFT);
    textSize(14);
    text('ユーザー状態', 10, 20);

    // 感情スコア
    textSize(12);
    text('感情: ' + (assistant.userMood > 0.5 ? 'ポジティブ' : 'ネガティブ'), 10, 40);

    // 感情メーター
    fill(200);
    rect(10, 50, 180, 20);

    if (assistant.userMood > 0.5) {
        fill(0, 255, 0);
    } else {
        fill(255, 0, 0);
    }

    rect(10, 50, assistant.userMood * 180, 20);

    // 姿勢
    text('姿勢: ' + currentExpression, 10, 90);

    pop();
}

function drawVideoPreview() {
    // 小さなビデオプレビュー
    push();
    translate(550, 460);
    scale(0.3);
    image(video, 0, 0);
    pop();
}

// アバタークラス
class Avatar {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.mood = 'neutral';
        this.blinkTimer = 0;
        this.eyeOpen = true;
    }

    update(mood) {
        this.mood = mood;

        // まばたき
        this.blinkTimer++;
        if (this.blinkTimer > random(120, 180)) {
            this.eyeOpen = !this.eyeOpen;
            if (!this.eyeOpen) {
                this.blinkTimer = 0;
            } else {
                this.blinkTimer = 110;
            }
        }
    }

    display() {
        push();
        translate(this.x, this.y);

        // 顔の輪郭
        fill(250, 220, 180);
        noStroke();
        ellipse(0, 0, 100, 120);

        // 目
        if (this.eyeOpen) {
            fill(255);
            ellipse(-20, -10, 25, 30);
            ellipse(20, -10, 25, 30);

            fill(50, 100, 200);
            ellipse(-20, -10, 15, 15);
            ellipse(20, -10, 15, 15);

            fill(0);
            ellipse(-20, -10, 8, 8);
            ellipse(20, -10, 8, 8);
        } else {
            stroke(0);
            strokeWeight(2);
            line(-30, -10, -10, -10);
            line(10, -10, 30, -10);
        }

        // 口
        noFill();
        stroke(0);
        strokeWeight(2);

        if (this.mood === 'happy') {
            // 笑顔
            arc(0, 20, 40, 30, 0, PI);
        } else if (this.mood === 'concerned') {
            // 心配顔
            arc(0, 30, 40, 30, PI, TWO_PI);
        } else {
            // 通常
            line(-20, 25, 20, 25);
        }

        pop();
    }
}
```

## 7.4 プロジェクトの公開とデプロイ

### Webアプリケーションとしてのデプロイ

```javascript
// デプロイ用の完成版プロジェクト構造
/*
project/
├── index.html
├── style.css
├── sketch.js
├── models/
│   └── custom-model.json
├── assets/
│   ├── images/
│   └── sounds/
└── libraries/
    ├── p5.min.js
    └── ml5.min.js
*/

// index.html
`<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ml5.js AIアプリケーション</title>

    <!-- メタタグ -->
    <meta name="description" content="ml5.jsを使用したAIアプリケーション">
    <meta property="og:title" content="ml5.js AIアプリ">
    <meta property="og:description" content="機械学習を使ったインタラクティブアプリ">

    <!-- スタイル -->
    <link rel="stylesheet" href="style.css">

    <!-- ライブラリ -->
    <script src="https://cdn.jsdelivr.net/npm/p5@1.7.0/lib/p5.min.js"></script>
    <script src="https://unpkg.com/ml5@latest/dist/ml5.min.js"></script>
</head>
<body>
    <div id="app-container">
        <header>
            <h1>ml5.js AIアプリケーション</h1>
            <nav>
                <button id="startBtn">開始</button>
                <button id="stopBtn">停止</button>
                <button id="settingsBtn">設定</button>
            </nav>
        </header>

        <main id="sketch-container">
            <!-- p5.jsキャンバスがここに挿入される -->
        </main>

        <aside id="info-panel">
            <h2>情報</h2>
            <div id="status">準備中...</div>
            <div id="performance">
                <p>FPS: <span id="fps">0</span></p>
                <p>遅延: <span id="latency">0</span>ms</p>
            </div>
        </aside>

        <!-- 設定モーダル -->
        <div id="settings-modal" class="modal">
            <div class="modal-content">
                <h2>設定</h2>
                <label>
                    カメラ選択:
                    <select id="camera-select"></select>
                </label>
                <label>
                    品質:
                    <select id="quality-select">
                        <option value="low">低</option>
                        <option value="medium" selected>中</option>
                        <option value="high">高</option>
                    </select>
                </label>
                <button id="save-settings">保存</button>
            </div>
        </div>
    </div>

    <script src="sketch.js"></script>
</body>
</html>`;

// style.css
`* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

#app-container {
    background: white;
    border-radius: 10px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    overflow: hidden;
    max-width: 1200px;
    width: 90%;
}

header {
    background: #333;
    color: white;
    padding: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    font-size: 1.5rem;
}

nav button {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    margin-left: 0.5rem;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.3s;
}

nav button:hover {
    background: #764ba2;
}

main {
    display: flex;
    justify-content: center;
    padding: 2rem;
    min-height: 500px;
}

#info-panel {
    position: absolute;
    right: 20px;
    top: 100px;
    background: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 5px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
}

.modal-content {
    background: white;
    margin: 15% auto;
    padding: 2rem;
    width: 80%;
    max-width: 500px;
    border-radius: 10px;
}

@media (max-width: 768px) {
    #app-container {
        width: 100%;
        border-radius: 0;
    }

    #info-panel {
        position: static;
        margin-top: 1rem;
    }
}`;

// デプロイ設定
const deployConfig = {
    // Vercel設定
    vercel: {
        name: "ml5-ai-app",
        buildCommand: "npm run build",
        outputDirectory: "dist",
        framework: "vanilla"
    },

    // Netlify設定
    netlify: {
        build: {
            command: "npm run build",
            publish: "dist"
        },
        headers: {
            "/*": {
                "Access-Control-Allow-Origin": "*"
            }
        }
    },

    // GitHub Pages設定
    githubPages: {
        source: "main",
        folder: "/docs"
    }
};

// パフォーマンス最適化
class PerformanceOptimizer {
    constructor() {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
    }

    update() {
        this.frameCount++;
        const currentTime = performance.now();

        if (currentTime - this.lastTime > 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = currentTime;

            this.updateDisplay();
        }
    }

    updateDisplay() {
        const fpsElement = document.getElementById('fps');
        if (fpsElement) {
            fpsElement.textContent = this.fps;
        }
    }

    // モデルの遅延読み込み
    async lazyLoadModel(modelName) {
        return new Promise((resolve) => {
            if (modelName === 'poseNet') {
                ml5.poseNet(video, resolve);
            } else if (modelName === 'handpose') {
                ml5.handpose(video, resolve);
            }
        });
    }
}

// プログレッシブWebアプリ(PWA)設定
const pwaConfig = {
    manifest: {
        name: "ml5.js AIアプリ",
        short_name: "ml5AI",
        description: "機械学習を使ったWebアプリ",
        start_url: "/",
        display: "standalone",
        background_color: "#ffffff",
        theme_color: "#667eea",
        icons: [
            {
                src: "/icon-192.png",
                sizes: "192x192",
                type: "image/png"
            },
            {
                src: "/icon-512.png",
                sizes: "512x512",
                type: "image/png"
            }
        ]
    }
};

// サービスワーカー
const serviceWorker = `
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('v1').then((cache) => {
            return cache.addAll([
                '/',
                '/index.html',
                '/style.css',
                '/sketch.js',
                '/libraries/p5.min.js',
                '/libraries/ml5.min.js'
            ]);
        })
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            return response || fetch(event.request);
        })
    );
});`;
```

## 総合練習問題と解答

### 総合問題1
これまで学んだ技術を組み合わせて、「バーチャル楽器」アプリを作成してください。要件：
- HandPoseで手の動きを検出
- 手の位置で音程を制御
- ジェスチャーで楽器の種類を切り替え
- 演奏の録音・再生機能

### 総合問題2
「AIフィットネストレーナー」を作成してください。要件：
- PoseNetで運動フォームをチェック
- 音声コマンドで運動メニューを選択
- リアルタイムでフィードバック
- 運動記録の保存

### 総合問題3
「感情対応型学習アプリ」を作成してください。要件：
- ユーザーの表情や姿勢から集中度を判定
- 感情分析で学習者の状態を把握
- 状態に応じて問題の難易度を調整
- 学習進捗の可視化

## 解答（概要）

これらの総合問題は、本書で学んだすべての技術を統合する実践的なプロジェクトです。各章で学んだ以下の技術を組み合わせて実装します：

1. **バーチャル楽器**
   - 第5章のHandPose技術
   - Web Audio APIとの連携
   - 第2章のセットアップとデバッグ技術

2. **AIフィットネストレーナー**
   - 第4章のPoseNet技術
   - 第6章の音声認識
   - データの永続化（localStorage）

3. **感情対応型学習アプリ**
   - 第3章の画像分類
   - 第6章の感情分析
   - 適応的アルゴリズムの実装

## まとめ

第7章では、ml5.jsの様々な機能を組み合わせて実践的なアプリケーションを作成する方法を学びました。ARフィルター、ゲーム、チャットボットなど、実際に使用できるレベルのプロジェクトを通じて、機械学習をWebアプリケーションに統合する技術を習得しました。

本書を通じて学んだml5.jsの技術を活用して、あなた独自のAIアプリケーションを作成してみてください。機械学習の力を使って、より創造的でインタラクティブなWeb体験を生み出すことができるはずです。

Happy Coding with ml5.js! 🚀