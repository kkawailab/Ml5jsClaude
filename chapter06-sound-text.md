# 第6章: 音声認識とテキスト分析

## 6.1 SoundClassifierの基礎

### SoundClassifierとは

SoundClassifierは、音声や環境音を分類するためのml5.jsモデルです。Speech CommandsモデルやカスタムモデルFを使用して、音声コマンドや特定の音を認識できます。

### Speech Commandsモデル

```javascript
// Speech Commandsモデルの基本使用
let soundClassifier;
let resultLabel = '聞き取り中...';
let confidence = 0;

function setup() {
    createCanvas(640, 480);

    // オプション設定
    let options = {
        probabilityThreshold: 0.7  // 信頼度の閾値
    };

    // Speech Commands 18wモデルを読み込み
    // 18個の音声コマンドを認識可能
    soundClassifier = ml5.soundClassifier('SpeechCommands18w', options, modelReady);
}

function modelReady() {
    console.log('音声認識モデル準備完了');

    // 分類を開始
    soundClassifier.classify(gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // 結果を保存
    resultLabel = results[0].label;
    confidence = results[0].confidence;

    // 継続的に分類
    // （自動的に継続されるため、再度呼び出す必要はない）
}

function draw() {
    background(0);

    // 認識可能な単語リスト
    drawWordList();

    // 現在の認識結果を表示
    drawResult();

    // 音声レベルメーター
    drawSoundLevel();
}

function drawWordList() {
    // Speech Commands 18wで認識可能な単語
    const words = [
        'zero', 'one', 'two', 'three', 'four',
        'five', 'six', 'seven', 'eight', 'nine',
        'up', 'down', 'left', 'right', 'go',
        'stop', 'yes', 'no'
    ];

    fill(100);
    textAlign(LEFT);
    textSize(12);
    text('認識可能な単語:', 20, 30);

    // グリッドで表示
    for (let i = 0; i < words.length; i++) {
        let x = 20 + (i % 6) * 100;
        let y = 60 + Math.floor(i / 6) * 30;

        // 認識された単語をハイライト
        if (words[i] === resultLabel) {
            fill(0, 255, 0);
        } else {
            fill(150);
        }

        text(words[i], x, y);
    }
}

function drawResult() {
    // 結果表示エリア
    fill(255);
    textAlign(CENTER);
    textSize(48);
    text(resultLabel, width / 2, height / 2);

    // 信頼度バー
    fill(0, 255, 0);
    rect(width / 2 - 100, height / 2 + 30, confidence * 200, 20);

    fill(255);
    textSize(16);
    text(`信頼度: ${(confidence * 100).toFixed(1)}%`, width / 2, height / 2 + 80);
}

function drawSoundLevel() {
    // 簡易的な音声レベル表示
    push();
    translate(width - 100, height - 100);

    // 背景円
    fill(50);
    noStroke();
    ellipse(0, 0, 80);

    // アニメーション
    if (resultLabel !== '_background_noise_') {
        let pulseSize = 40 + sin(frameCount * 0.1) * 20;
        fill(0, 255, 0, 100);
        ellipse(0, 0, pulseSize);
    }

    fill(255);
    textAlign(CENTER);
    textSize(10);
    text('LISTENING', 0, 0);

    pop();
}
```

### カスタム音声分類器の作成

```javascript
// カスタム音声分類器の作成
let soundClassifier;
let soundModel = 'https://teachablemachine.withgoogle.com/models/[YOUR-MODEL-ID]/';
let label = '待機中';

function preload() {
    // Teachable Machineで作成したカスタムモデルを読み込み
    soundClassifier = ml5.soundClassifier(soundModel + 'model.json');
}

function setup() {
    createCanvas(640, 480);

    // 分類開始
    soundClassifier.classify(gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    // カスタムクラスの結果
    label = results[0].label;
}

function draw() {
    background(0);

    // 結果を表示
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(32);
    text(label, width / 2, height / 2);
}
```

## 6.2 音声コマンドの認識

### 音声コマンドでのアプリ制御

```javascript
// 音声コマンドでゲームを制御
let soundClassifier;
let player;
let obstacles = [];
let score = 0;
let lastCommand = '';
let commandConfidence = 0;

class Player {
    constructor() {
        this.x = width / 2;
        this.y = height / 2;
        this.size = 30;
        this.speed = 5;
        this.targetX = this.x;
        this.targetY = this.y;
    }

    update() {
        // スムーズな移動
        this.x = lerp(this.x, this.targetX, 0.1);
        this.y = lerp(this.y, this.targetY, 0.1);

        // 画面内に制限
        this.x = constrain(this.x, this.size, width - this.size);
        this.y = constrain(this.y, this.size, height - this.size);
    }

    executeCommand(command) {
        const moveDistance = 50;

        switch(command) {
            case 'up':
                this.targetY -= moveDistance;
                break;
            case 'down':
                this.targetY += moveDistance;
                break;
            case 'left':
                this.targetX -= moveDistance;
                break;
            case 'right':
                this.targetX += moveDistance;
                break;
            case 'stop':
                this.targetX = this.x;
                this.targetY = this.y;
                break;
            case 'go':
                // 前進（現在の方向に）
                let angle = atan2(this.targetY - this.y, this.targetX - this.x);
                this.targetX += cos(angle) * moveDistance;
                this.targetY += sin(angle) * moveDistance;
                break;
        }
    }

    display() {
        fill(0, 255, 0);
        noStroke();
        ellipse(this.x, this.y, this.size);

        // 方向インジケーター
        stroke(0, 255, 0);
        strokeWeight(2);
        let angle = atan2(this.targetY - this.y, this.targetX - this.x);
        line(this.x, this.y,
             this.x + cos(angle) * 20,
             this.y + sin(angle) * 20);
    }
}

function setup() {
    createCanvas(640, 480);

    player = new Player();

    // 音声認識を初期化
    let options = {
        probabilityThreshold: 0.85  // 高い閾値で誤認識を減らす
    };

    soundClassifier = ml5.soundClassifier('SpeechCommands18w', options, modelReady);
}

function modelReady() {
    console.log('音声コマンドゲーム準備完了');
    soundClassifier.classify(gotCommand);
}

function gotCommand(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    const command = results[0].label;
    const confidence = results[0].confidence;

    // 移動コマンドのみ処理
    const moveCommands = ['up', 'down', 'left', 'right', 'stop', 'go'];

    if (moveCommands.includes(command) && confidence > 0.85) {
        lastCommand = command;
        commandConfidence = confidence;
        player.executeCommand(command);

        // コマンド実行のフィードバック
        createCommandFeedback(command);
    }
}

function createCommandFeedback(command) {
    // ビジュアルフィードバック用のパーティクル
    for (let i = 0; i < 10; i++) {
        obstacles.push({
            x: player.x,
            y: player.y,
            vx: random(-5, 5),
            vy: random(-5, 5),
            life: 255,
            color: color(random(255), random(255), random(255))
        });
    }
}

function draw() {
    // 背景（フェードエフェクト）
    fill(0, 20);
    rect(0, 0, width, height);

    // グリッド
    drawGrid();

    // プレイヤーを更新・表示
    player.update();
    player.display();

    // パーティクルを更新・表示
    updateParticles();

    // UI表示
    drawUI();
}

function drawGrid() {
    stroke(50);
    strokeWeight(1);

    for (let i = 0; i < width; i += 40) {
        line(i, 0, i, height);
    }

    for (let i = 0; i < height; i += 40) {
        line(0, i, width, i);
    }
}

function updateParticles() {
    for (let i = obstacles.length - 1; i >= 0; i--) {
        let p = obstacles[i];

        // 更新
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 5;
        p.vx *= 0.98;
        p.vy *= 0.98;

        // 表示
        fill(red(p.color), green(p.color), blue(p.color), p.life);
        noStroke();
        ellipse(p.x, p.y, 10);

        // 削除
        if (p.life <= 0) {
            obstacles.splice(i, 1);
        }
    }
}

function drawUI() {
    // コマンド表示エリア
    fill(0, 200);
    rect(0, 0, width, 60);

    fill(255);
    textAlign(LEFT);
    textSize(16);
    text('音声コマンド: "up", "down", "left", "right", "stop", "go"', 10, 25);

    // 最後のコマンド
    if (lastCommand) {
        textSize(20);
        fill(0, 255, 0);
        text(`最後のコマンド: ${lastCommand} (${(commandConfidence * 100).toFixed(0)}%)`, 10, 50);
    }

    // プレイヤー位置
    fill(255);
    textAlign(RIGHT);
    textSize(12);
    text(`位置: (${Math.round(player.x)}, ${Math.round(player.y)})`, width - 10, 25);
}
```

### 音声パターン認識

```javascript
// 複数の音声パターンを認識
let soundClassifier;
let commandSequence = [];
let recognizedPattern = '';
let patternTimer = 0;

// パターン定義
const patterns = {
    'konami': ['up', 'up', 'down', 'down', 'left', 'right', 'left', 'right'],
    'circle': ['up', 'right', 'down', 'left'],
    'zigzag': ['left', 'right', 'left', 'right'],
    'emergency': ['stop', 'stop', 'stop']
};

function setup() {
    createCanvas(640, 480);

    soundClassifier = ml5.soundClassifier('SpeechCommands18w', modelReady);
}

function modelReady() {
    console.log('音声パターン認識準備完了');
    soundClassifier.classify(gotResult);
}

function gotResult(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    const command = results[0].label;
    const confidence = results[0].confidence;

    // 方向コマンドとストップコマンドのみ処理
    const validCommands = ['up', 'down', 'left', 'right', 'stop'];

    if (validCommands.includes(command) && confidence > 0.9) {
        addToSequence(command);
        checkPatterns();
    }
}

function addToSequence(command) {
    // タイマーリセット
    patternTimer = millis();

    // シーケンスに追加
    commandSequence.push(command);

    // シーケンスの長さを制限
    if (commandSequence.length > 10) {
        commandSequence.shift();
    }
}

function checkPatterns() {
    // 各パターンをチェック
    for (let patternName in patterns) {
        const pattern = patterns[patternName];

        if (matchesPattern(commandSequence, pattern)) {
            recognizedPattern = patternName;
            executePattern(patternName);
            commandSequence = [];  // シーケンスをクリア
            break;
        }
    }
}

function matchesPattern(sequence, pattern) {
    if (sequence.length < pattern.length) return false;

    // シーケンスの末尾がパターンと一致するか確認
    const start = sequence.length - pattern.length;
    for (let i = 0; i < pattern.length; i++) {
        if (sequence[start + i] !== pattern[i]) {
            return false;
        }
    }

    return true;
}

function executePattern(patternName) {
    console.log(`パターン認識: ${patternName}`);

    // パターンに応じたアクション
    switch(patternName) {
        case 'konami':
            // 特別なエフェクト
            createSpecialEffect();
            break;
        case 'circle':
            // 円を描く
            createCircleAnimation();
            break;
        case 'zigzag':
            // ジグザグエフェクト
            createZigzagEffect();
            break;
        case 'emergency':
            // 緊急停止
            background(255, 0, 0);
            break;
    }
}

function draw() {
    background(20);

    // タイムアウトチェック（3秒でシーケンスをリセット）
    if (millis() - patternTimer > 3000 && commandSequence.length > 0) {
        commandSequence = [];
    }

    // シーケンス表示
    drawSequence();

    // パターン表示
    drawPatterns();

    // 認識されたパターン
    if (recognizedPattern) {
        drawRecognizedPattern();
    }
}

function drawSequence() {
    fill(255);
    textAlign(LEFT);
    textSize(16);
    text('コマンドシーケンス:', 20, 30);

    // 現在のシーケンスを表示
    for (let i = 0; i < commandSequence.length; i++) {
        let x = 20 + i * 60;
        let y = 60;

        // 矢印で表示
        drawArrow(x, y, commandSequence[i]);
    }
}

function drawArrow(x, y, direction) {
    push();
    translate(x, y);

    stroke(0, 255, 0);
    strokeWeight(2);
    fill(0, 255, 0);

    switch(direction) {
        case 'up':
            line(0, 10, 0, -10);
            triangle(-5, -5, 5, -5, 0, -15);
            break;
        case 'down':
            line(0, -10, 0, 10);
            triangle(-5, 5, 5, 5, 0, 15);
            break;
        case 'left':
            line(10, 0, -10, 0);
            triangle(-5, -5, -5, 5, -15, 0);
            break;
        case 'right':
            line(-10, 0, 10, 0);
            triangle(5, -5, 5, 5, 15, 0);
            break;
        case 'stop':
            noFill();
            ellipse(0, 0, 20);
            line(-7, -7, 7, 7);
            break;
    }

    pop();
}

function drawPatterns() {
    fill(150);
    textAlign(LEFT);
    textSize(14);
    text('パターン一覧:', 20, 150);

    let y = 180;
    for (let patternName in patterns) {
        text(`${patternName}: ${patterns[patternName].join(', ')}`, 20, y);
        y += 25;
    }
}

function drawRecognizedPattern() {
    fill(255, 255, 0);
    textAlign(CENTER);
    textSize(48);
    text(recognizedPattern.toUpperCase(), width / 2, height / 2);

    // フェードアウト
    if (frameCount % 60 === 0) {
        recognizedPattern = '';
    }
}

function createSpecialEffect() {
    // レインボーエフェクト
    for (let i = 0; i < 100; i++) {
        fill(random(255), random(255), random(255), 100);
        let x = random(width);
        let y = random(height);
        let size = random(10, 50);
        ellipse(x, y, size);
    }
}

function createCircleAnimation() {
    // 円形アニメーション
    push();
    translate(width / 2, height / 2);
    noFill();

    for (let i = 0; i < 10; i++) {
        stroke(0, 255, 0, 255 - i * 20);
        strokeWeight(3);
        ellipse(0, 0, i * 30);
    }

    pop();
}

function createZigzagEffect() {
    // ジグザグライン
    stroke(255, 255, 0);
    strokeWeight(3);

    beginShape();
    for (let x = 0; x < width; x += 40) {
        let y = height / 2 + (x % 80 === 0 ? -50 : 50);
        vertex(x, y);
    }
    endShape();
}
```

## 6.3 Sentiment分析で感情を判定

### テキスト感情分析

```javascript
// Sentimentモデルで感情分析
let sentiment;
let sentimentResult;
let inputText;
let history = [];

function setup() {
    createCanvas(800, 600);

    // Sentimentモデルを初期化
    sentiment = ml5.sentiment('movieReviews', modelReady);

    // テキスト入力フィールド
    inputText = createInput('今日はとても良い天気です');
    inputText.size(400);
    inputText.position(200, height - 100);

    // 分析ボタン
    let analyzeButton = createButton('感情分析');
    analyzeButton.position(inputText.x + inputText.width + 10, height - 100);
    analyzeButton.mousePressed(analyzeSentiment);
}

function modelReady() {
    console.log('感情分析モデル準備完了');
    analyzeSentiment();
}

function analyzeSentiment() {
    const text = inputText.value();

    if (text.length > 0) {
        // 感情を予測
        const prediction = sentiment.predict(text);

        sentimentResult = {
            text: text,
            score: prediction.score,
            timestamp: new Date().toLocaleTimeString()
        };

        // 履歴に追加
        history.unshift(sentimentResult);
        if (history.length > 10) {
            history.pop();
        }
    }
}

function draw() {
    background(240);

    // タイトル
    fill(0);
    textAlign(CENTER);
    textSize(32);
    text('感情分析システム', width / 2, 50);

    if (sentimentResult) {
        // 現在の分析結果を表示
        drawCurrentResult();

        // 感情メーター
        drawSentimentMeter();

        // 履歴グラフ
        drawHistoryGraph();
    }

    // 使い方
    drawInstructions();
}

function drawCurrentResult() {
    push();
    translate(width / 2, 150);

    // テキスト表示
    fill(0);
    textAlign(CENTER);
    textSize(16);
    text('分析テキスト:', 0, -40);

    textSize(20);
    fill(50);
    text('"' + sentimentResult.text + '"', 0, -10);

    // スコア
    let emotionLabel;
    let emotionColor;

    if (sentimentResult.score > 0.7) {
        emotionLabel = 'ポジティブ';
        emotionColor = color(0, 200, 0);
    } else if (sentimentResult.score > 0.3) {
        emotionLabel = 'ニュートラル';
        emotionColor = color(200, 200, 0);
    } else {
        emotionLabel = 'ネガティブ';
        emotionColor = color(200, 0, 0);
    }

    fill(emotionColor);
    textSize(36);
    text(emotionLabel, 0, 40);

    textSize(16);
    fill(0);
    text(`スコア: ${sentimentResult.score.toFixed(3)}`, 0, 70);

    pop();
}

function drawSentimentMeter() {
    push();
    translate(width / 2, 300);

    // メーター背景
    let meterWidth = 400;
    let meterHeight = 30;

    // グラデーション背景
    for (let i = 0; i < meterWidth; i++) {
        let inter = map(i, 0, meterWidth, 0, 1);
        let c = lerpColor(color(200, 0, 0), color(0, 200, 0), inter);
        stroke(c);
        line(i - meterWidth / 2, 0, i - meterWidth / 2, meterHeight);
    }

    // 枠
    noFill();
    stroke(0);
    strokeWeight(2);
    rect(-meterWidth / 2, 0, meterWidth, meterHeight);

    // インジケーター
    let indicatorX = map(sentimentResult.score, 0, 1, -meterWidth / 2, meterWidth / 2);
    fill(255);
    stroke(0);
    strokeWeight(2);
    triangle(indicatorX - 10, meterHeight + 5,
             indicatorX + 10, meterHeight + 5,
             indicatorX, meterHeight);

    // ラベル
    fill(0);
    noStroke();
    textAlign(LEFT);
    textSize(12);
    text('ネガティブ', -meterWidth / 2, -5);
    textAlign(RIGHT);
    text('ポジティブ', meterWidth / 2, -5);
    textAlign(CENTER);
    text('ニュートラル', 0, -5);

    pop();
}

function drawHistoryGraph() {
    push();
    translate(50, 400);

    // グラフ背景
    fill(255);
    stroke(0);
    strokeWeight(1);
    rect(0, 0, width - 100, 150);

    // 軸
    stroke(0);
    strokeWeight(2);
    line(0, 75, width - 100, 75);  // 中央線（0.5）

    // グリッド
    stroke(200);
    strokeWeight(1);
    for (let i = 0; i < 5; i++) {
        let y = map(i, 0, 4, 0, 150);
        line(0, y, width - 100, y);
    }

    // データポイント
    if (history.length > 1) {
        noFill();
        strokeWeight(2);

        beginShape();
        for (let i = 0; i < history.length; i++) {
            let x = map(i, 0, 9, width - 150, 50);
            let y = map(history[i].score, 0, 1, 150, 0);

            // 色分け
            if (history[i].score > 0.7) {
                stroke(0, 200, 0);
            } else if (history[i].score > 0.3) {
                stroke(200, 200, 0);
            } else {
                stroke(200, 0, 0);
            }

            vertex(x, y);

            // データポイント
            fill(255);
            ellipse(x, y, 8);
            noFill();
        }
        endShape();
    }

    // ラベル
    fill(0);
    textAlign(RIGHT, CENTER);
    textSize(10);
    text('1.0', -10, 0);
    text('0.5', -10, 75);
    text('0.0', -10, 150);

    textAlign(CENTER);
    textSize(12);
    text('感情スコアの履歴', (width - 100) / 2, 170);

    pop();
}

function drawInstructions() {
    fill(100);
    textAlign(CENTER);
    textSize(14);
    text('テキストを入力して感情を分析します（ポジティブ/ネガティブ）', width / 2, height - 130);
}
```

## 6.4 Word2Vecで単語の関係を探る

### 単語の意味的関係を可視化

```javascript
// Word2Vecを使った単語の関係探索
let word2Vec;
let targetWord = '';
let similarWords = [];
let wordRelations = [];
let inputField;

function setup() {
    createCanvas(800, 600);

    // Word2Vecモデルを読み込み
    word2Vec = ml5.word2vec('data/wordvecs1000.json', modelReady);

    // 入力フィールド
    inputField = createInput('king');
    inputField.position(20, height - 40);
    inputField.size(150);

    // ボタン類
    createButtons();
}

function modelReady() {
    console.log('Word2Vec準備完了');
    findSimilarWords();
}

function createButtons() {
    // 類似語検索ボタン
    let similarBtn = createButton('類似語検索');
    similarBtn.position(180, height - 40);
    similarBtn.mousePressed(findSimilarWords);

    // 単語演算ボタン
    let mathBtn = createButton('単語演算');
    mathBtn.position(270, height - 40);
    mathBtn.mousePressed(wordMath);

    // 関係性マップボタン
    let mapBtn = createButton('関係マップ');
    mapBtn.position(350, height - 40);
    mapBtn.mousePressed(createRelationMap);
}

function findSimilarWords() {
    targetWord = inputField.value();

    if (targetWord && word2Vec) {
        // 類似語を検索（上位10個）
        word2Vec.nearest(targetWord, 10, (err, result) => {
            if (err) {
                console.error(err);
                similarWords = [];
            } else {
                similarWords = result;
                createWordNetwork();
            }
        });
    }
}

function wordMath() {
    // 単語の演算例：king - man + woman = ?
    const positive = ['king', 'woman'];
    const negative = ['man'];

    word2Vec.subtract(positive, negative, 5, (err, result) => {
        if (!err) {
            console.log('king - man + woman =', result);
            displayWordMathResult(result);
        }
    });
}

function createRelationMap() {
    // 複数の単語間の関係を可視化
    const words = ['king', 'queen', 'man', 'woman', 'boy', 'girl'];
    wordRelations = [];

    for (let i = 0; i < words.length; i++) {
        for (let j = i + 1; j < words.length; j++) {
            word2Vec.similarity(words[i], words[j], (err, similarity) => {
                if (!err) {
                    wordRelations.push({
                        word1: words[i],
                        word2: words[j],
                        similarity: similarity
                    });
                }
            });
        }
    }
}

function createWordNetwork() {
    // 単語ネットワークのノードを作成
    if (similarWords.length > 0) {
        // 中心の単語と関連語をネットワーク状に配置
        // アニメーション用の座標を計算
    }
}

function draw() {
    background(240);

    // タイトル
    fill(0);
    textAlign(CENTER);
    textSize(24);
    text('Word2Vec 単語関係探索', width / 2, 30);

    // 検索結果の表示
    if (targetWord && similarWords.length > 0) {
        drawWordCloud();
    }

    // 関係マップの表示
    if (wordRelations.length > 0) {
        drawRelationGraph();
    }
}

function drawWordCloud() {
    push();
    translate(width / 2, height / 2);

    // 中心の単語
    fill(0);
    textAlign(CENTER);
    textSize(36);
    text(targetWord, 0, 0);

    // 類似語を円形に配置
    for (let i = 0; i < similarWords.length; i++) {
        let angle = TWO_PI * i / similarWords.length;
        let radius = 150;

        // 類似度で距離を調整
        let similarity = similarWords[i].distance;
        radius = map(similarity, 0, 1, 200, 100);

        let x = cos(angle) * radius;
        let y = sin(angle) * radius;

        // 線で接続
        stroke(150, 150);
        strokeWeight(similarity * 3);
        line(0, 0, x, y);

        // 単語を表示
        noStroke();
        fill(0, map(similarity, 0, 1, 100, 255));
        textSize(map(similarity, 0, 1, 12, 24));
        text(similarWords[i].word, x, y);

        // 類似度を表示
        textSize(10);
        fill(100);
        text(similarity.toFixed(3), x, y + 15);
    }

    pop();
}

function drawRelationGraph() {
    push();
    translate(100, 400);

    // 関係性をグラフで表示
    let maxWidth = 600;
    let barHeight = 15;

    wordRelations.sort((a, b) => b.similarity - a.similarity);

    for (let i = 0; i < min(10, wordRelations.length); i++) {
        let rel = wordRelations[i];
        let barWidth = rel.similarity * maxWidth;

        // バー
        fill(0, 150, 255);
        rect(0, i * (barHeight + 5), barWidth, barHeight);

        // ラベル
        fill(0);
        textAlign(LEFT);
        textSize(12);
        text(`${rel.word1} ↔ ${rel.word2}`, 5, i * (barHeight + 5) + 12);

        textAlign(RIGHT);
        text(rel.similarity.toFixed(3), barWidth - 5, i * (barHeight + 5) + 12);
    }

    pop();
}

function displayWordMathResult(result) {
    // 単語演算の結果を表示
    console.log('単語演算結果:', result);
}
```

### 文章生成と自然言語処理

```javascript
// CharRNNを使った文章生成
let charRNN;
let textInput;
let generating = false;
let generatedText = '';
let temperature = 0.5;  // 生成の多様性

function setup() {
    createCanvas(800, 600);

    // CharRNNモデルを読み込み（例：シェイクスピアスタイル）
    charRNN = ml5.charRNN('models/shakespeare/', modelReady);

    // UI作成
    createUIElements();
}

function modelReady() {
    console.log('文章生成モデル準備完了');
}

function createUIElements() {
    // シードテキスト入力
    textInput = createInput('To be or not to be');
    textInput.position(20, height - 100);
    textInput.size(300);

    // 生成ボタン
    let generateBtn = createButton('文章生成');
    generateBtn.position(330, height - 100);
    generateBtn.mousePressed(generateText);

    // 温度スライダー
    let tempSlider = createSlider(0.1, 1.0, 0.5, 0.1);
    tempSlider.position(20, height - 60);
    tempSlider.input(() => {
        temperature = tempSlider.value();
    });
}

function generateText() {
    if (!generating && charRNN) {
        generating = true;
        generatedText = textInput.value();

        // 文章を生成
        let data = {
            seed: generatedText,
            temperature: temperature,
            length: 200
        };

        charRNN.generate(data, gotResult);
    }
}

function gotResult(err, result) {
    if (err) {
        console.error(err);
    } else {
        generatedText = result.sample;
    }

    generating = false;
}

function draw() {
    background(240);

    // タイトル
    fill(0);
    textAlign(CENTER);
    textSize(24);
    text('AI文章生成システム', width / 2, 30);

    // 生成された文章を表示
    if (generatedText) {
        displayGeneratedText();
    }

    // 設定表示
    displaySettings();

    // 生成中インジケーター
    if (generating) {
        drawLoadingIndicator();
    }
}

function displayGeneratedText() {
    push();

    // テキストエリア
    fill(255);
    stroke(0);
    strokeWeight(1);
    rect(50, 100, width - 100, 300);

    // 生成されたテキスト
    fill(0);
    textAlign(LEFT);
    textSize(14);
    textWrap(WORD);

    let lines = generatedText.match(/.{1,70}/g) || [];
    for (let i = 0; i < lines.length && i < 15; i++) {
        text(lines[i], 60, 120 + i * 20);
    }

    pop();
}

function displaySettings() {
    fill(0);
    textAlign(LEFT);
    textSize(14);
    text(`温度: ${temperature.toFixed(1)} (創造性)`, 150, height - 45);

    // 温度の説明
    textSize(12);
    fill(100);
    text('低: 予測可能', 20, height - 25);
    text('高: 創造的', 200, height - 25);
}

function drawLoadingIndicator() {
    push();
    translate(width / 2, height / 2);

    noFill();
    stroke(0, 150, 255);
    strokeWeight(3);

    let angle = frameCount * 0.1;
    arc(0, 0, 50, 50, angle, angle + PI);

    fill(0);
    noStroke();
    textAlign(CENTER);
    textSize(14);
    text('生成中...', 0, 40);

    pop();
}
```

## 練習問題

### 問題1: 音声認識の理解
Speech Commands 18wモデルで認識できる18個の単語をすべて挙げ、それらを使ってできるアプリケーションを1つ提案してください。

### 問題2: 感情分析アプリ
リアルタイムでユーザーの入力を監視し、ネガティブな感情が検出されたら励ましのメッセージを表示するアプリを作成してください。

### 問題3: 音声計算機
音声コマンド（数字と"plus", "minus"など）を使って計算ができる音声計算機を作成してください。

### 問題4: インタラクティブストーリー
Word2VecとCharRNNを組み合わせて、ユーザーが選んだキーワードに基づいてストーリーを生成するアプリを作成してください。

## 解答

### 問題1の解答

**Speech Commands 18wの18個の単語：**
1. zero, one, two, three, four, five, six, seven, eight, nine（数字0-9）
2. up, down, left, right（方向）
3. go, stop（動作）
4. yes, no（確認）

**提案アプリケーション：「音声制御エレベーター」**
- 数字で階を指定
- up/downで上下移動
- goで確定、stopで緊急停止
- yes/noで確認応答

### 問題2の解答

```javascript
// リアルタイム感情モニタリングアプリ
let sentiment;
let inputField;
let currentMood = 0.5;
let moodHistory = [];
let encouragementMessages = [
    '大丈夫、きっとうまくいきます！',
    '深呼吸して、リラックスしましょう',
    '小さな成功を積み重ねていきましょう',
    '今日も一日お疲れ様でした',
    'あなたは素晴らしい人です！'
];
let currentMessage = '';
let messageTimer = 0;

function setup() {
    createCanvas(800, 600);

    // 感情分析モデル
    sentiment = ml5.sentiment('movieReviews', modelReady);

    // 入力フィールド
    inputField = createElement('textarea');
    inputField.position(50, 100);
    inputField.size(700, 200);
    inputField.attribute('placeholder', '今の気持ちを書いてください...');

    // リアルタイム監視
    inputField.input(analyzeInput);
}

function modelReady() {
    console.log('感情サポートシステム準備完了');
}

function analyzeInput() {
    const text = inputField.value();

    if (text.length > 10) {
        const prediction = sentiment.predict(text);
        currentMood = prediction.score;

        // 履歴に追加
        moodHistory.push({
            score: currentMood,
            time: millis()
        });

        // 古いデータを削除
        if (moodHistory.length > 100) {
            moodHistory.shift();
        }

        // ネガティブな感情を検出
        if (currentMood < 0.3) {
            showEncouragement();
        }
    }
}

function showEncouragement() {
    // ランダムなメッセージを選択
    currentMessage = random(encouragementMessages);
    messageTimer = millis();
}

function draw() {
    // 背景（気分に応じて色を変更）
    let bgColor = lerpColor(
        color(100, 100, 200),  // ネガティブ：青
        color(255, 200, 100),  // ポジティブ：暖色
        currentMood
    );
    background(bgColor);

    // タイトル
    fill(255);
    textAlign(CENTER);
    textSize(32);
    text('感情サポートシステム', width / 2, 50);

    // 気分グラフ
    drawMoodGraph();

    // 気分メーター
    drawMoodMeter();

    // 励ましメッセージ
    if (currentMessage && millis() - messageTimer < 5000) {
        drawEncouragement();
    }

    // アドバイス
    drawAdvice();
}

function drawMoodGraph() {
    push();
    translate(50, 350);

    // グラフ背景
    fill(255, 255, 255, 100);
    rect(0, 0, 700, 150);

    // 基準線
    stroke(255);
    strokeWeight(1);
    line(0, 75, 700, 75);

    // 気分の推移
    noFill();
    strokeWeight(3);
    beginShape();

    for (let i = 0; i < moodHistory.length; i++) {
        let x = map(i, 0, 99, 0, 700);
        let y = map(moodHistory[i].score, 0, 1, 150, 0);

        // 色分け
        if (moodHistory[i].score < 0.3) {
            stroke(255, 100, 100);
        } else if (moodHistory[i].score < 0.7) {
            stroke(255, 255, 100);
        } else {
            stroke(100, 255, 100);
        }

        vertex(x, y);
    }

    endShape();

    // ラベル
    fill(255);
    noStroke();
    textAlign(LEFT);
    textSize(12);
    text('気分の推移', 10, -10);

    pop();
}

function drawMoodMeter() {
    push();
    translate(width / 2, 320);

    // メーター
    let meterAngle = map(currentMood, 0, 1, -PI, 0);

    strokeWeight(20);
    noFill();

    // 背景アーク
    stroke(100);
    arc(0, 0, 150, 150, -PI, 0);

    // 現在の気分
    if (currentMood < 0.3) {
        stroke(255, 100, 100);
    } else if (currentMood < 0.7) {
        stroke(255, 255, 100);
    } else {
        stroke(100, 255, 100);
    }

    arc(0, 0, 150, 150, -PI, meterAngle);

    // 顔文字
    textAlign(CENTER);
    textSize(48);
    noStroke();
    fill(255);

    if (currentMood < 0.3) {
        text('😢', 0, 10);
    } else if (currentMood < 0.7) {
        text('😐', 0, 10);
    } else {
        text('😊', 0, 10);
    }

    pop();
}

function drawEncouragement() {
    push();

    // メッセージボックス
    fill(255, 255, 100, 200);
    stroke(255);
    strokeWeight(2);
    rect(width / 2 - 250, height - 150, 500, 80, 10);

    // メッセージ
    fill(0);
    textAlign(CENTER);
    textSize(20);
    text('💝 ' + currentMessage, width / 2, height - 100);

    pop();
}

function drawAdvice() {
    fill(255);
    textAlign(CENTER);
    textSize(14);

    if (currentMood < 0.3) {
        text('休憩を取ることも大切です', width / 2, height - 30);
    } else if (currentMood > 0.7) {
        text('素晴らしい気分ですね！', width / 2, height - 30);
    }
}
```

### 問題3の解答

```javascript
// 音声計算機
let soundClassifier;
let calculator = {
    display: '0',
    memory: 0,
    operation: null,
    waitingForNumber: false
};
let lastCommand = '';
let commandHistory = [];

function setup() {
    createCanvas(400, 600);

    // 音声認識を初期化
    let options = {
        probabilityThreshold: 0.95
    };

    soundClassifier = ml5.soundClassifier('SpeechCommands18w', options, modelReady);
}

function modelReady() {
    console.log('音声計算機準備完了');
    soundClassifier.classify(gotCommand);
}

function gotCommand(error, results) {
    if (error) {
        console.error(error);
        return;
    }

    const command = results[0].label;
    const confidence = results[0].confidence;

    // 数字とコマンドを処理
    processCommand(command);

    // 履歴に追加
    commandHistory.push({
        command: command,
        time: millis()
    });

    if (commandHistory.length > 5) {
        commandHistory.shift();
    }

    lastCommand = command;
}

function processCommand(command) {
    // 数字の処理
    const numbers = ['zero', 'one', 'two', 'three', 'four',
                    'five', 'six', 'seven', 'eight', 'nine'];

    if (numbers.includes(command)) {
        const digit = numbers.indexOf(command);

        if (calculator.waitingForNumber || calculator.display === '0') {
            calculator.display = digit.toString();
            calculator.waitingForNumber = false;
        } else {
            calculator.display += digit.toString();
        }
    }

    // 演算コマンドの処理
    switch(command) {
        case 'up':  // プラス
            performOperation('+');
            break;
        case 'down':  // マイナス
            performOperation('-');
            break;
        case 'left':  // クリア
            calculator.display = '0';
            calculator.memory = 0;
            calculator.operation = null;
            break;
        case 'right':  // イコール
            calculate();
            break;
        case 'stop':  // クリアエントリ
            calculator.display = '0';
            break;
    }
}

function performOperation(op) {
    if (calculator.operation && !calculator.waitingForNumber) {
        calculate();
    }

    calculator.memory = parseFloat(calculator.display);
    calculator.operation = op;
    calculator.waitingForNumber = true;
}

function calculate() {
    if (calculator.operation) {
        const current = parseFloat(calculator.display);

        switch(calculator.operation) {
            case '+':
                calculator.display = (calculator.memory + current).toString();
                break;
            case '-':
                calculator.display = (calculator.memory - current).toString();
                break;
        }

        calculator.operation = null;
        calculator.waitingForNumber = true;
    }
}

function draw() {
    background(50);

    // 計算機の画面
    drawCalculatorDisplay();

    // ボタンレイアウト
    drawButtons();

    // 音声コマンド履歴
    drawCommandHistory();

    // 説明
    drawInstructions();
}

function drawCalculatorDisplay() {
    // ディスプレイ背景
    fill(200, 255, 200);
    rect(20, 20, width - 40, 80);

    // 表示値
    fill(0);
    textAlign(RIGHT);
    textSize(36);
    text(calculator.display, width - 30, 70);

    // 演算子表示
    if (calculator.operation) {
        textAlign(LEFT);
        textSize(24);
        text(calculator.operation, 30, 70);
    }
}

function drawButtons() {
    // 数字ボタン（視覚的な表示のみ）
    const buttonSize = 60;
    const startX = 50;
    const startY = 150;

    // 数字配置
    const layout = [
        ['7', '8', '9'],
        ['4', '5', '6'],
        ['1', '2', '3'],
        ['C', '0', '=']
    ];

    for (let row = 0; row < layout.length; row++) {
        for (let col = 0; col < layout[row].length; col++) {
            let x = startX + col * (buttonSize + 20);
            let y = startY + row * (buttonSize + 20);

            // ボタン背景
            fill(100);
            if (lastCommand === convertToCommand(layout[row][col])) {
                fill(0, 255, 0);
            }
            rect(x, y, buttonSize, buttonSize, 10);

            // ボタンテキスト
            fill(255);
            textAlign(CENTER, CENTER);
            textSize(24);
            text(layout[row][col], x + buttonSize / 2, y + buttonSize / 2);
        }
    }

    // 演算ボタン
    const operations = ['+', '-'];
    for (let i = 0; i < operations.length; i++) {
        let x = startX + 250;
        let y = startY + i * (buttonSize + 20);

        fill(150, 150, 255);
        rect(x, y, buttonSize, buttonSize, 10);

        fill(255);
        textAlign(CENTER, CENTER);
        text(operations[i], x + buttonSize / 2, y + buttonSize / 2);
    }
}

function convertToCommand(button) {
    const mapping = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        'C': 'left', '=': 'right'
    };
    return mapping[button] || button;
}

function drawCommandHistory() {
    fill(255);
    textAlign(LEFT);
    textSize(12);
    text('音声コマンド履歴:', 20, 450);

    for (let i = 0; i < commandHistory.length; i++) {
        let alpha = map(i, 0, commandHistory.length - 1, 100, 255);
        fill(255, alpha);
        text(commandHistory[i].command, 20, 470 + i * 20);
    }
}

function drawInstructions() {
    fill(200);
    textAlign(CENTER);
    textSize(14);
    text('音声コマンド: 数字(zero-nine), up(+), down(-), left(C), right(=)',
         width / 2, height - 20);
}
```

### 問題4の解答

```javascript
// インタラクティブストーリー生成
let word2vec;
let charRNN;
let storyElements = {
    character: '',
    setting: '',
    action: '',
    emotion: ''
};
let generatedStory = '';
let isGenerating = false;
let relatedWords = {};

function setup() {
    createCanvas(800, 700);

    // モデルを読み込み
    word2vec = ml5.word2vec('data/wordvecs.json', word2vecReady);
    charRNN = ml5.charRNN('models/shakespeare/', charRNNReady);

    // UI作成
    createStoryUI();
}

function word2vecReady() {
    console.log('Word2Vec準備完了');
}

function charRNNReady() {
    console.log('CharRNN準備完了');
}

function createStoryUI() {
    // キャラクター入力
    let charInput = createInput('');
    charInput.position(150, 100);
    charInput.attribute('placeholder', 'キャラクター');
    charInput.input(() => {
        storyElements.character = charInput.value();
        findRelatedWords('character', storyElements.character);
    });

    // 設定入力
    let settingInput = createInput('');
    settingInput.position(150, 140);
    settingInput.attribute('placeholder', '場所・設定');
    settingInput.input(() => {
        storyElements.setting = settingInput.value();
        findRelatedWords('setting', storyElements.setting);
    });

    // アクション入力
    let actionInput = createInput('');
    actionInput.position(150, 180);
    actionInput.attribute('placeholder', 'アクション');
    actionInput.input(() => {
        storyElements.action = actionInput.value();
        findRelatedWords('action', storyElements.action);
    });

    // 感情入力
    let emotionInput = createInput('');
    emotionInput.position(150, 220);
    emotionInput.attribute('placeholder', '感情・ムード');
    emotionInput.input(() => {
        storyElements.emotion = emotionInput.value();
        findRelatedWords('emotion', storyElements.emotion);
    });

    // 生成ボタン
    let generateBtn = createButton('ストーリー生成');
    generateBtn.position(350, 260);
    generateBtn.mousePressed(generateStory);
}

function findRelatedWords(category, word) {
    if (word && word2vec) {
        word2vec.nearest(word, 5, (err, result) => {
            if (!err) {
                relatedWords[category] = result;
            }
        });
    }
}

function generateStory() {
    if (isGenerating) return;

    isGenerating = true;

    // ストーリーのプロンプトを作成
    let prompt = createStoryPrompt();

    // CharRNNで生成
    let data = {
        seed: prompt,
        temperature: 0.75,
        length: 500
    };

    charRNN.generate(data, (err, result) => {
        if (!err) {
            generatedStory = prompt + result.sample;
        }
        isGenerating = false;
    });
}

function createStoryPrompt() {
    let prompt = '';

    if (storyElements.character) {
        prompt += `The ${storyElements.character} `;
    }

    if (storyElements.setting) {
        prompt += `in the ${storyElements.setting} `;
    }

    if (storyElements.action) {
        prompt += `${storyElements.action} `;
    }

    if (storyElements.emotion) {
        prompt += `with ${storyElements.emotion}. `;
    }

    return prompt || 'Once upon a time, ';
}

function draw() {
    background(245);

    // タイトル
    fill(0);
    textAlign(CENTER);
    textSize(28);
    text('AIストーリージェネレーター', width / 2, 40);

    // 入力ラベル
    drawInputLabels();

    // 関連語の表示
    drawRelatedWords();

    // 生成されたストーリー
    if (generatedStory) {
        drawGeneratedStory();
    }

    // 生成中インジケーター
    if (isGenerating) {
        drawLoadingAnimation();
    }
}

function drawInputLabels() {
    fill(0);
    textAlign(RIGHT);
    textSize(14);

    text('キャラクター:', 140, 110);
    text('場所・設定:', 140, 150);
    text('アクション:', 140, 190);
    text('感情・ムード:', 140, 230);
}

function drawRelatedWords() {
    push();
    translate(500, 100);

    fill(0);
    textAlign(LEFT);
    textSize(12);
    text('関連語:', 0, 0);

    let y = 20;
    for (let category in relatedWords) {
        if (relatedWords[category]) {
            fill(100);
            text(category + ':', 0, y);

            for (let i = 0; i < relatedWords[category].length; i++) {
                let word = relatedWords[category][i];
                fill(0, 100, 200);
                text(word.word, 20, y + (i + 1) * 15);
            }

            y += 100;
        }
    }

    pop();
}

function drawGeneratedStory() {
    push();
    translate(50, 300);

    // ストーリーボックス
    fill(255);
    stroke(0);
    strokeWeight(1);
    rect(0, 0, 700, 350);

    // ストーリーテキスト
    fill(0);
    noStroke();
    textAlign(LEFT);
    textSize(14);
    textWrap(WORD);

    // テキストを行ごとに分割
    let lines = generatedStory.match(/.{1,80}/g) || [];
    for (let i = 0; i < lines.length && i < 20; i++) {
        text(lines[i], 10, 20 + i * 18);
    }

    pop();
}

function drawLoadingAnimation() {
    push();
    translate(width / 2, height / 2);

    for (let i = 0; i < 8; i++) {
        let angle = TWO_PI * i / 8 + frameCount * 0.05;
        let x = cos(angle) * 30;
        let y = sin(angle) * 30;

        fill(0, 150, 255, 150 - i * 15);
        noStroke();
        ellipse(x, y, 10);
    }

    fill(0);
    textAlign(CENTER);
    text('ストーリー生成中...', 0, 60);

    pop();
}
```

## まとめ

第6章では、ml5.jsの音声認識とテキスト分析機能について学習しました。SoundClassifierによる音声コマンド認識、Sentimentモデルによる感情分析、Word2Vecによる単語の意味的関係の探索、そしてCharRNNによる文章生成まで、自然言語処理の幅広い応用を習得しました。

次章では、これまで学んだ技術を組み合わせて、実践的なプロジェクトを作成します。