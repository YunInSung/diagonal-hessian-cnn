import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from myModel_2opt import MyModel
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel
import pandas as pd
import time
import gc

# ----------------- GPU 메모리 그로스 설정 -----------------
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ----------------- 하이퍼파라미터 -----------------
SEEDS           = list(range(20))   # 20회 반복
EPOCHS          = 25
BATCH_SIZE      = 64
DROPOUT_RATE    = 0.2
L2_REG          = 1e-2                     # optional

# ----------------- 데이터셋 목록 -----------------
DATASETS = ['cifar10', 'cifar100']

# ----------------- 모델 빌더 -----------------
def build_base(num_classes):
    inputs = tf.keras.Input(shape=(32,32,3))
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x); x = layers.Dropout(DROPOUT_RATE)(x)
        return x
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return inputs, outputs

# ----------------- 데이터 로드 함수 -----------------
def load_data(name):
    if name == 'cifar10':
        (xt, yt), (xv, yv) = tf.keras.datasets.cifar10.load_data()
        xt = xt.astype('float32') / 255.0
        xv = xv.astype('float32') / 255.0
        num_classes = 10

    elif name == 'cifar100':
        (xt, yt), (xv, yv) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        xt = xt.astype('float32') / 255.0
        xv = xv.astype('float32') / 255.0
        num_classes = 100

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # one-hot 인코딩
    yt_ohe = tf.keras.utils.to_categorical(yt, num_classes)
    yv_ohe = tf.keras.utils.to_categorical(yv, num_classes)

    return (xt, yt_ohe), (xv, yv_ohe), num_classes

# ----------------- 1회 학습 -----------------
def train_one(x_train, y_train, x_val, y_val, num_classes, seed, model_class='baseline'):
    # 재현성 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    sgd_opt = tf.keras.optimizers.SGD(
        learning_rate=0.01,   # 학습률
        momentum=0.9,         # 모멘텀 계수
        nesterov=True        # Nesterov 모멘텀 사용 여부 (필요 시 True)
    )
    inputs, outputs = build_base(num_classes)
    if model_class == 'baseline':
        net = Model(inputs, outputs)
    else:  # 'custom'
        net = MyModel(inputs=inputs, outputs=outputs)  # 내부에서 opt 관리

    net.compile(optimizer=sgd_opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    start = time.time()
    net.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=0,
            validation_data=(x_val, y_val))
    train_time = time.time() - start

    # 메트릭 계산
    val_pred = net.predict(x_val, verbose=0)
    y_true = np.argmax(y_val, axis=1)
    f1 = f1_score(y_true, np.argmax(val_pred, axis=1), average='macro')
    val_loss, val_acc = net.evaluate(x_val, y_val, verbose=0)[:2]
    tf.keras.backend.clear_session()
    del net
    gc.collect()
    return val_loss, val_acc, f1, train_time

# ----------------- 전체 파이프라인 -----------------
for ds in DATASETS:
    print(f"\n>> Dataset: {ds}")
    (x_train, y_train), (x_val, y_val), num_classes = load_data(ds)

    metrics_baseline = []
    metrics_custom = []
    for seed in SEEDS:
        metrics_baseline.append(train_one(x_train, y_train, x_val, y_val, num_classes, seed, 'baseline'))
        metrics_custom.append(train_one(x_train, y_train, x_val, y_val, num_classes, seed, 'custom'))
        # 메모리 비우기
        tf.keras.backend.clear_session()
        gc.collect()

    m_ad = np.array(metrics_baseline)
    m_cu = np.array(metrics_custom)

# ----------------- 통계 요약 -----------------

    # 요약 출력
    def summ(name, arr):
        print(f"{name:6} loss={arr[:,0].mean():.4f}±{arr[:,0].std():.4f}, "
                f"acc={arr[:,1].mean():.4f}±{arr[:,1].std():.4f}, "
                f"f1={arr[:,2].mean():.4f}±{arr[:,2].std():.4f}, "
                f"time={arr[:,3].mean():.1f}s±{arr[:,3].std():.1f}s")
    summ("Baseline", m_ad)
    summ("Custom", m_cu)

# paired t-test (양측)
    for i, metric in enumerate(['val_loss','val_acc','f1','train_time']):
        t, p = ttest_rel(m_cu[:,i], m_ad[:,i])
        print(f"  t-test {metric:<10} t={t:.3f}, p={p:.4f}")

# 1) seed별 메트릭 DataFrame 생성
    rows = []
    for idx, seed in enumerate(SEEDS):
        for name, arr in [('Baseline', m_ad[idx]), ('Custom', m_cu[idx])]:
            rows.append({
                'dataset': ds,
                'seed': seed,
                'optimizer': name,
                'val_loss': arr[0],
                'val_acc': arr[1],
                'f1': arr[2],
                'train_time': arr[3]
            })
    df_met = pd.DataFrame(rows)
    df_met.to_csv(f'{ds}_sgd_optimizer_metrics.csv', index=False)

    sum_rows = []
    for opt, grp in df_met.groupby('optimizer'):
        for metric in ['val_loss','val_acc','f1','train_time']:
            sum_rows.append({
                'dataset': ds,
                'optimizer': opt,
                'metric': metric,
                'mean': grp[metric].mean(),
                'std': grp[metric].std()
            })
    pd.DataFrame(sum_rows).to_csv(f'{ds}_sgd_optimizer_summary.csv', index=False)

    # 4) paired t-test 결과 저장
    stat_rows = []
    for i, metric in enumerate(['val_loss','val_acc','f1','train_time']):
        t, p = ttest_rel(m_cu[:,i], m_ad[:,i])
        stat_rows.append({'dataset': ds, 'metric': metric, 't_stat': t, 'p_value': p})
    pd.DataFrame(stat_rows).to_csv(f'{ds}_sgd_optimizer_ttest.csv', index=False)

    print(f"✅ {ds}: metrics/summary/ttest CSVs saved\n")