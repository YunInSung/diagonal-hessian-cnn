import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from myModel_2opt import MyModel
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel
import pandas as pd
import time
import gc
import argparse
datasets = ['cifar10', 'cifar100']
parser = argparse.ArgumentParser(description="Optimizer 비교 실험 스크립트")
parser.add_argument(
    '--datasets',
    nargs='+',
    default=['cifar10', 'cifar100'],
    choices=datasets,
    help='실험할 데이터셋 목록 (cifar10, cifar100)'
)
args = parser.parse_args()

# ----------------- GPU 메모리 그로스 설정 -----------------
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ----------------- 하이퍼파라미터 -----------------
SEEDS           = list(range(20))   # 20회 반복
EPOCHS          = 50
BATCH_SIZE      = 64
LEARNING_RATE   = 1e-3
DROPOUT_RATE    = 0.4
L2_REG          = 1e-2                     # optional

# ----------------- 데이터셋 목록 -----------------
DATASETS = args.datasets

# ----------------- 모델 빌더 -----------------
def build_base(num_classes):
    inputs = tf.keras.Input(shape=(32,32,3))
    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        return x

    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    x = layers.Flatten()(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Dropout(0.5)(x)

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
def train_pair(x_train, y_train, x_val, y_val, num_classes, seed):
    # 재현성
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # 1) 참조 모델로 초기 가중치 하나 생성
    ref_inp, ref_out = build_base(num_classes)
    ref_model = Model(ref_inp, ref_out)
    base_weights = ref_model.get_weights()
    # ref_model은 이후 안 씀

    # 2) Baseline(AdamW) 만들고 동일 가중치 세팅
    in_b, out_b = build_base(num_classes)
    baseline = Model(in_b, out_b)
    baseline.set_weights(base_weights)
    opt_b = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=5e-4)
    baseline.compile(optimizer=opt_b, loss='categorical_crossentropy', metrics=['accuracy'])

    # 3) Custom(MyModel) 만들고 동일 가중치 세팅 (WD 미적용, optimizer는 형식상 Adam)
    in_c, out_c = build_base(num_classes)
    custom = MyModel(inputs=in_c, outputs=out_c)   # 내부에서 직접 업데이트한다면 외부 opt는 사실상 미사용
    custom.set_weights(base_weights)
    opt_c = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    custom.compile(optimizer=opt_c, loss='categorical_crossentropy', metrics=['accuracy'])

    # 4) 같은 셔플 순서를 쓰도록, seed 고정된 tf.data 만들기
    def make_train_ds():
        return (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(buffer_size=len(x_train), seed=seed, reshuffle_each_iteration=True)
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    train_ds_b = make_train_ds()
    train_ds_c = make_train_ds()
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

    # 5) 학습 + 시간
    t0 = time.time()
    baseline.fit(train_ds_b, epochs=EPOCHS, verbose=0, validation_data=val_ds)
    t_base = time.time() - t0

    t0 = time.time()
    custom.fit(train_ds_c, epochs=EPOCHS, verbose=0, validation_data=val_ds)
    t_cust = time.time() - t0

    # 6) 메트릭
    y_true = np.argmax(y_val, axis=1)

    pred_b = baseline.predict(x_val, batch_size=BATCH_SIZE, verbose=0)
    f1_b = f1_score(y_true, np.argmax(pred_b, axis=1), average='macro')
    val_loss_b, val_acc_b = baseline.evaluate(x_val, y_val, verbose=0)[:2]

    pred_c = custom.predict(x_val, batch_size=BATCH_SIZE, verbose=0)
    f1_c = f1_score(y_true, np.argmax(pred_c, axis=1), average='macro')
    val_loss_c, val_acc_c = custom.evaluate(x_val, y_val, verbose=0)[:2]

    # 정리
    res_baseline = (val_loss_b, val_acc_b, f1_b, t_base)
    res_custom   = (val_loss_c, val_acc_c, f1_c, t_cust)

    tf.keras.backend.clear_session()
    del baseline, custom, ref_model
    gc.collect()

    return res_baseline, res_custom

# ----------------- 전체 파이프라인 -----------------
for ds in DATASETS:
    print(f"\n>> Dataset: {ds}")
    (x_train, y_train), (x_val, y_val), num_classes = load_data(ds)

    if ds == 'cifar10':
        EPOCHS = 50
    elif ds == 'cifar100' :
        EPOCHS = 60

    metrics_baseline = []
    metrics_custom = []
    for seed in SEEDS:
        res_b, res_c = train_pair(x_train, y_train, x_val, y_val, num_classes, seed)
        metrics_baseline.append(res_b)
        metrics_custom.append(res_c)
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
    df_met.to_csv(f'{ds}_adam_optimizer_metrics.csv', index=False)

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
    pd.DataFrame(sum_rows).to_csv(f'{ds}_adam_optimizer_summary.csv', index=False)

    # 4) paired t-test 결과 저장
    stat_rows = []
    for i, metric in enumerate(['val_loss','val_acc','f1','train_time']):
        t, p = ttest_rel(m_cu[:,i], m_ad[:,i])
        stat_rows.append({'dataset': ds, 'metric': metric, 't_stat': t, 'p_value': p})
    pd.DataFrame(stat_rows).to_csv(f'{ds}_adam_optimizer_ttest.csv', index=False)

    print(f"✅ {ds}: metrics/summary/ttest CSVs saved\n")