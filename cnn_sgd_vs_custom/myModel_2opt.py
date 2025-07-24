import tensorflow as tf
from tensorflow.keras import Model

class MyModel(Model):
    def __init__(self, inputs, outputs,
                 beta1=0.9, beta2=0.999,
                 learning_rate=7.5e-5, epsilon=1e-7,
                 **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        # 하이퍼파라미터
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr    = learning_rate
        self.epsilon = epsilon

        # velocities, moments 초기화 플래그
        self._slots_initialized = False

        # (기존 diff/square 계수 계산)
        self.diff = 0.3
        self.square = 5
        self.coef_H = self.square * (self.diff) ** (self.square - 1)
        self.coef_outerDp = self.square * (self.square - 1) * (self.diff) ** (self.square - 2)

    def _init_slots(self):
        # m, v 두 슬롯을 각각 trainable_variables 수만큼 생성
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False, name=f"m_{i}")
                  for i, v in enumerate(self.trainable_variables)]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False, name=f"v_{i}")
                  for i, v in enumerate(self.trainable_variables)]
        self._slots_initialized = True

    def train_step(self, data):
        if not self._slots_initialized:
            self._init_slots()

        x, y = data
        vars_ = self.trainable_variables         # 가독성용 별칭

        # ──────────────────────────────────────────────
        # 1) 1차 gradient + 2차 diagonal Hessian
        # ──────────────────────────────────────────────
        with tf.GradientTape(persistent=True) as t2:   # 2차 테이프
            with tf.GradientTape() as t1:              # 1차 테이프
                y_pred = self(x, training=True)
                loss   = self.compiled_loss(y, y_pred,
                                            regularization_losses=self.losses)
            grads = [g if g is not None else tf.zeros_like(v)
                    for g, v in zip(t1.gradient(loss, vars_), vars_)]

            hessians = []
            for g, v in zip(grads, vars_):
                if g is None:
                    hessians.append(tf.zeros_like(v))
                    continue

                # 1) flatten
                g_flat = tf.reshape(g, [-1])  # (N,)
                v_flat = tf.reshape(v, [-1])  # (N,)

                # 2) v_flat 을 명시적으로 감시
                t2.watch(v_flat)

                # 3) jacobian → (N, N)
                J = t2.jacobian(g_flat, v_flat)

                # 4) J가 None인지 확인 후 diag 추출
                if J is None:
                    h_diag_flat = tf.zeros_like(v_flat)
                else:
                    h_diag_flat = tf.linalg.diag_part(J)  # (N,)

                # 5) 원래 모양으로 복원
                h_diag = tf.reshape(h_diag_flat, tf.shape(v))
                hessians.append(h_diag)

        # 테이프 닫기
        del t2
        # 3) step 획득 (bias-correction용)
        step = tf.cast(self.optimizer.iterations + 1, tf.float32)

        lr = self.lr
        # 4) 커스텀 업데이트 루프
        for i, (var, g, h) in enumerate(zip(vars_, grads, hessians)):
            # raw momentum term
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            if isinstance(h, tf.IndexedSlices):
                h = tf.convert_to_tensor(h)
            newH = self.coef_outerDp * tf.square(g) + self.coef_H * h
            newH = tf.abs(newH)
            newH = newH + tf.fill(newH.shape, 0.01)
            d2W = g * self.coef_H / newH

            # 1차 모멘텀 업데이트
            m_i = self.m[i]
            m_i.assign(self.beta1 * m_i + (1 - self.beta1) * d2W)
            m_hat = m_i / (1 - tf.pow(self.beta1, step))

            # 2차 모멘텀(RMSProp) 업데이트
            v_i = self.v[i]
            v_i.assign(self.beta2 * v_i + (1 - self.beta2) * tf.square(g))
            v_hat = v_i / (1 - tf.pow(self.beta2, step))

            # parameter update
            var.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + self.epsilon))

        # 5) iterations 증가
        #    apply_gradients를 쓰지 않으므로 직접 증가시켜야 합니다.
        self.optimizer.iterations.assign_add(1)

        # 6) metrics 업데이트 및 반환
        y_pred = self(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results
