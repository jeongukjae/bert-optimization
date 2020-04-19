import tensorflow as tf


class BertScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_ratio, total_steps, name=None):
        super().__init__()

        self.warmup_ratio = warmup_ratio
        self.total_steps = float(total_steps)
        self.warmup_steps = warmup_ratio * total_steps
        self.name = name

    def __call__(self, step):
        with tf.name_scope("BertScheduler"):
            total_steps = tf.convert_to_tensor(self.total_steps, name="total_steps")
            warmup_steps = tf.convert_to_tensor(self.warmup_steps, name="warmup_steps")

            current_step = step + 1.0

            return tf.cond(
                current_step < warmup_steps,
                lambda: self.warmup(current_step, warmup_steps),
                lambda: self.decay(current_step, total_steps, warmup_steps),
            )

    @tf.function
    def warmup(self, step, warmup_steps):
        return step / tf.math.maximum(tf.constant(1.0), warmup_steps)

    @tf.function
    def decay(self, step, total_steps, warmup_steps):
        return tf.math.maximum(
            tf.constant(0.0), (total_steps - step) / tf.math.maximum(tf.constant(1.0), total_steps - warmup_steps)
        )

    def get_config(self):
        return {
            "warmup_ratio": self.warmup_ratio,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }
