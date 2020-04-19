import tensorflow as tf


class BertScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_ratio, total_steps, name=None):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.warmup_ratio = warmup_ratio
        self.total_steps = float(total_steps)
        self.warmup_steps = warmup_ratio * total_steps
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "BertScheduler"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            total_steps = tf.convert_to_tensor(self.total_steps, name="total_steps")
            warmup_steps = tf.convert_to_tensor(self.warmup_steps, name="warmup_steps")

            step += 1.0

            return initial_learning_rate * tf.cond(
                step < warmup_steps,
                lambda: self.warmup(step, warmup_steps),
                lambda: self.decay(step, total_steps, warmup_steps),
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
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }
