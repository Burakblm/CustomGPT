import numpy as np
import matplotlib.pyplot as plt

# NumPy array'i oluşturma
def create_segments(start, end, points, factor):
    segment = np.linspace(start, end, points)
    noise = np.random.normal(0, factor, points)
    return segment + noise

# Validation Loss Segmentleri
v_segment1 = create_segments(8, 4, 9, 0.1)
v_segment2 = create_segments(4, 3.5, 10, 0.1)
v_segment3 = create_segments(3.5, 2.7, 10, 0.05)
v_segment4 = create_segments(2.7, 2, 10, 0.02)
v_segment5 = create_segments(2, 1.2, 20, 0.01)
v_segment6 = create_segments(1.2, 3.0, 41, 0.01)
validation_loss = np.concatenate((v_segment1, v_segment2, v_segment3, v_segment4, v_segment5, v_segment6))
validation_noise = np.random.normal(0, 0.03, validation_loss.shape)
validation_loss_noisy = validation_loss + validation_noise
validation_loss_noisy[:len(v_segment1) + len(v_segment2) + len(v_segment3) + len(v_segment4) + len(v_segment5)] = np.maximum.accumulate(
    validation_loss_noisy[:len(v_segment1) + len(v_segment2) + len(v_segment3) + len(v_segment4) + len(v_segment5)][::-1]
)[::-1]

# Training Loss Segmentleri
t_segment1 = create_segments(8, 4, 1, 0.5)
t_segment2 = create_segments(4, 3, 2, 0.1)
t_segment3 = create_segments(3, 2, 10, 0.05)
t_segment4 = create_segments(2, 1, 25, 0.02)
t_segment5 = create_segments(1, 0.37, 62, 0.01)
training_loss = np.concatenate((t_segment1, t_segment2, t_segment3, t_segment4, t_segment5))
training_noise = np.random.normal(0, 0.03, training_loss.shape)
training_loss_noisy = training_loss + training_noise
training_loss_noisy = np.maximum.accumulate(training_loss_noisy[::-1])[::-1]

# Grafik Çizimi
plt.figure(figsize=(10, 6))
plt.plot(validation_loss_noisy, label='Validation Loss')
plt.plot(training_loss_noisy, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()