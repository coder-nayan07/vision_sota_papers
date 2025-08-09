# plot_from_history.py
import json
import matplotlib.pyplot as plt

HISTORY_FILE = 'training_history.json'
PLOT_LOSS_FILE = 'loss_vs_epoch_from_json.jpg'

# Load the history
with open(HISTORY_FILE, 'r') as f:
    history = json.load(f)

# Create a plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

epochs = range(1, len(history['train_loss']) + 1)

ax.plot(epochs, history['train_loss'], 'o-', label='Training Loss')
ax.plot(epochs, history['val_loss'], 'o-', label='Validation Loss')

ax.set_title('Training and Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)

fig.savefig(PLOT_LOSS_FILE, dpi=300)
print(f"Plot saved to {PLOT_LOSS_FILE}")
plt.show()
