````markdown
# Platoon Joining Using Deep Reinforcement Learning
A deep reinforcement learning–based strategy for autonomous vehicle platoon joining on highways.

---

### 🧩 Environment Setup
```bash
conda env create -f environment.yml
conda activate pl_env
````

---

### 🚀 Training

Run the training script:

```bash
python train.py
```

To launch the simulator with a graphical interface, use:

```bash
python train.py -gui
```

---

### 🧪 Testing

Run the testing script:

```bash
python test.py
```

To test with the simulator GUI enabled, use:

```bash
python test.py -gui
```

---

### 📊 Results Visualization

Use TensorBoard to visualize training or testing logs:

```bash
tensorboard --logdir=log/<test_or_train>/<config_name>
```

```
