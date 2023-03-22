# stable-diffusion关键词生成器

## 项目简介

关键词生成器项目使用GPT-2模型为给定的输入句子生成提示词。首先，我们使用提供的训练数据集训练GPT-2模型。然后，我们根据输入句子使用训练好的模型生成提示词。

## 项目文件结构

```markdown
tag/
│
├── README.md               # 项目说明文件，描述项目的目标和使用方法
├── train.py                # 训练模型的脚本
├── predict.py              # 使用训练好的模型生成提示词的脚本
├── dataset.json            # 训练数据集，包含句子、提示词、反向提示词和得分
├── train_text.txt          # 从dataset.json生成的训练文本文件
├── requirements.txt        # 项目依赖库列表，用于安装所需的Python库
│
├── cached_lm_GPT2Tokenizer_128_train_text.txt # GPT-2分词器缓存的训练数据集，加快训练过程中的数据读取
│
├── LICENSE                 # 项目许可证文件，描述项目的许可条款
│
└── output/                 # 训练后的模型和分词器输出目录
    ├── run/                # 训练过程中生成的日志文件，包括损失和其他指标
    ├── config.json         # 模型配置文件，描述模型的结构和参数
    ├── pytorch_model.bin   # 训练后的模型权重文件（PyTorch格式）
    ├── tokenizer_config.json   # 分词器配置文件，描述分词器的设置
    ├── vocab.json          # 词汇表文件，包含模型识别的所有单词及其ID
    ├── generation_config.json   # 生成配置文件，可用于调整生成参数
    ├── special_tokens_map.json   # 特殊令牌映射文件，用于处理特殊字符
    ├── training_args.bin   # 训练参数文件，包含训练过程中使用的设置
    └── merges.txt          # 合并文件，用于将多个字符组合成一个令牌
```

### 文件说明

1. `train.py`用于训练模型的Python脚本。运行此脚本将使用`dataset.json`中的数据训练一个模型，并将训练后的模型保存到`output/`目录中。可以通过修改脚本中的训练参数来优化模型性能。

2. `predict.py`使用训练好的模型生成提示词的Python脚本。运行此脚本时，需要将模型从`output/`目录中加载，并根据输入的句子生成提示词。可以根据需要修改脚本中的生成参数。

3. `dataset.json`JSON格式的训练数据集文件，包含句子、提示词、反向提示词和得分。在运行`train.py`之前，需要准备好此文件。可以通过收集更多数据并添加到此文件中来扩展数据集。

4. `train_text.txt`: 从`dataset.json`生成的训练文本文件。`train.py`脚本将使用此文件作为训练数据输入。在运行`train.py`之前，请确保生成此文件。

5. `requirements.txt`: 包含项目所需Python库的列表。

6. `cached_lm_GPT2Tokenizer_128_train_text.txt`: GPT-2分词器缓存的训练数据集文件。在训练过程中，分词器会将输入数据缓存以加快数据读取速度。这个文件会在运行`train.py`时自动生成，不需要手动创建。

7. `LICENSE`: MIT项目许可证文件。

8. `output/`: 训练后的模型和分词器输出目录。运行`train.py`时，训练后的模型和分词器相关文件将保存在此目录中。`predict.py`脚本将从这里加载模型和分词器。

   - `run/`: 训练过程中生成的日志文件目录，包括损失和其他指标。可以用于监控训练进度和模型性能。
   - `config.json`: 模型配置文件，描述模型的结构和参数。在训练过程中自动生成。
   - `pytorch_model.bin`: 训练后的模型权重文件（PyTorch格式）。在训练过程中自动生成。`predict.py`脚本将从此文件加载模型权重。

   - `tokenizer_config.json`: 分词器配置文件，描述分词器的设置。在训练过程中自动生成。`predict.py`脚本将从此文件加载分词器配置。
   - `vocab.json`: 词汇表文件，包含模型识别的所有单词及其ID。在训练过程中自动生成。`predict.py`脚本将从此文件加载词汇表。
   - `generation_config.json`: 可选的生成配置文件，可以用于调整生成参数，例如生成长度、温度等。如果需要调整生成参数，请在`predict.py`脚本中手动添加相应的设置。
   - `special_tokens_map.json`: 特殊令牌映射文件，用于处理特殊字符，例如换行符、未知字符等。在训练过程中自动生成。`predict.py`脚本将从此文件加载特殊令牌映射。
   - `training_args.bin`: 训练参数文件，包含训练过程中使用的设置。在训练过程中自动生成。如果需要调整训练参数，请在`train.py`脚本中手动添加相应的设置。
   - `merges.txt`: 合并文件，用于将多个字符组合成一个令牌。在训练过程中自动生成。`predict.py`脚本将从此文件加载合并规则。

## 如何使用

### 安装依赖库

在命令

行中运行以下命令以安装依赖：

```bash
pip install -r requirements.txt
```

### 准备训练数据集

在`dataset.json`文件中按照以下格式放置训练数据：

```json
[    ["句子1", "提示词1", "反向提示词1", "得分1"],
    ["句子2", "提示词2", "反向提示词2", "得分2"],
]
```

我已经在`dataset.json`文件里放入了一个模拟数据集，用于测试环境配置是否正确。

### 训练模型

运行`train.py`脚本以训练GPT-2模型。训练完成后，模型将保存在`output`目录中。

```bash
python train.py
```

### 生成提示词

运行`predict.py`脚本以生成提示词。您可以修改`predict.py`中的`prompt`变量以使用不同的输入句子。

```bash
python predict.py
```

### 使用TensorBoard

要在项目中引入 TensorBoard 以可视化训练过程，您需要执行以下步骤：

1. 确保已经安装了 TensorBoard。如果没有，请使用以下命令安装：

```bash
pip install tensorboard
```

2. 在 `train.py` 文件中，已经配置好了 TensorBoard。在创建 `Trainer` 对象时，`args` 参数已经包含了 TensorBoard 日志所需的设置。默认情况下，日志将保存在 `output/runs` 目录中。

3. 在训练模型时，运行以下命令启动 TensorBoard：

```bash
tensorboard --logdir=output/runs
```

4. 启动 TensorBoard 后，它将在终端中显示一个 URL，通常是 `http://localhost:6006/`。在网络浏览器中打开该 URL，将看到 TensorBoard 的界面。

5. 训练模型时，TensorBoard 将自动更新，显示损失、评估指标等的变化。可以通过 TensorBoard 的不同选项卡查看不同类型的信息。

注意：要查看实时更新的训练信息，请确保在训练模型时保持 TensorBoard 运行。如果在训练完成后启动 TensorBoard，将只能查看已经保存的训练日志。

## 自定义参数

### 训练参数

1. 数据集文件路径

更改`dataset.json`以指向您的数据集文件路径。

```python
with open("dataset.json", "r") as f:
```

2. 训练文本文件路径

更改`train_text.txt`以指向您想要存储训练文本的文件路径。

```python
with open("train_text.txt", "w") as f:
```

3. 训练参数

```python
training_args = TrainingArguments(
    output_dir="./output",             # 输出目录
    overwrite_output_dir=True,         # 覆盖输出目录
    num_train_epochs=5,                # 训练轮数
    per_device_train_batch_size=2,     # 每个设备的批处理大小
    save_steps=10_000,                 # 保存模型的步数
    save_total_limit=2,                # 保存的最大模型数量
)
```

### 生成参数

1. 模型文件路径

在`predict.py`文件中，可以通过修改`GPT2Tokenizer.from_pretrained()`和`GPT2LMHeadModel.from_pretrained()`函数中的参数来调整生成时的模型文件路径。

例如，将模型文件保存在名为`custom_model`的文件夹中，您可以像下面这样修改`predict.py`文件：

```python
tokenizer = GPT2Tokenizer.from_pretrained("custom_model")
model = GPT2LMHeadModel.from_pretrained("custom_model")
```

2. 生成参数

您可以在`predict.py`文件中修改生成参数以控制生成提示词的随机性和多样性。以下是生成参数设置示例：

```python
outputs = model.generate(inputs, max_length=75, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
```

- `max_length`生成序列的最大长度。

- `num_return_sequences`要生成的序列数量。

- `do_sample`是否采用随机采样生成文本。如果为`False`，则仅选择最

- 可能的输出；如果为`True`，则从概率分布中随机选择输出。

  - `top_k`在每个生成步骤中，仅从最可能的前`top_k`个输出中进行采样。
  - `top_p`在每个生成步骤中，仅从概率累积到`top_p`的输出中进行采样。
  - `temperature`较低的值会使生成的文本更具确定性，较高的值会使生成的文本更具随机性。


## 注意事项

1. 使用较大的数据集和/或较多的训练轮数可能会导致更好的性能，但也可能增加训练时间。
2. 在生成提示词时，可以通过调整生成参数来控制生成文本的随机性和多样性。例如，可以尝试使用不同的`top_k`、`top_p`和`temperature`值。
3. 生成的提示词质量可能会受到训练数据集质量的影响。如果生成的提示词不满意，请考虑使用更高质量或更大的训练数据集，并增加训练轮数。
4. 请确保在运行`train.py`和`predict.py`之前已正确安装了所有依赖库。如果遇到任何问题，请检查错误消息并尝试解决问题。如果需要帮助，请随时提问。
5. 在使用训练好的模型生成提示词时，确保模型和分词器已正确保存在`output`目录中。如果缺少任何必需文件，请重新运行`train.py`以保存模型和分词器。



