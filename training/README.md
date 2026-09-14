# 训练（QLoRA 微调）

本目录是**训练侧**，与 `app/`（服务侧）刻意隔离：依赖不同、环境不同、产物不同。
服务镜像不包含本目录（见 `.dockerignore`）。

## 为什么训练和推理必须共用一套 prompt

微调最常见的失败方式是：训练时的 prompt 模板和推理时不一致。LLaMA-Factory
会把数据集里的 `human` 字段套上对话模板送进模型；推理侧也必须套同一个模板。
任何一边改动（多一个空格、换一种格式说明），训出来的适配器接上去就几乎无效。

因此本目录**不自己拼 prompt**，一律调用：

```python
from app.services.audit_contract import build_prompt, parse_model_output
```

- `build_prompt()` —— 与推理侧同一个函数，prompt 一致性由构造保证
- `parse_model_output()` —— 与推理侧同一个解析器，训练目标必须能通过它

`app/services/audit_contract.py` 是唯一来源。**改那里，两边同时生效。**

## 切换基座模型时必须同步的位置

换模型（例如从 Qwen2.5-7B 换成 Qwen3-8B）时，下面几处**必须一起改**，漏掉任何一处
都会造成 train/serve skew，微调效果大打折扣：

| # | 位置 | 说明 |
|---|---|---|
| 1 | `.env` 的 `BASE_MODEL` | 推理侧加载哪个模型 |
| 2 | `configs/qlora_sft.yaml` 的 `model_name_or_path` | 训练同一个模型 |
| 3 | `configs/qlora_sft.yaml` 的 `template` | 必须与推理侧对话模板一致 |
| 4 | `configs/export_adapter.yaml` 的两处路径 | 否则导出/加载找不到适配器 |

### Qwen3 特别注意：思考模式

Qwen3 默认**开启思考模式**，会先输出 ` thinking...` 再给答案。这对本项目有两个后果：

1. **JSON 解析会崩** —— `parse_model_output` 用 `find("{")` 定位 JSON，而思考内容里
   几乎必然出现花括号，会命中思考文本而不是最终结果。
2. **训练/推理必须同开同关** —— 两边不一致就是 skew。

因此本项目统一**关闭思考**：

- 推理侧：`app/services/audit_contract.py` 的 `encode_for_model` 会检测模板中是否含
  `enable_thinking`，含则传 `enable_thinking=False`
- 训练侧：`template` 必须用 **`qwen3_nothink`**，不能用 `qwen3`

### 改完后怎么验证对齐

两侧都使用 tokenizer 自带的对话模板，因此只要模型与 `enable_thinking` 一致，渲染结果
就应当一致。落地前建议实测一次：

```bash
# 1) 看推理侧渲染成什么样（需能访问模型权重或已有的本地模型目录）
python -c "
import sys; sys.path.insert(0, '..')
from transformers import AutoTokenizer
from app.services.audit_contract import build_prompt, encode_for_model
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
ids = encode_for_model(tok, build_prompt('基坑深4.5米', []))['input_ids']
print(tok.decode(ids[0], skip_special_tokens=False))
"
# 期望：以 <|im_start|>user 开头，以 <|im_start|>assistant 结尾，
#       且不含  thinking 或任何思考引导字样
```

```bash
# 2) 对比 LLaMA-Factory 用 qwen3_nothink 渲染同一条数据
llamafactory-cli ...  # 或直接用其 template 模块渲染一数据集样本
```

两边的 assistant 前缀一致即可。**若不一致，先改对再训练** —— 这一步花五分钟，
能省掉几小时白跑的训练。

## 工作流

### 0. 先跑基线（强烈建议）

在训练之前，先确认基座模型现在的表现。没有基线，训完无法判断是否有提升。
重点看 **JSON 解析成功率**——这是微调收益最明确的地方。

### 1. 准备环境

```bash
cd training
python -m venv .venv-train && source .venv-train/bin/activate
pip install -r requirements.txt
```

### 2. 准备数据

标注数据格式（每行一条 JSON）：

```json
{
  "paragraph": "施工方案段落原文",
  "clauses": [{"content": "条文原文", "metadata": {"spec_code": "JGJ 59-2011", "article_id": "3.11.3"}}],
  "violations": [
    {"clause_code": "JGJ 59-2011 3.11.3", "clause_content": "基坑周边1.2米范围内严禁堆载。",
     "risk_level": "high", "description": "堆载距离仅0.8米", "suggestion": "调整至1.2米以外", "confidence": 0.95}
  ]
}
```

`violations` 为空数组表示该段落无违规。**正负样本都要有**——全是违规样本会让
模型倾向于无条件报违规。

转换并校验：

```bash
python prepare_dataset.py convert --input labeled.jsonl --output data/train.jsonl
```

转换会同时做质量检查并打印分布统计，有问题会**中止而不写文件**：

- `risk_level` 只能是 `high` / `medium` / `low`。其它取值会被推理侧的聚合逻辑
  静默漏计，等于违规项凭空消失
- 训练目标必须能通过 `parse_model_output()`，即推理侧真能解析
- `has_violation` 必须与 `violations` 自洽

校验已有数据集：

```bash
python prepare_dataset.py check --input data/train.jsonl
```

### 3. 训练

```bash
llamafactory-cli train configs/qlora_sft.yaml
```

配置按 **A6000 单卡、约 16–21GB 可用** 调的（该卡常被其它服务占用约 30GB）。
若你独占整卡 48GB，去掉 `quantization_bit` 改成 bf16 LoRA，精度更高、速度更快。

### 4. 导出适配器

```bash
llamafactory-cli export configs/export_adapter.yaml
```

导出的是**适配器**（几十 MB），不是合并后的完整权重——推理侧
`model_loader.py` 会 `PeftModel.from_pretrained` 加载后 `merge_and_unload`。

### 5. 接回服务

```bash
# .env
LORA_PATH=/abs/path/to/BuildSentry/training/exports/buildsentry-7b-lora
```

重启服务。注意两点：

- `model_loader` 是**懒加载**：首次审核请求才下载/加载基座模型（约 15GB），
  那个请求会等很久，而前端轮询 30 秒就超时。**部署后先手动预热一次。**
- 若服务与训练同机，务必指定显卡。服务侧用 `.env` 的 `MODEL_GPU_INDEX`，训练侧用
  `CUDA_VISIBLE_DEVICES`（LLaMA-Factory 自带的多卡逻辑需要它）。两边指向不同的卡，
  避免训练把线上服务的显存挤掉。

## 目录说明

```
training/
├── README.md
├── requirements.txt              # 训练侧依赖（与服务镜像隔离）
├── dataset_info.json             # LLaMA-Factory 数据集注册
├── prepare_dataset.py            # 数据转换 + 校验（复用 app 的契约）
├── configs/
│   ├── qlora_sft.yaml            # QLoRA 训练配置
│   └── export_adapter.yaml       # 适配器导出配置
├── data/                         # 数据集（内容不提交）
├── saves/                        # 训练产物（不提交）
└── exports/                      # 导出的适配器（不提交）
```

## 调参提示

| 现象 | 调整 |
|---|---|
| 显存不足 | 降 `cutoff_len` → 降 `per_device_train_batch_size` → 升 `gradient_accumulation_steps` 保持有效 batch |
| 输出格式仍不稳定 | 加数据量，尤其是格式规整的样本；或试 `--pretty` 让目标排版贴近 prompt 中的示例 |
| 过拟合（验证损失回升） | 降 `num_train_epochs`、降 `lora_rank`、加数据 |
| 不改 prompt 时效果差 | 确认 `template: qwen` 与推理侧 `apply_chat_template` 一致 |
