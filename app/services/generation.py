# app/services/generation.py
"""
文本生成工具，推理侧与数据生成侧共用。

为什么必须共用：这里有一处极易写错、且写错后症状很隐蔽的地方——
**只能解码新生成的 token**。`outputs[0]` 包含输入 prompt，而 prompt 里带有
格式示例 JSON，一旦整体解码，`parse_model_output` 的 `find("{")` 就会命中
示例而非模型输出，解析必然失败。这个 bug 真实发生过，所以逻辑集中在此，
调用方不要再自己写一遍解码。
"""

from typing import Any

from app.services.audit_contract import encode_for_model


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    do_sample: bool = True,
    enable_thinking: bool = False,
) -> str:
    """生成文本，只返回新生成的部分。

    注意这是同步的 CPU/GPU 计算：在 async 上下文中调用必须用
    asyncio.to_thread 包起来，否则会阻塞整个事件循环。
    """
    inputs = encode_for_model(tokenizer, prompt, enable_thinking=enable_thinking)
    # tokenizer 产出的是 CPU 张量，必须显式搬到模型所在卡上
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
