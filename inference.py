import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def build_prompt(text: str) -> str:
    return f"""任务：判断下面文本是否包含舆论操控行为。
舆论操控的定义:
舆论操控是指通过各种信息呈现方式与话语策略，刻意引导受众形成片面认知或极端判断，以影响其认知、态度或行为为目的，并可能对公共舆论环境、社会共识及信息传播秩序产生负面影响的行为。

文本：
{text}

输出：
0 = 非舆论操控
1 = 舆论操控"""


# =====================================================
# 标签映射与解析
# =====================================================
LABEL_MAP = {0: "非舆论操控", 1: "舆论操控"}


def extract_prediction(output_text: str):
    """从模型生成文本中提取分类标签"""
    output_text = output_text.strip()
    nums = re.findall(r'[01]', output_text)
    if not nums:
        return None
    return int(nums[-1])


# =====================================================
# 核心推理函数
# =====================================================
@torch.no_grad()
def predict(model, tokenizer, text: str, device: str = "cuda"):
    """
    对单条文本进行舆论操控检测
    Returns: (label_str, probability_note)
    """
    prompt = build_prompt(text)
    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    pred_label = extract_prediction(generated_text)

    if pred_label is None or pred_label not in LABEL_MAP:
        return "未知", generated_text.strip()

    return LABEL_MAP[pred_label], generated_text.strip()


# =====================================================
# 交互式推理模式
# =====================================================
def interactive_mode(model, tokenizer, device):
    """循环接收用户输入并输出预测结果"""
    print("\n" + "=" * 50)
    print("  舆论操控文本检测系统 (交互模式)")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            text = input("请输入待检测文本：\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            print("再见！")
            break

        label, raw_output = predict(model, tokenizer, text, device)
        print(f"\n预测结果：{label}")
        print(f"模型原始输出：{raw_output}")
        print("-" * 50 + "\n")


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="舆论操控文本检测推理脚本"
    )
    parser.add_argument(
        "--model_path", type=str, default="./model",
        help="LoRA适配器权重路径"
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen3-8B",
        help="基础模型名称或路径"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="直接传入待检测文本（不进入交互模式）"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备 (cuda / cpu / mps)"
    )
    args = parser.parse_args()

    # ---------- 加载模型 ----------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    print("Model loaded successfully!\n")

    # ---------- 推理入口 ----------
    if args.text:
        # 命令行单次预测模式
        label, raw_output = predict(model, tokenizer, args.text, args.device)
        print(f"Prediction: {label}")
        print(f"Raw Output: {raw_output}")
    else:
        # 交互式预测模式
        interactive_mode(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
