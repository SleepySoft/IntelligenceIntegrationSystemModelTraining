import json
import torch
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 1. 针对 MI50 的环境配置 =================
# MI50 (gfx906) 必备
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.6"
# 解决碎片化显存分配失败的问题
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"


TRAIN_PROMPT = """
请基于以下新闻报道，提取并结构化关键信息，严格按照以下要求输出：
EVENT_TITLE：用一句简洁的中文概括核心事件，避免直接复制原标题。
EVENT_BRIEF：用1-2句中文提炼事件最核心的要素（人物、地点、核心冲突/风险）。
EVENT_TEXT：用一个连贯的中文段落（约3-5句）详细描述事件背景、经过、各方立场和现状。整合>原文关键细节，确保信息完整。
RATE：对事件在以下维度的潜在或实际影响进行0-10分的评分（0为无影响，10为极大影响）：
国家政策：依据政策影响的广度与深度评分，从>地方性措施到重大国策。
国际关系：依据事件对国际或地区局势的改变程度评分，从日常活动到战争冲突。
政治影响：依据事件的政治敏感性与层级评分，从日常活动到最高层重大变故。
商业金融：依据对经济金融体系的冲击程度评分，从公司动向到系统危机。
科技信息：依据技术的突破性与影响力评分，从一般报道到颠覆性突破。
社会事件：主要依据事件恶性程度与发生地（中国国内事件评分显著高于国外同类事件）评分。
其它信息：用于归类上述六类之外的信息，并根据其价值给予0-8分。
内容准确率：基于原文>信息明确性和来源可信度评分。
"""


def main(adapter_path, base_model_path, test_data_path, output_file):
    print(f"🔄 Processing Adapter: {adapter_path}")
    print(f"🤖 Base Model: {base_model_path}")
    print(f"📂 Input Data: {test_data_path}")
    print(f"💾 Output File: {output_file}")

    # ================= 2. 加载数据与断点续传检测 =================
    with open(test_data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f_out:
            lines = [line for line in f_out if line.strip()]
            start_index = len(lines)
            if start_index > 0:
                print(f"⚠️  Found existing file with {start_index} samples. Resuming...")

    data_to_process = all_data[start_index:]
    if len(data_to_process) == 0:
        print("🎉 All data processed! Nothing to do.")
        return

    # ================= 3. 加载模型 (改为 FP16 以适应 MI50) =================
    print("⏳ Loading model into VRAM (FP16 Mode)...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 【关键修改】使用 float16。
    # 14B FP16 ~28GB VRAM。单张 MI50 (32G) 勉强能放下，
    # 但建议使用 device_map="auto" 让两张卡分担，留出空间给推理上下文。
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"🔗 Merging LoRA weights from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # ================= 4. 流式推理与写入 =================
    print(f"🚀 Starting inference on remaining {len(data_to_process)} samples...")

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, 'a', encoding='utf-8', buffering=1) as f:
        for item in tqdm(data_to_process, desc="Inference"):
            # instruction = item.get("instruction", "")
            instruction = TRAIN_PROMPT              # 使用训练时的prompt
            input_text = item.get("input", "")
            # ground_truth = item.get("output", "") # 推理时其实不需要 GT

            # 【关键修改】DeepSeek-R1 模板适配
            # 很多 R1 Distill 模型不需要强制 System Prompt，或者依靠 tokenizer_config.json 自动处理
            # 这里构建标准 chat 格式，让 tokenizer 自己去拼凑 <|im_start|> 等标记
            messages = [
                {"role": "user", "content": instruction + "\n" + input_text}
            ]

            # 如果你确实觉得需要 system prompt，可以在上面加，但 R1 经常会忽略它而直接开始 <think>

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=8192,  # R1 模型通常话比较多（包含思考过程），建议调大
                        temperature=0.6,  # R1 建议温度稍低一点，或者 0.6-0.7
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # 只截取新生成的部分
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # 可选：如果模型输出了 <think> 标签，你可能想在这里做些后处理

            except Exception as e:
                print(f"\n❌ Error generating sample: {e}")
                response = f"ERROR_GENERATION: {str(e)}"

            result_entry = {
                "instruction": instruction,
                "input": input_text,
                "model_output": response,
                "adapter": adapter_path
            }

            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            f.flush()

    print(f"✅ Done! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 增加 base_model 参数，不再硬编码
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to Base Model (DeepSeek-R1-Distill-Qwen-14B)")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA checkpoint folder")
    parser.add_argument("--data", type=str, default="alpaca_test.json", help="Path to test json")
    parser.add_argument("--output", type=str, required=True, help="Path to output jsonl")

    args = parser.parse_args()
    main(args.adapter, args.base_model, args.data, args.output)

"""
python validation1_batch_inference_v2.py \
--base_model /home/sleepy/Depot/ModelTrain/qwen/DeepSeek-R1-Distill-Qwen-14B \
--adapter /home/sleepy/Depot/ModelTrain/qwen/DeepSeek-R1-LoRA/checkpoint-700 \
--data /home/sleepy/Depot/ModelTrain/IntelligenceIntegrationSystemModelTraining/Data/v1/alpaca_test.json \
--output result_checkpoint-700.jsonl
"""

