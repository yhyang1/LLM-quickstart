import argparse
import torch
from transformers import GPT2Tokenizer, OPTForCausalLM
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
def print_memory_footprint(model):
    print("Memory footprint of model: {} GB".format(model.get_memory_footprint()/1024/1024/1024))

def load_model(model_id, load_in_8bit=False):
    model = OPTForCausalLM.from_pretrained(model_id, load_in_8bit=load_in_8bit)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=8,  # LoRA的秩，影响LoRA矩阵的大小
        lora_alpha=32,  # LoRA适应的比例因子
        # 指定将LoRA应用到的模型模块，通常是attention和全连接层的投影
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
        lora_dropout=0.05,  # 在LoRA模块中使用的dropout率
        bias="none",  # 设置bias的使用方式，这里没有使用bias
        task_type="CAUSAL_LM"  # 任务类型，这里设置为因果(自回归）语言模型
    )
    model = get_peft_model(model, config)
    return model, tokenizer

def load_dataset(dataset_path, tokenizer, field="quote"):
    dataset = load_dataset(dataset_path)
    tokenized_dataset = dataset.map(lambda samples: tokenizer(samples[field]), batched=True)
    return tokenized_dataset

def train(model, tokenizer, tokenized_dataset, args):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"{args.model_dir}/{args.model_id}-lora",  # 指定模型输出和保存的目录
        per_device_train_batch_size=args.batch_size,  # 每个设备上的训练批量大小
        learning_rate=args.learning_rate,  # 学习率
        fp16=args.fp16,  # 启用混合精度训练，可以提高训练速度，同时减少内存使用
        logging_steps=args.logging_steps,  # 指定日志记录的步长，用于跟踪训练进度
        max_steps=args.max_steps,  # 最大训练步长
        num_train_epochs=args.num_train_epochs  # 训练的总轮数
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    model.use_cache = False  # 不使用缓存，减少内存使用

    trainer.train()
    model_path = f"{args.model_dir}/{args.model_id}-lora"

    trainer.save_pretrained(model_path)

    return model_path

def inference(text, tokenizer, args):
    model_path = f"{args.model_dir}/{args.model_id}-lora"
    model = OPTForCausalLM.from_pretrained(model_path)

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(torch.device("cuda"))
    output = model.generate(input_ids, do_sample=True, max_length=50, top_p=0.95, top_k=60)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

def run(args):
    if args.inference:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_id)
        inference(args.text, tokenizer, args)
        return

    model, tokenizer = load_model(args.model_id, args._8bit)
    model.print_trainable_parameters()

    tokenized_dataset = load_dataset(args.dataset_path, tokenizer, args.tokenized_field)

    model_path = train(model, tokenizer, tokenized_dataset, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="gpt2", help="pretrained model name or path")
    parser.add_argument("--8bit", action="store_true", help="use 8bit quantization")
    parser.add_argument("--dataset_path", type=str, default="Abirate/english_quotes", help="dataset path")
    parser.add_argument("--tokenized_field", type=str, default="quote", help="tokenized field name")
    parser.add_argument("--model_dir", type=str, default="models", help="directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--fp16", action="store_true", help="use fp16 training")
    parser.add_argument("--logging_steps", type=int, default=20, help="logging steps")
    parser.add_argument("--max_steps", type=int, default=100, help="max training steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--inference", action="store_true", help="inference mode")
    parser.add_argument("--text", type=str, default="Two things are infinite: ", help="text to inference")

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
