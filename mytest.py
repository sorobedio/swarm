import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score


def batch_generate(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=10):
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    outputs = []

    for i in tqdm(range(num_batches)):
        batch = prompts[i * batch_size: (i + 1) * batch_size]
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids.to(f"cuda:{gpu_id}")
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        for j in range(len(batch)):
            outputs.append(tokenizer.decode(output[j][len(input_ids[j]):], skip_special_tokens=True).strip())

        del input_ids, output
        torch.cuda.empty_cache()

    return outputs


def multiple_choice_prompt(instance_dict, dataset):
    prompt = f"Question: {instance_dict['question']}\n"
    if dataset != "knowledge_crosswords":
        prompt = "Please choose an option that best answers the question.\n" + prompt
    for key, val in instance_dict['choices'].items():
        prompt += f"{key}: {val}\n"
    return prompt + "The answer is"


def multiple_choice_answer_parsing(instance_dict, output_text):
    for key in instance_dict['choices']:
        if key in output_text[:5] or key in output_text[-5:]:
            return key
        if instance_dict['choices'][key].lower() in output_text.lower():
            return key
    return "Z"


def evaluate(model_path, dataset, gpu_id, eval_type="multiple_choice", split="dev"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(f"cuda:{gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    with open(f"data/eval/{dataset}.json") as f:
        eval_data = json.load(f)[split]

    if eval_type == "multiple_choice":
        prompts = [multiple_choice_prompt(q, dataset) for q in eval_data]
        outputs = batch_generate(model, tokenizer, prompts, gpu_id)

        golds = []
        preds = []
        for q, out in zip(eval_data, outputs):
            golds.append(q["answer"])
            preds.append(multiple_choice_answer_parsing(q, out))

        return accuracy_score(golds, preds)

    elif eval_type == "exact_match":
        prompts = [q["question"] for q in eval_data]
        max_tokens = 200 if dataset == "gsm8k" else 10
        batch_size = 5 if dataset == "nlgraph" else 10

        outputs = batch_generate(model, tokenizer, prompts, gpu_id,
                                 batch_size=batch_size, max_new_tokens=max_tokens)

        if dataset == "gsm8k":
            outputs = [" ".join(out.split()[-5:]) for out in outputs]

        scores = [1 if q["answer"] in out else 0 for q, out in zip(eval_data, outputs)]
        return sum(scores) / len(scores)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", default="google/gemma-7b-it")
    parser.add_argument("--dataset", default="mmlu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--eval_type", default="multiple_choice")

    args = parser.parse_args()

    # determisnistic model behavior for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Result: {evaluate(args.model_path, args.dataset, args.gpu_id, args.eval_type):.4f}")