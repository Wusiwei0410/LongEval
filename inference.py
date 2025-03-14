from utils import load_model, load_json, json_save
import argparse, json
from tqdm import tqdm
import os

def internLM_generate(prompt):
    inputs = tokenizer([prompt], return_tensors="pt")
    for k,v in inputs.items():
        inputs[k] = v.cuda()
    gen_kwargs = {"max_length": 2048, "top_p": 0.9, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.1}
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    print(output)

    
    return output

def load_response_json(response):
    results = response


    results = results.replace("json", "")
    results = results.replace("\n", "")
    results = results.replace("```", "")

    if "##" in results:
        results = results.split("#")[0]
    elif "**" in results:
        results = results.split("*")[0]

    results = results.strip()

    step_data = json.loads(results)
    return step_data




def generate_response(model, tokenizer, prompt, system_prompt=None):
    if 'internlm2' in model_name:
        response = model(prompt).text
    elif 'LongWriter' in model_name:
        if system_prompt:
            prompt = system_prompt + ' ' + prompt
        prompt = f"[INST]{prompt}[/INST]"
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to('cuda')
        context_length = input.input_ids.shape[-1]
        output = model.generate(
            **input,
            max_new_tokens=32768,
            num_beams=1,
            do_sample=True,
            temperature=0.5,
        )[0]
        response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    else:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            # max_new_tokens=2048
            max_new_tokens=4154,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def agent_long_text_generate(
    model, tokenizer, title, headlines, length_requirement, sections, ref, exp_results
):
    context = []
    length_b = []

    for h, s in tqdm(zip(headlines, sections)):
        l = len(s.split(" "))
        if dataset_name == "paper_hf":
            if context:
                if "introduction" in s.lower():
                    prompt = f"Title: {title} \n\n  Context: {context} \n\n please generate a Introduction that is more than {l} words based on the given bullet point: {h}, the title, and the context."  # You can also use the content in the table1-3 which is related with the bullet point: {h}
                elif "related work" in s.lower():
                    prompt = f"Title: {title} \n\n  Context: {context} \n\n Referneces: {ref} \n\n please generate a Related Work section of the paper that is more than {l} words based on the given bullet point: {h}, the title, and the context."  # You can also use the content in the table1-3 which is related with the bullet point: {h}
                else:
                    ##判断是否需要使用实验结果
                    prompt = f"Give you a bullet point: {h} \n\n and experiment results:{exp_results} \n\n. please judge whether you need to use the experiments results to write the section based on the bullet point. Answer me 'Yes' or 'No'"
                    judge = generate_response(model, tokenizer, prompt)
                    if "Yes" in judge:
                        prompt = f"Title: {title} \n\n  Context: {context} \n\n experiment results:{exp_results} \n\n please continually generate a section that is more than {l} words based on the given bullet point: {h}, the title, experiments results, and the context."  # You can also use the content in the table1-3 which is related with the bullet point: {h}
                    else:
                        prompt = f"Title: {title} \n\n  Context: {context} \n\n please continually generate a section that is more than {l} words based on the given bullet point: {h}, the title, and the context."  # You can also use the content in the table1-3 which is related with the bullet point: {h}
            else:
                prompt = f"Title: {title} \n\n  please generate a abstract that is more than {l} words based on the given bullet point: {h}, and the title."  # You can also use the content in the table1-3 which is related with the bullet point: {h}
               
            p = generate_response(model, tokenizer, prompt)
            context.append(p)
            length_b.append(len(p.split(" ")) / l)

        elif dataset_name == "blog":
            if context:
                if "introduction" in s.lower():
                    prompt = f"Title: {title} \n\n  Context: {context} \n\n please generate a Introduction that is more than {l} words based on the given bullet point: {h}, the title, and the context."
                else:
                    prompt = f"Title: {title} \n\n  Context: {context} \n\n please continually generate a section that is more than {l} words based on the given bullet point: {h}, the title, and the context."
            else:
                prompt = f"Title: {title} \n\n  please generate a section that is more than {l} words based on the given bullet point: {h}, and the title."
            p = generate_response(model, tokenizer, prompt)
            context.append(p)

            length_b.append(len(p.split(" ")) / l)
        elif dataset_name == "wikipedia":
            if context:
                if "introduction" in s.lower():
                    prompt = f"Title: {title} \n\n  Context: {context} \n\n please generate a Introduction that is more than {l} words based on the given bullet point: {h}, the title, and the context."
                else:
                    prompt = f"Title: {title} \n\n  Context: {context} \n\n please continually generate a section that is more than {l} words based on the given bullet point: {h}, the title, and the context."
            else:
                prompt = f"Title: {title} \n\n References: {ref} \n\n please generate a section that is more than {l} words based on the given bullet point: {h}, references, and the title."
            p = generate_response(model, tokenizer, prompt)
            context.append(p)

            length_b.append(len(p.split(" ")) / l)

    return context, length_b


def LLM_long_text_generate(
    model, tokenizer, title, headlines, length_requirement, sections, ref, exp_results
):
    context = []
    length_b = []
    l = len(" ".join(sections).split(" "))
    prompt = f"Title: {title} \n\n  The headlines of the paper: {headlines} \n\n please generate the paper directtly that is around {l} words based on the given headlines: {h}."  # You can also use the content in the table1-3 which is related with the bullet point: {h}
    context = generate_response(model, tokenizer, prompt)
    length_b = len(context.split(" "))

    return context, length_b


def evaluate_response(
    model, tokenizer, title, context, dimensions, sections, headlines, ref, exp_results
):
    context_str = "\n\n".join(context)
    raw_text = "\n\n".join(sections)
    score_responses = {}
    total_len = 0
    for s in sections:
        total_len += len(s.split(" "))

    predicted_len = 0
    for s in context:
        predicted_len += len(s.split(" "))

    for level in dimensions:
        if level == "section-level":
            for i, d in enumerate(dimensions[level]):
                system_prompt = """You are an expert AI assistant and you need to generate a score of the section from a given dimension, and you just need to give me score and do not need to generate explanation.

                                Example of a valid JSON response:
                                ```json
                                {
                                    "score": your_score
                                }```
                                """
                d_text = dimensions[level][d]

                if dataset_name == "paper_hf":
                    Abstract = context[0]
                    Introduction = context[1]
                    raw_introduction = sections[1]
                    rest_context = "\n\n".join(context[2:])
                else:
                    Introduction = context[0]
                    rest_context = "\n\n".join(context[1:])
                    raw_introduction = None

                if d == "Introduction Quality":
                    evaluate_prompt = f"Give you the title of the paper: {title} \n\n Raw Introduction: {raw_introduction} \n\n Generated Introduction: {Introduction} \n\n {d_text}"
                    while True:  # 一直尝试，直到没有错误
                        try:
                            score_response = generate_response(
                                model, tokenizer, evaluate_prompt, system_prompt
                            )
                            score_response = load_response_json(score_response)
                            break  # 成功后退出循环
                        except Exception as e:
                            print(f"Error occurred for generated json file")
                    if type(score_response) == type(2):
                        score_responses[d] = score_response
                    else:
                        score_responses[d] = score_response["score"]

                elif d == "Instruction-following-sec":
                    instrct_follo_s = []
                    for s, h in zip(context, headlines):
                        evaluate_prompt = f"Give you the title of the paper: {title}, Headline: {h}, Generated Section: {s} \n\n {d_text}"
                        while True:  # 一直尝试，直到没有错误
                            try:
                                score_response = generate_response(
                                    model, tokenizer, evaluate_prompt, system_prompt
                                )
                                score_response = load_response_json(score_response)
                                break  # 成功后退出循环
                            except Exception as e:
                                print(f"Error occurred for generated json file")
                        if type(score_response) == type(1):
                            instrct_follo_s.append(score_response)
                        else:
                            instrct_follo_s.append(score_response["score"])
                    score_responses[d] = sum(instrct_follo_s) / len(instrct_follo_s)

                elif d == "Related-Work Compare":
                    count = 0
                    for s, c, h in zip(sections, context, headlines):
                        if "related work" in s.lower():
                            count = 1
                            evaluate_prompt = f"Give you Generated text: {c}, Raw Text: {s} \n\n {d_text}"
                            while True:  # 一直尝试，直到没有错误
                                try:
                                    score_response = generate_response(
                                        model, tokenizer, evaluate_prompt, system_prompt
                                    )
                                    score_response = load_response_json(score_response)
                                    break  # 成功后退出循环
                                except Exception as e:
                                    print(f"Error occurred for generated json file")

                            if type(score_response) == type(1):
                                score_response = score_response
                            else:
                                score_response = score_response["score"]
                    score_responses[d] = score_response
        score_responses["total_len_score"] = predicted_len / total_len
    return score_responses


def process_data(model, tokenizer):
    from datasets import load_dataset

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("SiweiWu/LongEval")
    if dataset_name == "paper_hf":
        all_data = ds['Paper']
    elif dataset_name == "blog":
        all_data = ds['Blog']
    elif dataset_name == "wikipedia":
        all_data = ds['Wikipedia']

    result_data = []
    for data in tqdm(all_data):
        ref = data["related_work_ref"]
        if len(ref) == 1 and ref[0] == '':
            ref = None

        exp_results = data["experiment_results"]
        if len(exp_results) == 1 and exp_results[0] == '':
            exp_results = None


        title = data["title"]
        bullet_points = data["bullet_points"]
        sections = data["sections"]
        length_requirement = data["length_requirement"]

        context, length_b = agent_long_text_generate(
            model,
            tokenizer,
            title,
            bullet_points,
            length_requirement,
            sections,
            ref,
            exp_results,
        )

        data["result"] = context
        data["length_b"] = length_b
        result_data.append(data)
    
    if os.path.exists(f"./data/{dataset_name}_result/") == False:
        os.mkdir(f"./data/{dataset_name}_result/")

    json_save(
        result_data,
        f"./data/{dataset_name}_result/{model_name}_bullet_points_result_data.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for long-text evaluation")

    parser.add_argument("--dataset_name", type=str, default="paper_hf")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-72B")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name

    data_dir = f"./data/{dataset_name}/"

    fns = os.listdir(data_dir)
    model, tokenizer = load_model(model_name)



    process_data(model, tokenizer)