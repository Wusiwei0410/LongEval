from utils import load_model, load_json, json_save
import argparse, json
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams


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

    if "This is a friendly reminder -" in results:
        step_data = None
    else:
        step_data = json.loads(results)
    return step_data


def generate_response(model, tokenizer, prompt, system_prompt=None):
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
        max_new_tokens=9048,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def evaluate_response(
    model, tokenizer, title, context, dimensions, sections, headlines, ref, exp_results
):
    score_responses = {}

    system_prompt = """You are an expert AI assistant and you need to generate a score for the section on a given critereon. You just need to give a score and should not generate an explanation.

                                Example of a valid JSON response:
                                ```json
                                {
                                    "score": your_score
                                }```
                                """

    context_str = "\n\n".join(context)
    raw_text = "\n\n".join(sections)

    lenth_re = []
    for s, c in zip(sections, context):
        if len(s.split(" ")) <= len(c.split(" ")):
            lenth_re.append(1)
        else:
            lenth_re.append(len(c.split(" ")) / len(s.split(" ")))

    if dataset_name == "paper_hf":
        Abstract = context[0]
        Abstract_ref = sections[0]
        Introduction = context[1]
        Introduction_ref = sections[1]
        rest_context = "\n\n".join(context[2:])
    else:
        Abstract = None
        Abstract_ref = None
        Introduction = context[0]
        Introduction_ref = sections[0]
        rest_context = "\n\n".join(context[1:])
        raw_introduction = None

    count = 0
    score_response = None
    for s, c, h in zip(sections, context, headlines):
        if "related work" in s.lower():
            d_text = "Evaluate whether the original paper and the paper written based on the model are similar. Grade the paper (1-10 points)."
            evaluate_prompt = (
                f"Give you Generated text: {c}, Raw Text: {s} \n\n {d_text}"
            )
            while True:  # 一直尝试，直到没有错误
                try:
                    score_response = generate_response(
                        model, tokenizer, evaluate_prompt, system_prompt
                    )
                    score_response = load_response_json(score_response)

                    break  # 成功后退出循环
                except Exception as e:
                    score_response = None
                    break

    if dataset_name == "wikipedia":
        d_text = "Evaluate whether the original text and the text written based on the model are similar. Grade the paper (1-10 points)."
        evaluate_prompt = (
            f"model generated text: {context_str} \n\n raw text:{raw_text}\n\n {d_text}"
        )
        while True:  # 一直尝试，直到没有错误
            try:
                score_response = generate_response(
                    model, tokenizer, evaluate_prompt, system_prompt
                )
                score_response = load_response_json(score_response)
                break  # 成功后退出循环
            except Exception as e:
                score_response = None
                break

        if score_response:
            score_responses["Related-Work Compare"] = score_response["score"]
        else:
            score_responses["Related-Work Compare"] = None

    else:
        if score_response:
            score_responses["Related-Work Compare"] = score_response["score"]
        else:
            score_responses["Related-Work Compare"] = None


    d_text = "Evaluate whether the content in the Introduction corresponds to the rest content in the paper. Grade the paper (1-10 points)."
    evaluate_prompt = f"Give you Generated text: {Introduction}, Raw Text: {Introduction_ref} \n\n {d_text}"
    while True:  # 一直尝试，直到没有错误
        try:
            score_response = generate_response(
                model, tokenizer, evaluate_prompt, system_prompt
            )
            score_response = load_response_json(score_response)
            break  # 成功后退出循环
        except Exception as e:
            score_response = None
            break
    score_responses["Introduction Compare"] = score_response["score"]

    EA_score = []
    ME_score = []
    Instruction_score = []

    Context_str = "\n\n".join(context)
    evaluate_prompt = f"""Given the model-generated text: {Context_str} \n\n. Evaluate whether the model-generated text has repetitive content. The more repetitive the content, the lower the score. Grade the paper (1-10 points).
            If the paper contains many repetitive sections, it would score 2 points.
            if the paper contains a small amount of repetitive sections, it would score 5 points.
            if the paper does not contain repetitive sections, but some content is semantically redundant, and the writing does not effectively express the author's points, it would be score 7 points.
            If the content of the paper is concise, with efficient and precise language, and no informational redundancy, it would score 10 points.
            """
    score_response = generate_response(model, tokenizer, evaluate_prompt, system_prompt)

    score_response = load_response_json(score_response)
    if score_response:
        if type(score_response) == type({"A": 1}):
            Redundancy_score = score_response["score"]
        else:
            Redundancy_score = score_response

    for c, s, h in zip(context, sections, headlines):
        evaluate_prompt = f"""Given a section of the headlines: {h} \n\n Given the model-generate section: {c} \n\n. Evaluate whether the content of this model-generated section meets the key points required in the headline. Grade it based on the following criteria (0-10 points).
                If the degree of relevance between the model-generated text and the headline is low, it would score 2 points.
                If the model-generated text includes all the points from the headline, but the content is somewhat redundant and does not effectively address each issue raised in the headline,  it would score 5 points.
                If the model-generated text covers all the points in the headline, and it can, to some extent, address or clearly express the content of the headline at an academic level, it would score 7 points.
                If the model-generated text covers all the points in the headline, perfectly addressing and clearly expressing the content of the headline at an academic level, and also demonstrates a deep academic exploration with rigorous logic,  it would score 10 points.
                """
        score_response = generate_response(
            model, tokenizer, evaluate_prompt, system_prompt
        )
        score_response = load_response_json(score_response)
        if type(score_response) == type({"A": 1}):
            Instruction_score.append(score_response["score"])
        else:
            Instruction_score.append(score_response)

        if c != Abstract and c != Introduction:
            ##判断是否有实验，否则就是
            evaluate_prompt = f"Given you a section of paper: {s} \n\n Please help me judge whether the section has an analysis of the experimental results. Just respond with 'Yes' or 'No'"
            response = generate_response(model, tokenizer, evaluate_prompt)

            if "yes" in response.lower():
                ## 有实验分析
                evaluate_prompt = f"""Given a section of the raw paper: {s} \n\n Given the model-generated section: {c} \n\n And the headlines that we used to generate: {h} \n\n 
                Compared with the section of raw paper, please help evaluate whether the experimental analysis for the model-generated content is sufficient based on the following criteria (1-10 points):
                1-2 points: The experimental analysis section generated by the model merely reiterates the content of the headline in a simple manner.
                2-4 points: The experimental analysis section generated by the model not only includes the content of the headline but also provides a simple analysis of data variations, supporting the analysis with relevant content.
                4-6 points: The experimental analysis section generated by the model not only includes data analysis and the content of the headline but also further explores the possible reasons behind various experimental phenomena.
                6-8 points: The experimental analysis section generated by the model not only includes data analysis, the content of the headline, and an exploration of the possible causes for the experimental results, but also additionally analyzes the relationships between various experimental results, providing stronger experimental evidence to demonstrate the effectiveness of the methods proposed in the paper.
                9-10 points: The experimental analysis section generated by the model not only includes data analysis, the content of the headline, and an exploration of the possible causes for the experimental results, but also provides additional analysis of the relationships between various experimental results. It demonstrates strong coherence, effectively integrating all experimental analyses under a unified theme."""
                score_response = generate_response(
                    model, tokenizer, evaluate_prompt, system_prompt
                )
                score_response = load_response_json(score_response)

                if type(score_response) == type({"A": 1}):
                    EA_score.append(score_response["score"])
                else:
                    EA_score.append(score_response)
            else:
                evaluate_prompt = f"""Given a section of the raw paper: {s} \n\n Given the model-generated section: {c} \n\n And the headlines that we used to generate: {h} \n\n 
                Compared with the section of raw paper, evaluate whether the model-generated section describing the method is detailed and specific (1-10 points):
                1-2 points: The description of the method simply repeats the content of the headline.
                3-4 points: The description of the method provides a brief introduction to each concept corresponding to the points in the headline but lacks detailed analysis or explanation of the specific content of each model. Alternatively, it may be missing specific formulas for the methods.
                5-6 points: The description of the method uses some basic formulas to introduce the specific approach or provides a brief explanation of how certain method modules operate.
                7-8 points: The description of the methodology section provides a good introduction to the details of the algorithm or experiment, with necessary explanations using formulas. However, the writing lacks coherence between sentences.
                9-10 points: The description of the methodology section provides a thorough introduction to the details of the algorithm or experiment, with formulas used appropriately. The writing style is rigorous, and the context flows smoothly, enabling readers to clearly understand the purpose of each module and its specific details.
                """
                score_response = generate_response(
                    model, tokenizer, evaluate_prompt, system_prompt
                )
                score_response = load_response_json(score_response)

                if type(score_response) == type({"A": 1}):
                    ME_score.append(score_response["score"])
                else:
                    ME_score.append(score_response)
                ME_score.append(score_response["score"])

    if len(EA_score) > 0:
        EA_score = sum(EA_score) / len(EA_score)
        score_responses["EA_score"] = EA_score
    if len(ME_score) > 0:
        ME_score = sum(ME_score) / len(ME_score)
        score_responses["ME_score"] = ME_score
    score_responses["Instruction_following-score"] = sum(Instruction_score) / len(
        Instruction_score
    )
    score_responses["Redundancy_score"] = Redundancy_score

    score_responses["total_len_score"] = sum(lenth_re) / len(
        lenth_re
    )  # predicted_len / total_len
    return score_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for long-text evaluation")

    parser.add_argument("--dataset_name", type=str, default="paper_hf")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-72B")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name

    # result_data = load_json(
    #     f"./data/paper_with_ref/CS/{model_name}_bullet_points_result_data.json"
    # )
    result_data = load_json(
        f"./data/{dataset_name}_result/{model_name}_bullet_points_result_data.json"
    )

    result_data = result_data#[:26]
    score_data = []
    model, tokenizer = load_model("Qwen2.5-72B")
    # model, tokenizer = None, None
    for item in tqdm(result_data):
        sections = item["sections"]
        context = item["result"]
        length_b = item["length_b"]
        # model, tokenizer, title, context, dimensions, sections, headlines
        title = item["title"]
        sections = item["sections"]
        headlines = item["headlines"]
        bullet_points = item["bullet_points"]

        if "ref" in item:
            ref = item["ref"]
        else:
            ref = None

        if "exp_results" in item:
            exp_results = item["exp_results"]
        else:
            exp_results = None

        dimensions = None

        score_responses = evaluate_response(
            model,
            tokenizer,
            title,
            context,
            dimensions,
            sections,
            bullet_points,
            ref,
            exp_results,
        )
        # golden_scores = evaluate_response(model, tokenizer, title, sections, dimensions, sections, headlines)
        print(score_responses)

        item["score_responses"] = score_responses
        score_data.append(item)
    json_save(
        score_data,
        f"./data/paper_with_ref/CS/{model_name}_bullet_points_score_data.json",
    )
    score_data = load_json(
        f"./data/paper_with_ref/CS/{model_name}_bullet_points_score_data.json"
    )
    scores = {}
    for key in score_data[0]["score_responses"]:
        scores[key] = []

    for s in score_data:
        for key in scores:
            if key in s["score_responses"] and s["score_responses"][key] != None:
                scores[key].append(s["score_responses"][key])

    print(scores)
    for key in scores:
        if len(scores[key]) == 0:
            scores[key] = -1
        else:
            scores[key] = sum(scores[key]) / len(scores[key])

    print(scores)

    for key in scores:
        print(key, " : ", scores[key])

    average_s = []
    for key in scores:
        if key == "total_len_score":
            if scores[key] > 1:
                average_s.append(1)
            else:
                average_s.append(scores[key])
        else:
            average_s.append(scores[key])
    average_s = sum(average_s) / len(average_s)
    print("average_score:", average_s)
    # json_save(score_data, f'./data/paper_with_ref/CS/{model_name}_score_data.json')
