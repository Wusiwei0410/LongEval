# LongEval

This is the repo for the paper: ["LongEval: A Comprehensive Analysis on Long-Text Generation Through Plan-based Paradigm"](https://arxiv.org/pdf/2502.19103).



## Dataset

Our benchmark has uploaded to the hf:

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("SiweiWu/LongEval")
```

## Agent-based Long Text inference

As for Qwen-2.5-72B-Instrcut, you can use the following code to infer:

```
python inference.py  --dataset_name paper_hf --model_name Qwen2.5-3B
python inference.py  --dataset_name blog --model_name Qwen2.5-3B
python inference.py  --dataset_name wikipedia --model_name Qwen2.5-3B
```

## Evaluation

As for Qwen-2.5-72B-Instrcut, you can use the following code to evaluate:

```
python evaluator_api.py --model_name  LLaMa3.1-8B --dataset_name paper_hf 
python evaluator_api.py --model_name  LLaMa3.1-8B --dataset_name blog 
python evaluator_api.py --model_name  LLaMa3.1-8B --dataset_name wikipedia 
```
## Citation


```
@misc{wu2025longevalcomprehensiveanalysislongtext,
      title={LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm}, 
      author={Siwei Wu and Yizhi Li and Xingwei Qu and Rishi Ravikumar and Yucheng Li and Tyler Loakman and Shanghaoran Quan and Xiaoyong Wei and Riza Batista-Navarro and Chenghua Lin},
      year={2025},
      eprint={2502.19103},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19103}, 
}
```
