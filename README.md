# ChineseAlpacaEval

得益于建模语言分布能力的进一步提升，LLMs不仅可以解决广泛的传统NLP任务，更能够针对自然语言指令生成符合人类偏好的响应。然而，模型在传统NLP基准上较高的得分并不能说明它拥有较高的用户偏好，这是因为这些基准局限在有限的一组任务上。因此，研究者陆续发布了[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)、[MT Bench](https://huggingface.co/spaces/lmsys/mt-bench)、[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval#use-cases)等自动化评估来测试LLMs的指令跟随能力。但是这些基准均主要针对英语，对汉语等其它语言测试不足。

当前涌现出了许多中文大模型的评估基准，例如[C-Eval](https://cevalbenchmark.com/)、[CMMLU](https://github.com/haonan-li/CMMLU)、[GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench)等等。不过这些基准大多使用多项选择的形式测试模型的中文知识，或是集成各类传统中文NLP任务。而对于LLMs能否遵循中文用户的指令，生成符合中文用户偏好的响应，是缺乏评估的。

因此，我们提出了中文指令跟随能力评估基准ChineseAlpacaEval，它基于被广泛认可的AlpacaEval评估基准，利用GPT-4等大模型进行自动化的评测。我们将AlpacaEval的测试指令集经过翻译、中文背景替换、人工校正三步转换成中文，使指令在具有较高流利度的前提下具有中文知识背景。我们将目标模型与text-davinci-003在ChineseAlpacaEval指令集上的回复进行比较，计算获胜率，作为其基准得分。

## Quick Start

### Setup

我们在项目中使用python 3.11。项目中用到的几个python包可通过如下命令进行安装：

```shell
pip install requirements.txt
```

### Usage

**第一步：准备模型生成文件**

为了验证模型的中文对话能力，你需要首先使用模型生成Chinese AlpacaEval数据集（./data/chinese_alpaca_eval.jsonl）中每一个指令（"instruction_zh"字段）的回复。

模型生成文件应是一个JSON Lines文件，每一行包含一个json对象，其格式如下：

```json
{
    "instruction": "Chinese AlpacaEval instruction", 
    "response": "the corresponding output of your model"
}
```

模型生成文件中指令及对应回复的顺序应与原Chinese AlpacaEval数据集保持一致。

生成部分的代码可以参考generation_demo.py。

**第二步：进行评估**

准备好模型生成文件后，git clone该仓库，将生成文件命名为 `<model_name>.jsonl`，放置在 `./model_outputs/` 文件夹下。修改evaluate.sh中的配置内容，然后就可以使用evaluate.sh计算模型的ChineseAlpacaEval得分。

```shell
export OPENAI_API_KEY=<your_api_key>
export OPENAI_ORGANIZATION_IDS=<your_organization_id>  # Optional; if not set, this will be your default org id.

python evaluate.py --model_name='<model_name>' \
    --reference='text-davinci-003' \
    --evaluator='gpt-4-0613' \
```

- --model_name是待评估模型名称。

- --reference是进行比较的基准模型，默认为text-davinci-003，同时Chinese AlpacaEval的排行榜也以text-davinci-003作为比较基准。当前可选模型为['gpt-4-0613', 'gpt-3.5-turbo-0613', 'text-davinci-003']。

- --evaluator为自动评估器，默认为gpt-4-0613，排行榜也使用gpt-4-0613。当前可选模型为['gpt-4-0613', 'gpt-3.5-turbo-0613']。

- 评估结果文件将会在 `./results/<model_name>_vs_<reference>.jsonl`中

### Contributing a model

在经过以上的步骤并生成评估结果文件之后，你可以使用如下的方式提交模型的结果文件。我们将会不断更新ChineseAlpacaEval排行榜。

1. 在Github中Fork该仓库

2. 克隆这个Fork后的仓库 `git clone <URL>`

3. 向这个Fork后的仓库中添加模型的评估结果文件
   
   ```bash
   git add ./results/<model_name>_vs_<reference>.jsonl
   git commit -m "Add <model_name> to ChineseAlpacaEval"
   git push
   ```

4. 完成上述操作之后，向ChineseAlpacaEval提出pull request 

## Leaderboard

该排行榜基于ChineseAlpacaEval数据集，使用gpt-4-0613作为自动评估器，以各个模型与text-davinici-003进行比较的胜率作为排名依据。该排行榜中的所有模型均以temperature=0.7的采样策略进行生成。

| Model                      | Win Rate(%) | Lose Rate(%) | Error(%) |
| -------------------------- | ----------- | ------------ | -------- |
| GPT-4                      | 91.19       | 8.81         | 0        |
| ChatGLM-Pro                | 90.57       | 9.43         | 0        |
| ChatGPT                    | 89.43       | 10.57        | 0        |
| Ernie_bot                  | 88.81       | 11.19        | 0        |
| Baichuan-13B-chat          | 85.28       | 14.72        | 0        |
| Baichuan2-13B-chat         | 85.16       | 14.72        | 0.13     |
| Spark                      | 82.89       | 16.98        | 0.13     |
| Xwin-LM-13B-V0.1           | 82.14       | 17.86        | 0        |
| Chinese-alpaca-2-13b       | 79.87       | 20.13        | 0        |
| Qwen-14B-Chat              | 76.23       | 23.52        | 0.25     |
| WizardLM-13B-V1.2          | 75.6        | 24.4         | 0        |
| ChatGLM2-6B                | 74.34       | 25.66        | 0        |
| Qwen-7B-Chat               | 69.69       | 30.31        | 0        |
| ChatGLM-6B                 | 68.18       | 31.82        | 0        |
| Qwen_turbo                 | 65.66       | 34.34        | 0        |
| Firefly-llama2-13b-chat    | 52.2        | 47.8         | 0        |
| BELLE-Llama2-13B-chat-0.4M | 51.45       | 48.55        | 0        |
| text-davinci-003           | 50.0        | 50.0         | 0        |
| Vicuna-13B                 | 41.89       | 58.11        | 0        |
| Vicuna-7B                  | 30.31       | 69.69        | 0        |
| LLaMA2-13b-chat            | 9.06        | 90.94        | 0        |

**Metric**

我们使用胜率（Win rate）作为衡量模型中文对话能力的指标。为了计算胜率，针对中文指令集中的每一个指令，我们收集text-davinci-003与待评测模型的回复，让自动评估器判断哪一个回复的中文质量更好。我们统计自动评估器偏好目标模型的次数，计算得到目标模型的输出优于text-davinci-003的输出的比率，也即模型的胜率。

我们发现进行评估时，GPT-4有极小的概率输出错误的格式导致无法解析，我们将这种情况命名为’Error’。所以最后的结果中有一小部分模型的胜率与负率相加不到100%。

在最终排序时，若胜率一致，负率较小的优先。text-davinci-003的胜率设置为50%。

**Evaluation Method**

我们的评估方式参考了AlpacaEval，让自动评估模型在两个模型输出中挑选它更偏好的那个。我们目前使用GPT-4作为自动评估模型，使用AlpacaEval评估prompt的中文版本进行评估。该prompt将接收一条指令和一对模型输出，分别对应待评测的两个模型。Prompt如下所示：

```markdown
我希望你创建一个针对大语言模型中文能力的排行榜。为此，我将提供给你中文指令以及两个模型对应的中文回复。请根据中文用户的偏好对这些模型进行排名。所有的输入和输出都应该是Python字典。

这里是提供给模型的指令:
{{
    "instruction": """{instruction}""",
}}

这里是两个模型的输出:
[
    {{
        "model": "model_1",
        "answer": """{output_1}"""
    }},
    {{
        "model": "model_2",
        "answer": """{output_2}"""
    }}
]

现在请你按照中文答案的质量对模型进行排序，以便排在第 1 位的模型输出结果最好。然后返回模型名称和排名的列表，即生成以下输出：
[
    {{'model': <model-name>, 'rank': <model-rank>}},
    {{'model': <model-name>, 'rank': <model-rank>}}
]

你的回答必须是一个有效的Python字典列表，并且除此之外不应包含任何其他内容，因为我们将直接在Python中执行它。请提供大多数中文用户会给出的结果。
```

在输入评估模型前，我们随机化模型响应的顺序以避免位置偏差。评估模型将会参考指令中包含的问题，根据模型中文回复的质量对两个模型进行排序，并最终给出输出质量最高的那个模型。我们将作为评估模型的GPT-4的temperature设置为0，以提高评估的一致性。

## Limitations

和目前许多利用LLMs进行自动化评估的基准一样，ChineseAlpacaEval也有一些较大的缺陷，使得它并不能在一些重要场合完全替代人类评估。

- ChineseAlpacaEval暂时还没有进行详细的人类一致性测试。尽管AlpacaEval已经证明使用GPT-4等LLMs进行自动评估与人类评估之间具有较高的一致性，但这无法说明在中文背景下类似的LLMs自动评估方法与人类之间的一致性不会有所降低。

- ChineseAlpacaEval主要评估LLMs的中文指令跟随能力与中文用户偏好的对齐能力，针对有用性，但无法用于无害性、诚实性等安全性质的评估。

- 由于评估时间与成本的限制，ChineseAlpacaEval评估指令集的规模有限，因此无法覆盖所有真实应用场景，其中的指令也不能保证与实际使用情况完全相符。

- LLMs自动评估可能无法全部识别出模型生成中的幻觉现象。
