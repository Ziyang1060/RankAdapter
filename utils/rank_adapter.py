import json
import re

from loguru import logger
from pydantic import BaseModel

from .LLMClients import create_llm_client
from .ranking_utils import single_prompt_chat

SYSTEM_PROMPT = "You are an intelligent assistant specialized in information retrieval, equipped with powerful text analysis and logical reasoning capabilities."
SYSTEM_PROMPT_ZH = "你是一个专注于信息检索的智能助手，拥有强大的文本分析和逻辑推理能力。"


class Criterion(BaseModel):
    aspect: str
    weight: float
    explanation: str


class Thought(BaseModel):
    unit: str
    criteria: list[Criterion]


class RankAdapter:
    """
    Adaptive Prompt Augmentation for Domain-Specific Ranking Tasks

    This class implements the RankAdapter approach for dynamically augmenting
    generic prompts into query-specific ranking contexts. RankAdapter is designed
    to improve the ranking performance of large language models (LLMs) by
    leveraging ranking unit adaptation and relevance criteria generation.

    Attributes:
        llm_name (str): The name of the LLM to be used.
        llm_client (object): The LLM client object created for interfacing with the LLM.
        mode (str): The operational mode of RankAdapter. Options include "instruction", "adaptive", and "cot".
        language (str): The language of the prompts and tasks (e.g., "en" or "zh").
        total_prompt_tokens (int): Total number of prompt tokens used during execution.
        total_completion_tokens (int): Total number of completion tokens generated.

    Key Features:
        - **Ranking Unit Adaptation**: Adjusts the ranking focus to a domain-specific unit
          (e.g., "doctor" for doctor ranking task of medical domain).
        - **Relevance Criteria Generation**: Dynamically generates comprehensive,
          query-specific relevance criteria to enhance decision-making.
    """

    def __init__(
        self,
        llm: str,
        mode: str="adaptive",
        language: str="en",
    ):
        self.llm_name = llm
        self.llm_client = create_llm_client(llm)
        self.mode = mode
        self.language = language

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _generate_adaptive_unit_and_criteria(self, query: str, instruction: str):
        """
        Generate the adaptive ranking unit and relevance criteria based on the query and instruction.

        This method constructs a task-aware prompt to dynamically define:
            - The unit of analysis (e.g., "paragraph", "doctor").
            - Criteria for relevance evaluation, tailored to the task context.

        Args:
            query (str): The query describing the specific information need of user.
            instruction (str): The instruction specifying the ranking task.

        Returns:
            unit (str): The ranking unit (e.g., "paragraph", "doctor").
            criteria (str): The generated relevance criteria in textual format.
        """
        user_prompt_en = (
            f"Given a user query: {query}, you are performing the retrieval task <{instruction}>.\n"
            "You should carefully review the information above fristly, and then consider how to define task-aware relevance more precisely in the context of the task.\n"
            'Please give your output in JSON format with keys are "unit" and "criteria".\n'
            'Under the content of "unit", please output the information block unit (a word) to retrieve following the instruction, such as a paragraph/sentence for text retrieval and a doctor for doctor retrieval.\n'
            'Under the content of "criteria", please output the relevance criteria tailor to the task context. You need to measure relevance from a comprehensive array of perspectives and assign weights to each criterion. Ensuring that your criteria are in the list format with each item encapsulates a pivotal and unique aspect and has some explanation to it, specifically with keys are "aspect", "weight", "explanation". Your criteria should be correct, clear, executable, and straightforward.\n'
            "Provide a clear and concise response, strictly follow the JSON format as I request, and do not say any other words.\n"
        )
        instruction = "从候选医生池中找到一个擅长使用指定治疗方法治疗疾病的医生专家"
        user_prompt_zh = (
            f"给定用户查询：{query}，您正在执行检索任务<{instruction}>。\n"
            "您应该首先仔细查看上面的信息，然后考虑如何在任务上下文中更准确地定义任务相关性。\n"
            '请以JSON格式输出，其中键为"unit"和"criteria"。\n'
            '在"unit"的内容下，请输出根据指令检索的单位，例如文本检索的"段落"或"句子"，医生检索的"医生"。\n'
            '在"criteria"的内容下，请根据任务上下文输出相关性标准。您需要尽可能从多角度确定全面的相关性标准，并为每个标准分配权重。您应该以列表格式输出，具体键包括"aspect"，"weight"，"explanation"。您的标准应正确，清晰，可执行且简单明了。\n'
            '请您提供明确简洁的响应，严格遵循我要求的JSON格式，保证JSON格式的正确性，不要说其他任何话。\n'
        )
        user_prompt = user_prompt_zh if self.language == "zh" else user_prompt_en
        try:
            response = single_prompt_chat(
                llm_name=self.llm_name,
                llm_client=self.llm_client,
                system_prompt=SYSTEM_PROMPT if self.language == "en" else SYSTEM_PROMPT_ZH,
                user_prompt=user_prompt,
                temperature=0.8
            )
        except ValueError as e:
            print("ValueError", e)
            return "", ""
        self.total_prompt_tokens += int(response["usage"]["prompt_tokens"])
        self.total_completion_tokens += int(response["usage"]["completion_tokens"])
        intent = response["response"]
        unit = ""
        criteria = ""
        json_match = re.search(r"\{.*}", intent, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # use regex to remove trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r'"unit":\s*{\s*"([^"]+)"\s*}', r'"unit": "\1"', json_str, flags=re.DOTALL)
            try:
                model_thought = Thought.model_validate_json(json_str)
                unit = model_thought.unit
                criteria = "\n".join(
                    [
                        f"- <{criterion.aspect}, weight: {criterion.weight}> {criterion.explanation} "
                        for criterion in model_thought.criteria
                    ]
                )
            except Exception as e:
                # try to parse the json string using json.loads
                try:
                    json_data = json.loads(json_str)
                    if "type" in json_data["unit"]:
                        unit = json_data["unit"]["type"]
                    else:
                        unit = json_data["unit"]
                    criteria = "\n".join(
                        [
                            f"- <{criterion['aspect']}, weight: {criterion['weight']}> {criterion['explanation']} "
                            for criterion in json_data["criteria"]
                        ]
                    )
                except Exception as e:
                    logger.error(f"Validation Error: {e} for JSON: {json_str}")

        else:
            logger.error(f"JSON not found in response: {intent}")

        return unit, criteria

    def get_adaptive_ranking_prompt(self, query: str, instruction: str, passages: str, prompt_template: str):
        """
        Generate an adaptive ranking prompt by integrating task-specific relevance criteria.

        This is the primary interface method of the `RankAdapter` class. It takes a base prompt
        template and enhances it with domain-specific adaptations, such as relevance criteria and
        ranking units, to make the prompt contextually suitable for the ranking task.

        The method dynamically adjusts the prompt content based on the `mode`:
            - **instruction**: Uses the given instruction without additional adaptation.
            - **adaptive**: Dynamically generates task-specific relevance criteria and ranking units
              using `_generate_adaptive_unit_and_criteria`.
            - **cot**: Adds a "chain of thought" (CoT) process to define criteria first, then apply
            them to elicit a better ranking result.

        Args:
            query (str): The query describing the user search intent.
            instruction (str): The instruction specifying the ranking task.
            passages (str): The text passages or candidates to be ranked.
            prompt_template (str): The base template for the ranking prompt, containing placeholders
                                   for `query`, `passages`, `instruction`, `unit`, and `criteria`.

        Returns:
            str: The final prompt, enhanced with adaptive criteria and task-specific ranking instructions.
        """
        unit = "paragraph" if self.language == "en" else "段落"
        criteria = ""
        if self.mode == "instruction":
            pass
        elif self.mode == "adaptive":
            unit, criteria = self._generate_adaptive_unit_and_criteria(query, instruction)
            criteria = ((
                            f"你应该根据标准分析给定的{unit}的相关性，并做出最佳的排序决策：\n" if self.language == "zh" else
                            f"You should follow the criteria to analyze the given {unit}s' relevance and make the best possible ranking decision:\n")
                        + criteria
                        )
        elif self.mode == "cot":
            criteria_en = """Firstly, determine the appropriate criteria to make your decision the best. After writing down your decision criteria, you need to follow these criteria by thinking step by step."""
            criteria_zh = """首先，确定做出最佳决定所需的适当标准。在写下你的决策标准后，你需要按照这些标准一步一步地思考。"""
            replacement_mapping = {
                "Output only the identifier of the most relevant {unit}. Do not explain any reason.":
                    "Output your decision criteria first, then output the identifier of the most relevant {unit}.",
                "只输出最相关的{unit}的标识符。不要解释任何原因。":
                    "请先输出决策标准，然后输出最相关{unit}的标识符。",
                "Output only your judgement score. Do not explain any reason.":
                    "Output your decision criteria first, then output your judgement score.",
                "请输出您的判断分数，无需解释原因。":
                    "请先输出决策标准，然后输出您的判断分数。",
                "Output only the ranking results. Do not explain any reason.":
                    "Output your decision criteria first, then output the ranking results.",
                "请直接输出排名结果，无需解释原因。":
                    "请先输出决策标准，然后直接输出排名结果。"
            }
            for original, replacement in replacement_mapping.items():
                if original in prompt_template:
                    prompt_template = prompt_template.replace(original, replacement)
            criteria = criteria_en if self.language == "en" else criteria_zh

        return prompt_template.format(
            query=query,
            passages=passages,
            instruction=instruction,
            unit=unit,
            criteria=criteria,
        )


if __name__ == "__main__":
    from pyserini.search import get_topics

    rank_adapter = RankAdapter(llm="Qwen2-7B-Instruct", language="en")

    dataset_name = "touche"
    INSTRUCTIONS = {
        "touche": "Retrieve the top N most relevant scientific articles related to the query, focusing on research addressing the specific topic or question about COVID-19.",
        "covid": "Retrieve the top N arguments, either supporting or opposing the query’s claim. Focus on well-reasoned, relevant arguments for debate.",
    }

    query_map = {}
    topics_mapping = {
        'covid': 'beir-v1.0.0-trec-covid-test',
        'touche': 'beir-v1.0.0-webis-touche2020-test',
    }
    topics = get_topics(topics_mapping[dataset_name])
    for topic_id in list(topics.keys()):
        query_map[str(topic_id)] = topics[topic_id]['title']

    query2criteria = {}
    instruction = INSTRUCTIONS[dataset_name]
    for query_id, query in query_map.items():
        query2criteria[query] = rank_adapter._generate_adaptive_unit_and_criteria(query=query,
                                                                                  instruction=instruction)  # noqa
        print(f"Query ID: {query_id}, Query: {query}, Criteria: {query2criteria[query]}")
        exit()
