import os
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI


class LLMProcessor:
    """
    LLM处理器，用于生成论文的中文翻译和术语解释
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        初始化LLM处理器
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从环境变量获取
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API密钥未提供，请设置DEEPSEEK_API_KEY环境变量")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
    
    def count_tokens(self, text: str) -> int:
        """估算文本的token数量（简化版本）"""
        # 简单估算：1个token ≈ 4个英文字符或2个中文字符
        return len(text) // 4
    
    def generate_interpretation(self, title: str, abstract: str) -> Dict[str, Any]:
        """
        为论文生成中文翻译和术语解释
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            包含中文翻译和术语解释的字典
        """
        prompt = self._build_prompt(title, abstract)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的科学论文翻译和解释专家，擅长将复杂的科学概念转化为易于理解的中文。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_response(result_text)
            
        except Exception as e:
            print(f"LLM处理出错: {e}")
            return {
                "chinese_translation": {
                    "title": "翻译失败",
                    "abstract": "翻译失败"
                },
                "key_terms": []
            }
    
    def _build_prompt(self, title: str, abstract: str) -> str:
        """构建LLM提示词"""
        return f"""请阅读以下论文内容：
标题：{title}
摘要：{abstract}

请生成如下内容：
1. 标题与摘要的中文翻译
2. 摘要中的关键术语列表，并解释每个术语（适合博士水平）

请严格按照以下JSON格式返回结果：
{{
    "chinese_translation": {{
        "title": "中文标题翻译",
        "abstract": "中文摘要翻译"
    }},
    "key_terms": [
        {{
            "term": "术语英文名称",
            "chinese": "术语中文翻译",
            "explanation": "术语的详细解释（适合博士水平）"
        }}
    ]
}}

请确保：
- 中文翻译准确、专业、流畅
- 关键术语选择摘要中最重要、最专业的3-5个术语
- 术语解释要深入、准确，适合博士水平读者理解
- 直接返回JSON格式，不要有其他文本"""
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试直接解析JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # 如果还是失败，返回默认结构
            return {
                "chinese_translation": {
                    "title": "解析失败",
                    "abstract": "解析失败"
                },
                "key_terms": []
            }


class BatchLLMProcessor:
    """
    批量LLM处理器，用于处理多篇论文
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat",
                 max_papers_per_batch: int = 10):
        """
        初始化批量处理器
        
        Args:
            api_key: DeepSeek API密钥
            model: 使用的模型名称
            max_papers_per_batch: 每批次处理的最大论文数量
        """
        self.processor = LLMProcessor(api_key, model)
        self.max_papers_per_batch = max_papers_per_batch
    
    def process_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理论文列表
        
        Args:
            papers: 论文列表，每个论文包含title和abstract字段
            
        Returns:
            处理后的论文列表，包含llm_interpretation字段
        """
        processed_papers = []
        
        for i, paper in enumerate(papers):
            print(f"处理论文 {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # 检查是否已有解读结果
            if "llm_interpretation" in paper:
                processed_papers.append(paper)
                continue
            
            # 生成解读
            interpretation = self.processor.generate_interpretation(
                paper["title"], paper["abstract"]
            )
            
            # 添加到论文数据中
            paper_with_interpretation = paper.copy()
            paper_with_interpretation["llm_interpretation"] = interpretation
            processed_papers.append(paper_with_interpretation)
            
            # 添加延迟以避免API限制
            import time
            time.sleep(1)
        
        return processed_papers


# 工具函数
def save_interpretation_cache(interpretations: Dict[str, Any], cache_file: str = "llm_cache.json"):
    """保存解读结果到缓存文件"""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(interpretations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存缓存失败: {e}")


def load_interpretation_cache(cache_file: str = "llm_cache.json") -> Dict[str, Any]:
    """从缓存文件加载解读结果"""
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    # 测试代码
    processor = LLMProcessor()
    
    test_title = "A new suite of Lund-tree observables to resolve jets"
    test_abstract = "We introduce a class of collider observables, named Lund-Tree Shapes (LTS), defined from declustering trees originating from the Lund jet plane representation of the QCD radiation pattern in multi-jet scattering processes."
    
    result = processor.generate_interpretation(test_title, test_abstract)
    print(json.dumps(result, ensure_ascii=False, indent=2))