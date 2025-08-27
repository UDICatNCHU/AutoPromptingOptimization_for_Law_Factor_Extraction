import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class LegalCase:
    """法律案例資料結構"""
    case_id: str
    case_type: str  # 案例類型：偽造文書、詐欺等
    case_content: str  # 判決書內容
    verdict: str  # 判決結果
    sentence: str  # 刑期
    hidden_factors: List[str]  # 已知的隱藏要件（來自司法院）

@dataclass  
class PromptOptimizationResult:
    """提示優化結果"""
    iteration: int
    summarize_prompt: str
    optimized_prompt: str
    summarize_result: str
    optimized_result: str
    hidden_factors_extracted: List[str]
    similarity_score: float  # 與標準隱藏要件的相似度

class LegalAutoPrompting:
    """法律判決書自動提示優化系統"""
    
    def __init__(self, llm_model=None):
        """
        初始化系統
        Args:
            llm_model: 語言模型實例（需要實作generate方法）
        """
        self.llm_model = llm_model
        self.logger = self._setup_logger()
        
        # 基礎提示模板
        self.base_prompts = {
            "summarize": """
請分析以下法律判決書，提取關鍵的判決要素和隱藏要件：

判決書內容：
{case_content}

請從以下角度進行分析：
1. 案件事實摘要
2. 適用法條
3. 判決理由中的關鍵要素
4. 影響刑期的隱藏要件
5. 與類似案例的差異點

分析結果：
""",
            
            "enhancement": """
基於以下資訊，請生成一個更精確的分析提示：

原始提示：{original_prompt}
混合判斷結果：{mixed_judgments}
總結結果：{summarize_result}
目標標籤：{label}

請改進提示以更好地識別{case_type}案件的隱藏要件：
""",
            
            "extraction": """
請根據以下優化後的提示，分析判決書並提取隱藏要件：

{optimized_prompt}

判決書：{case_content}

請特別注意以下{case_type}的典型隱藏要件：
{standard_factors}

提取結果：
"""
        }
    
    def _setup_logger(self) -> logging.Logger:
        """設置日誌"""
        logger = logging.getLogger('LegalAutoPrompting')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def auto_prompting_algorithm(self, 
                                case_pair: Tuple[LegalCase, LegalCase],
                                mixed_judgments: str,
                                label: str,
                                max_iterations: int = 4) -> List[PromptOptimizationResult]:
        """
        主要的自動提示優化演算法
        
        Args:
            case_pair: 一對相似情境但判決差異大的案例
            mixed_judgments: 混合判斷結果
            label: 目標標籤
            max_iterations: 最大迭代次數
            
        Returns:
            每次迭代的優化結果列表
        """
        case1, case2 = case_pair
        results = []
        
        # 初始化
        summarize_prompt = self.base_prompts["summarize"]
        summarize_result = self._generate_initial_summary(case1, case2)
        
        self.logger.info(f"開始自動提示優化，目標：{label}，案例類型：{case1.case_type}")
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"開始第 {iteration} 次迭代")
            
            # Step 3-4: 結合各要素作為增強提示，並生成優化提示
            enhancement_prompt = self.base_prompts["enhancement"].format(
                original_prompt=summarize_prompt,
                mixed_judgments=mixed_judgments,
                summarize_result=summarize_result,
                label=label,
                case_type=case1.case_type
            )
            
            optimized_prompt = self._generate_optimized_prompt(enhancement_prompt)
            
            # Step 5: 更新總結提示
            summarize_prompt = optimized_prompt
            
            # Step 7-8: 使用混合判斷作為輸入，生成新的優化結果
            extraction_prompt = self.base_prompts["extraction"].format(
                optimized_prompt=optimized_prompt,
                case_content=f"案例1：{case1.case_content}\n\n案例2：{case2.case_content}",
                case_type=case1.case_type,
                standard_factors=self._get_standard_factors(case1.case_type)
            )
            
            optimized_result = self._generate_optimized_result(extraction_prompt, mixed_judgments)
            
            # 提取隱藏要件
            hidden_factors = self._extract_hidden_factors(optimized_result)
            
            # 計算與標準要件的相似度
            similarity_score = self._calculate_similarity(hidden_factors, case1.hidden_factors)
            
            # 記錄結果
            result = PromptOptimizationResult(
                iteration=iteration,
                summarize_prompt=summarize_prompt,
                optimized_prompt=optimized_prompt,
                summarize_result=summarize_result,
                optimized_result=optimized_result,
                hidden_factors_extracted=hidden_factors,
                similarity_score=similarity_score
            )
            
            results.append(result)
            
            # Step 10: 更新總結結果
            summarize_result = optimized_result
            
            self.logger.info(f"第 {iteration} 次迭代完成，相似度分數：{similarity_score:.3f}")
        
        return results
    
    def _generate_initial_summary(self, case1: LegalCase, case2: LegalCase) -> str:
        """生成初始總結"""
        if self.llm_model:
            prompt = f"""
請分析這兩個{case1.case_type}案例的差異：

案例1（刑期：{case1.sentence}）：
{case1.case_content[:500]}...

案例2（刑期：{case2.sentence}）：
{case2.case_content[:500]}...

請識別導致判決差異的關鍵因素。
"""
            return self.llm_model.generate(prompt)
        else:
            return f"分析{case1.case_type}案例差異的初始總結（模擬結果）"
    
    def _generate_optimized_prompt(self, enhancement_prompt: str) -> str:
        """生成優化後的提示"""
        if self.llm_model:
            return self.llm_model.generate(enhancement_prompt)
        else:
            return f"優化後的提示（基於增強提示的模擬結果）"
    
    def _generate_optimized_result(self, extraction_prompt: str, mixed_judgments: str) -> str:
        """生成優化結果"""
        if self.llm_model:
            full_prompt = f"{extraction_prompt}\n\n參考混合判斷：{mixed_judgments}"
            return self.llm_model.generate(full_prompt)
        else:
            return "模擬的優化結果：提取的隱藏要件包括..."
    
    def _get_standard_factors(self, case_type: str) -> str:
        """獲取特定案例類型的標準隱藏要件"""
        standard_factors_db = {
            "偽造文書": [
                "偽造意圖的明確性",
                "文書的重要性程度", 
                "造成損害的實際程度",
                "行為人的前科記錄",
                "偽造手法的精密程度"
            ],
            "詐欺": [
                "詐欺金額大小",
                "被害人數多寡",
                "詐欺手法的惡劣程度",
                "對社會的影響程度",
                "認罪態度與賠償情況"
            ]
        }
        
        factors = standard_factors_db.get(case_type, ["一般犯罪要件"])
        return "\n".join([f"- {factor}" for factor in factors])
    
    def _extract_hidden_factors(self, result_text: str) -> List[str]:
        """從結果文本中提取隱藏要件"""
        # 簡單的正則表達式提取（實際應用中可能需要更複雜的NLP處理）
        patterns = [
            r"隱藏要件[：:](.+?)(?:\n|$)",
            r"關鍵因素[：:](.+?)(?:\n|$)",
            r"重要要素[：:](.+?)(?:\n|$)"
        ]
        
        factors = []
        for pattern in patterns:
            matches = re.findall(pattern, result_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                # 清理和分割因素
                cleaned_factors = [f.strip() for f in match.split('、') if f.strip()]
                factors.extend(cleaned_factors)
        
        return list(set(factors))  # 去重
    
    def _calculate_similarity(self, extracted_factors: List[str], standard_factors: List[str]) -> float:
        """計算提取要件與標準要件的相似度"""
        if not extracted_factors or not standard_factors:
            return 0.0
        
        # 簡單的關鍵詞匹配相似度計算
        matches = 0
        for extracted in extracted_factors:
            for standard in standard_factors:
                if any(keyword in extracted for keyword in standard.split()):
                    matches += 1
                    break
        
        return matches / len(standard_factors)
    
    def evaluate_performance(self, results: List[PromptOptimizationResult]) -> Dict:
        """評估演算法性能"""
        if not results:
            return {}
        
        similarity_scores = [r.similarity_score for r in results]
        
        evaluation = {
            "total_iterations": len(results),
            "final_similarity_score": similarity_scores[-1],
            "average_similarity_score": sum(similarity_scores) / len(similarity_scores),
            "improvement_rate": similarity_scores[-1] - similarity_scores[0] if len(similarity_scores) > 1 else 0,
            "best_iteration": max(enumerate(similarity_scores), key=lambda x: x[1])[0] + 1,
            "convergence_achieved": similarity_scores[-1] > 0.8
        }
        
        return evaluation

# 使用範例
def example_usage():
    """使用範例"""
    
    # 創建測試案例
    case1 = LegalCase(
        case_id="CASE001",
        case_type="偽造文書",
        case_content="某甲偽造公司印章製作假合約，金額新台幣50萬元...",
        verdict="有罪",
        sentence="有期徒刑6個月",
        hidden_factors=["偽造意圖明確", "金額較小", "初犯"]
    )
    
    case2 = LegalCase(
        case_id="CASE002", 
        case_type="偽造文書",
        case_content="某乙偽造政府公文獲取不當利益，金額新台幣500萬元...",
        verdict="有罪",
        sentence="有期徒刑2年",
        hidden_factors=["偽造政府文書", "金額龐大", "累犯"]
    )
    
    # 初始化系統
    auto_prompting = LegalAutoPrompting()
    
    # 執行演算法
    mixed_judgments = "兩案例均涉及偽造文書，但刑期差異顯著"
    label = "識別影響{case_type}刑期的關鍵隱藏要件"
    
    results = auto_prompting.auto_prompting_algorithm(
        case_pair=(case1, case2),
        mixed_judgments=mixed_judgments,
        label=label,
        max_iterations=4
    )
    
    # 評估性能
    evaluation = auto_prompting.evaluate_performance(results)
    
    print("=== 自動提示優化結果 ===")
    for i, result in enumerate(results, 1):
        print(f"\n第 {i} 次迭代:")
        print(f"相似度分數: {result.similarity_score:.3f}")
        print(f"提取的隱藏要件: {result.hidden_factors_extracted}")
    
    print(f"\n=== 性能評估 ===")
    for key, value in evaluation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_usage()
