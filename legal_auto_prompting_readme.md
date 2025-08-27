# 法律判決書隱藏要件自動提示優化系統

## 📚 專案概述

本系統是一個專門針對法律判決書分析的自動提示優化工具，旨在透過迭代優化的方式，自動學習並生成最佳的提示（Prompt），以提取影響法律判決的隱藏要件。

### 🎯 研究背景

在法律實務中，相似情境的案例往往會有截然不同的判決結果。這些差異往往源於一些不易察覺的「隱藏要件」。本系統利用大型語言模型（LLM）和司法院公開資料，透過自動化的方式學習識別這些關鍵要件。

### 🔬 核心理念

- **自動化學習**：透過4次迭代優化，自動學習最佳的分析提示
- **對比分析**：針對情境相似但判決差異大的案例進行深度比較
- **標準化評估**：與司法院定義的隱藏要件進行對比，確保分析品質
- **領域專精**：專門針對偽造文書、詐欺等特定案例類型優化

## ✨ 主要功能

### 🔄 自動提示優化演算法
- 實作4次迭代的提示優化流程
- 動態調整分析策略以提高精確度
- 自動學習領域特定的分析模式

### 📋 隱藏要件提取
- 自動識別影響判決的關鍵因素
- 支援多種案例類型（偽造文書、詐欺等）
- 與司法院標準要件庫比對驗證

### 📊 性能評估機制
- 相似度分數計算
- 迭代改進追蹤
- 收斂性分析

### 🏛️ 法律領域專用設計
- 針對中華民國法律體系優化
- 整合司法院公開資料格式
- 支援繁體中文法律文書處理

## 🚀 快速開始

### 系統需求

```
Python >= 3.8
```

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from legal_auto_prompting import LegalAutoPrompting, LegalCase

# 1. 準備案例資料
case1 = LegalCase(
    case_id="CASE001",
    case_type="偽造文書",
    case_content="某甲偽造公司印章製作假合約...",
    verdict="有罪",
    sentence="有期徒刑6個月",
    hidden_factors=["偽造意圖明確", "金額較小", "初犯"]
)

case2 = LegalCase(
    case_id="CASE002", 
    case_type="偽造文書",
    case_content="某乙偽造政府公文獲取不當利益...",
    verdict="有罪",
    sentence="有期徒刑2年",
    hidden_factors=["偽造政府文書", "金額龐大", "累犯"]
)

# 2. 初始化系統
auto_prompting = LegalAutoPrompting(llm_model=your_llm_model)

# 3. 執行分析
results = auto_prompting.auto_prompting_algorithm(
    case_pair=(case1, case2),
    mixed_judgments="兩案例均涉及偽造文書，但刑期差異顯著",
    label="識別影響偽造文書刑期的關鍵隱藏要件",
    max_iterations=4
)

# 4. 查看結果
for i, result in enumerate(results, 1):
    print(f"第 {i} 次迭代相似度分數: {result.similarity_score:.3f}")
    print(f"提取的隱藏要件: {result.hidden_factors_extracted}")
```

## 📖 詳細文檔

### 核心類別

#### `LegalCase`
表示法律案例的資料結構

```python
@dataclass
class LegalCase:
    case_id: str          # 案例編號
    case_type: str        # 案例類型（偽造文書、詐欺等）
    case_content: str     # 判決書內容
    verdict: str          # 判決結果
    sentence: str         # 刑期
    hidden_factors: List[str]  # 已知隱藏要件
```

#### `LegalAutoPrompting`
主要的系統類別

##### 主要方法

**`auto_prompting_algorithm(case_pair, mixed_judgments, label, max_iterations=4)`**
- **功能**：執行自動提示優化演算法
- **參數**：
  - `case_pair`: 要比較的案例對
  - `mixed_judgments`: 混合判斷結果
  - `label`: 目標標籤
  - `max_iterations`: 最大迭代次數
- **回傳**：`List[PromptOptimizationResult]`

**`evaluate_performance(results)`**
- **功能**：評估演算法性能
- **參數**：`results` - 優化結果列表
- **回傳**：包含各項性能指標的字典

### 配置選項

#### 語言模型整合

```python
class YourLLMModel:
    def generate(self, prompt: str) -> str:
        # 實作您的LLM調用邏輯
        return "模型回應"

# 使用自訂模型
auto_prompting = LegalAutoPrompting(llm_model=YourLLMModel())
```

#### 擴展案例類型

```python
# 在 _get_standard_factors 方法中添加新的案例類型
standard_factors_db = {
    "偽造文書": [...],
    "詐欺": [...],
    "您的新案例類型": [
        "新的隱藏要件1",
        "新的隱藏要件2",
        # ...
    ]
}
```

## 📊 性能指標

系統提供以下性能評估指標：

- **最終相似度分數**：與標準隱藏要件的匹配程度
- **平均相似度分數**：所有迭代的平均表現
- **改進率**：從第一次到最後一次迭代的改進幅度
- **最佳迭代**：表現最好的迭代輪次
- **收斂狀態**：是否達到預設的收斂標準（0.8以上）

## 🗂️ 目錄結構

```
legal-auto-prompting/
├── legal_auto_prompting.py    # 主要系統代碼
├── README.md                  # 本文檔
├── requirements.txt           # 依賴列表
├── examples/                  # 使用範例
│   ├── basic_usage.py
│   └── advanced_usage.py
├── data/                      # 資料目錄
│   ├── sample_cases.json
│   └── standard_factors.json
├── tests/                     # 測試文件
│   ├── test_core.py
│   └── test_evaluation.py
└── docs/                      # 詳細文檔
    ├── api_reference.md
    └── case_study.md
```

## 🔧 進階配置

### 自訂提示模板

```python
custom_prompts = {
    "summarize": "您的自訂總結提示模板...",
    "enhancement": "您的自訂增強提示模板...",
    "extraction": "您的自訂提取提示模板..."
}

auto_prompting = LegalAutoPrompting()
auto_prompting.base_prompts.update(custom_prompts)
```

### 調整相似度計算

您可以重寫 `_calculate_similarity` 方法來實作更精確的相似度演算法：

```python
class CustomLegalAutoPrompting(LegalAutoPrompting):
    def _calculate_similarity(self, extracted_factors, standard_factors):
        # 實作您的自訂相似度計算邏輯
        return similarity_score
```

## 📋 待辦事項

- [ ] 整合更多語言模型支援（GPT-4, Claude, Gemini等）
- [ ] 擴充更多案例類型的標準要件庫
- [ ] 實作更精確的NLP隱藏要件提取演算法
- [ ] 添加視覺化分析工具
- [ ] 支援批量案例處理
- [ ] 建立Web介面

## 🤝 貢獻指南

我們歡迎任何形式的貢獻！請遵循以下步驟：

1. **Fork** 本專案
2. 創建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 **Pull Request**

### 程式碼規範

- 使用 Python 3.8+ 語法
- 遵循 PEP 8 程式碼風格
- 添加適當的註解和文檔字串
- 為新功能編寫測試

## 📄 授權條款

本專案採用 MIT 授權條款。詳細內容請參閱 [LICENSE](LICENSE) 文件。

## 📞 聯絡資訊

如有任何問題或建議，請透過以下方式聯絡：

- **Issues**：[GitHub Issues](https://github.com/your-repo/legal-auto-prompting/issues)
- **Email**：your-email@example.com

## 🙏 致謝

- 感謝司法院提供公開的判決書資料
- 感謝所有貢獻者的寶貴建議和程式碼貢獻
- 本研究受到 [研究機構/計畫名稱] 的支持

## 📚 相關資源

- [司法院法學資料檢索系統](https://law.judicial.gov.tw/)
- [相關學術論文]()
- [法律AI研究社群]()

---

**注意**：本系統僅供學術研究使用，不應作為實際法律判決的依據。任何法律問題請諮詢專業律師。