# 什么是Token

> [!info] **核心概念**：大语言模型处理文本的基本单位

##  Token的基本定义

**Token是LLM理解和处理文本的最小单元**，它可以是：
- 一个完整的单词
- 一个词的一部分（subword）
- 一个单独的字符
- 一个标点符号

### 示例说明
```
原始文本: "Hello, world!"
Token化: ["Hello", ",", "world", "!"]
```

## 🔧 Token化的作用

### 1. 文本预处理
- 将连续文本转换为离散的数字表示
- 为模型提供可处理的输入格式
- 控制模型的输入长度限制

### 2. 词汇表管理
- 建立Token到ID的映射关系
- 处理未知词汇（UNK token）
- 管理特殊Token（如[CLS], [SEP]）

### 3. 语义理解基础
- 相似语义的Token具有相近的向量表示
- 支持上下文相关的语义理解
- 为Attention机制提供计算基础

## 💰 Token的成本影响

### 计算成本
- **输入Tokens**: 用户输入的文本量
- **输出Tokens**: 模型生成的文本量
- **总Tokens**: 输入 + 输出的总和

### 成本计算示例
```python
# 假设价格：$0.002 / 1K tokens
输入tokens: 1000
输出tokens: 500
总成本: (1000 + 500) / 1000 * $0.002 = $0.003
```

## 📏 Token与字符的关系

### 转换比率
- **英文**: 1 Token ≈ 4个字符
- **中文**: 1 Token ≈ 1-2个汉字
- **代码**: 1 Token ≈ 3-6个字符

### 上下文长度限制
- GPT-3.5: 4K tokens
- GPT-4: 8K/32K tokens
- Claude: 100K tokens

## 🎯 产品经理关注点

### 用户体验优化
- 合理设置输入提示长度
- 控制回答长度和详细程度
- 优化API调用成本

### 功能设计考虑
- 长文本处理的分段策略
- 上下文记忆的管理
- Token耗尽的处理机制

## 🔗 相关概念

- [[Token知识 - AI产品经理版]] - 产品视角的Token应用
- [[图像Token化]] - 视觉数据的Token化技术
- [[LLM完整生命周期]] - Token在模型训练和推理中的作用
- [[大模型关键技术栈]] - Token在整体技术架构中的位置

## 📚 进一步学习

### 推荐阅读
- "The Tokenization Handbook" - 详细介绍各种Tokenization方法
- "Understanding BERT Tokenization" - BERT模型的WordPiece技术
- "GPT Tokenization Explained" - GPT系列的Byte Pair Encoding

### 实践工具
- **TikToken**: OpenAI官方Token计算器
- **HuggingFace Tokenizers**: 开源Tokenization工具
- **在线Token计数器**: 快速估算Token数量

---

*标签：#基础概念 #LLM核心 #AI产品经理*
*相关项目：[[AI产品经理技术栈项目]]*
*学习状态：#基础概念 ✅*