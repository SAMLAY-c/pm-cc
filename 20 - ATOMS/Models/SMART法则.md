# 思维模型：SMART法则

**标签**: #思维模型/执行 #目标管理

**来源**: [[产品经理思维模型MOC]]

---

> [!abstract] 核心概念
> 一个优秀的目标必须是具体的 (Specific)、可衡量的 (Measurable)、可实现的 (Achievable)、相关的 (Relevant) 和有时限的 (Time-bound)。

---

### 质询与思辨 (Interrogation & Reflection)

>[!question] 我的质询
>- **对于AI产品，"M-可衡量"到底指什么？** 是模型指标（如精度、召回率）？还是用户价值指标（如采纳率、满意度）？或是商业指标（如调用成本降低）？在 `[[项目A：智能BI助手]]` 中，我们当前的衡量指标是哪个？是否合理？
>- **"A-可实现"的陷阱是什么？** 技术上可实现，但数据获取成本极高，算"可实现"吗？一个需要巨大算力支持的目标，对我们目前的资源来说是"可实现"的吗？
>- **"R-相关"如何避免自嗨？** 我的产品目标，如何确保与 `[[公司Q4战略目标：降本增效]]` 真正相关，而不是看起来相关？

---

### 我的应用场景 (My Application Scenarios)

这个模型可以帮我：
- **定义PRD中的需求目标**: 在写每个PRD时，强制用SMART检查核心目标。 #PRD #需求分析
- **制定我的个人OKR**: 将模糊的"提升能力"转化为 `[[我的Q4个人OKR]]`。
- **进行项目复盘**: 评估 `[[项目B：AIGC图片生成功能]]` 的初始目标是否符合SMART，为什么会偏离？ #项目复盘

---

### 我的实践案例 (My Practical Cases)

> [!tip] 使用Dataview自动聚合
> 这里会自动列出所有链接到 [[SMART法则]] 的项目笔记和周报。

```dataview
TABLE task_goal as "我设定的SMART目标", outcome as "最终结果"
FROM #项目 OR #周报 
WHERE contains(file.inlinks, this.file.link)
```