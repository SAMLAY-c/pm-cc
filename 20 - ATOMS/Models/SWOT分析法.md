# 思维模型：SWOT分析法

**标签**: #思维模型/分析 #战略规划

**来源**: [[产品经理思维模型MOC]]

---

> [!abstract] 核心概念
> 分析内部优势(Strengths)、劣势(Weaknesses)和外部机会(Opportunities)、威胁(Threats)，制定战略决策。

---

### 质询与思辨 (Interrogation & Reflection)

>[!question] 我的质询
>- **AI创业公司的SWOT分析重点是什么？** 技术团队算优势还是成本？数据积累是优势还是负担？
>- **如何识别AI领域的真正机会？** 技术突破是机会，但监管风险可能是更大的威胁？
>- **SWOT分析在快速变化的AI行业如何保持时效性？** 一个月前的分析可能已经过时，如何动态更新？

---

### 我的应用场景 (My Application Scenarios)

这个模型可以帮我：
- **产品战略规划**: 分析产品在市场中的竞争地位。 #产品策略
- **职业发展规划**: 评估个人职业发展的优劣势和机会。 #个人成长
- **项目风险评估**: 识别项目面临的内外部风险。 #风险管理

---

### 我的实践案例 (My Practical Cases)

```dataview
TABLE 项目名称 as "相关项目", 分析对象 as "SWOT对象", 战略决策 as "制定策略"
FROM #项目 OR #会议
WHERE contains(file.inlinks, this.file.link)
```