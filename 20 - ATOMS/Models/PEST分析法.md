# 思维模型：PEST分析法

**标签**: #思维模型/分析 #宏观环境

**来源**: [[产品经理思维模型MOC]]

---

> [!abstract] 核心概念
> 分析政治(Political)、经济(Economic)、社会(Social)和技术(Technical)宏观环境因素，评估市场机会和风险。

---

### 质询与思辨 (Interrogation & Reflection)

>[!question] 我的质询
>- **AI行业的PEST分析有何特殊性？** 技术因素可能比政治因素更重要，如何平衡各因素权重？
>- **如何预测AI监管政策的变化趋势？** 各国对AI的监管态度不同，如何影响产品全球化策略？
>- **社会对AI的接受度如何量化？** 除了调查数据，还有什么方法可以评估社会对AI的真实态度？

---

### 我的应用场景 (My Application Scenarios)

这个模型可以帮我：
- **市场进入策略**: 评估新市场的宏观环境风险。 #市场分析
- **产品路线图规划**: 基于技术趋势制定产品发展路线。 #产品规划
- **风险预警**: 识别可能影响产品的宏观环境变化。 #风险管理

---

### 我的实践案例 (My Practical Cases)

```dataview
TABLE 项目名称 as "相关项目", 分析维度 as "PEST维度", 关键发现 as "洞察"
FROM #项目 OR #概念
WHERE contains(file.inlinks, this.file.link)
```