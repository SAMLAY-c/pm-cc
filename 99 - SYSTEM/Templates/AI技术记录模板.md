# AI技术记录模板

## 技术概览
> [!info] 基本信息
- **技术名称**: [[技术名称]]
- **技术领域**: [[领域分类]]
- **发现日期**: <% tp.date.now("YYYY-MM-DD") %>
- **技术成熟度**: #概念阶段 #实验阶段 #应用阶段 #成熟阶段

## 核心特性
> [!quote] 技术亮点
- 主要功能和技术优势
- 与现有技术的区别
- 潜在应用场景

## 深度质询
> [!question] 我的质询
- 这个技术解决了什么具体问题？
- 在我们的产品中如何应用？
- 技术实现的复杂度如何？
- 与现有技术栈的兼容性？

## 潜在风险
> [!danger] 风险评估
- 技术成熟度风险
- 实施难度风险
- 成本控制风险
- 团队技能要求

## 应用联想
> [!idea] 应用场景
- 可以应用到[[现有项目A]]的什么环节？
- 与[[现有技术B]]的结合点？
- 需要在[[相关会议]]中讨论吗？

## 实验计划
> [!todo] 行动项
- [ ] 技术调研和验证
- [ ] 小规模原型测试
- [ ] 与[[相关人员]]讨论可行性
- [ ] 在[[项目文档]]中记录应用方案

## 相关资源
- **官方文档**: 
- **技术博客**: 
- **开源项目**: 
- **相关论文**: 

## 标签
#AI技术 #技术调研 #创新应用

---

## 动态连接

```dataview
TABLE file.mtime as "最后更新", tags as "标签"
FROM #AI技术
WHERE contains(file.inlinks, this.file.link)
SORT file.mtime DESC
```

## 应用追踪

```dataview
TABLE length(file.outlinks) as "引用数"
FROM #AI技术
WHERE contains(file.inlinks, this.file.link)
SORT length(file.outlinks) DESC
```