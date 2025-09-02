# AI技术武器库 MOC

**标签**: #MOC/AI技术

**最后更新**: <% tp.date.now("YYYY-MM-DD") %>

---

这个MOC是我的AI技术追踪中心，目标不是记录所有技术，而是深度理解并应用到实际项目中。

---

## 🚀 技术分类导航

### 大语言模型 (LLM)
- [[20 - ATOMS/Concepts/AI/LLM基础模型]]
- [[20 - ATOMS/Concepts/AI/模型微调技术]]
- [[20 - ATOMS/Concepts/AI/提示工程]]

### 多模态AI
- [[20 - ATOMS/Concepts/AI/图像识别]]
- [[20 - ATOMS/Concepts/AI/语音处理]]
- [[20 - ATOMS/Concepts/AI/视频理解]]

### AI应用架构
- [[20 - ATOMS/Concepts/AI/RAG系统]]
- [[20 - ATOMS/Concepts/AI/Agent框架]]
- [[20 - ATOMS/Concepts/AI/工作流自动化]]

### 开发工具链
- [[20 - ATOMS/Concepts/AI/AI开发框架]]
- [[20 - ATOMS/Concepts/AI/模型部署工具]]
- [[20 - ATOMS/Concepts/AI/监控与优化]]

---

## 🔍 技术成熟度矩阵

```dataview
TABLE 技术成熟度 as "成熟度", file.mtime as "最后更新"
FROM #AI技术
WHERE 技术成熟度 != null
SORT 技术成熟度 DESC
```

## 📈 应用热度排行

```dataview
TABLE length(file.inlinks) as "被引用次数", tags as "标签"
FROM #AI技术
SORT length(file.inlinks) DESC
LIMIT 10
```

## 🎯 项目应用追踪

```dataview
TABLE 项目名称 as "应用项目", 应用状态 as "状态"
FROM #AI技术
WHERE 项目名称 != null
SORT file.mtime DESC
```

---

## 🔗 动态连接

```dataview
TABLE file.mtime as "最后更新", tags as "标签"
FROM #AI技术
WHERE contains(file.inlinks, this.file.link)
SORT file.mtime DESC
```

## 📊 统计概览

- **总技术数**: 
- **已应用技术**: 
- **实验中技术**: 
- **待调研技术**: 

---

## 🎯 技术调研优先级

### 高优先级 (与当前项目相关)
- [[待调研技术1]]
- [[待调研技术2]]

### 中优先级 (潜在应用)
- [[待调研技术3]]
- [[待调研技术4]]

### 低优先级 (长期关注)
- [[待调研技术5]]
- [[待调研技术6]]

---

## 📝 定期回顾

### 每周回顾
- [ ] 检查新技术发现
- [ ] 更新技术成熟度
- [ ] 记录应用进展

### 每月回顾
- [ ] 评估技术趋势
- [ ] 调整调研优先级
- [ ] 更新项目应用策略

---

## 🤔 思考与洞察

> [!idea] 
> (记录对AI技术发展趋势的洞察)

> [!question] 
> (需要深入研究的核心问题)

---

## 📚 相关资源

### 学习资源
- [[AI学习路径]]
- [[技术博客列表]]
- [[研究论文库]]

### 工具资源
- [[AI开发工具]]
- [[数据集资源]]
- [[社区论坛]]