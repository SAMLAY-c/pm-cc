# 产品经理思维模型武器库 MOC

**标签**: #MOC/思维模型

**最后更新**: 2025-09-01

---

这篇文章是我的思维模型库的基础，我的目标不是记住它们，而是将它们内化并应用到我的每一个项目中。

---

### 一、 目标与执行类 (从想到做)
- [[20 - ATOMS/Models/SMART法则]]
- [[20 - ATOMS/Models/WBS任务分解法]]
- [[20 - ATOMS/Models/PDCA循环]]

### 二、 需求与战略类 (做正确的事)
- [[20 - ATOMS/Models/卡诺模型 (Kano Model)]]
- [[20 - ATOMS/Models/马斯洛需求层次理论]]
- [[20 - ATOMS/Models/用户体验五要素]]

### 三、 环境与竞争分析类 (知己知彼)
- [[20 - ATOMS/Models/SWOT分析法]]
- [[20 - ATOMS/Models/PEST分析法]]

### 四、 过程与沟通类 (把事做好)
- [[20 - ATOMS/Models/5W2H分析法]]
- [[20 - ATOMS/Models/MVP原则]]
- [[20 - ATOMS/Models/STAR法则]]
- [[20 - ATOMS/Models/四象限法则]]

---

### 🔗 动态连接

```dataview
TABLE file.mtime as "最后更新", tags as "标签"
FROM #思维模型
WHERE contains(file.inlinks, this.file.link)
SORT file.mtime DESC
```

### 📊 应用统计

```dataview
TABLE length(file.inlinks) as "被引用次数"
FROM #思维模型
WHERE contains(file.inlinks, this.file.link)
SORT length(file.inlinks) DESC
```