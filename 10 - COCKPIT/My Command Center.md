# My Command Center

> **动态战略指挥中心** - 集成所有关键信息的作战沙盘

---

## 🚨 逾期警报

```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due < dv.date('today'))
    .sort(t => t.due, 'asc'), false)
```

---

## 🎯 今日待办

```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due.toISODate() === dv.date('today').toISODate())
    .sort(t => t.priority, 'desc'), false)
```

---

## 📅 本周计划

```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due.weekyear === dv.date('today').weekyear && t.due.weekNumber === dv.date('today').weekNumber)
    .groupBy(t => t.due.toFormat("DDDD, yyyy-MM-dd"))
    .sort(g => g.key, 'asc'), false)
```

---

## 🚀 核心项目状态

| 项目 | 状态 | 进度 | 下一步 |
|------|------|------|--------|
| [[项目C：利用AI进行周报摘要]] | #进行中 | 20% | 完成PRD |
| [[项目A：智能BI助手]] | #规划中 | 10% | 需求调研 |
| [[项目B：AIGC图片生成功能]] | #已完成 | 100% | 复盘总结 |

---

## 🧠 思维模型库入口

[[产品经理思维模型MOC]] - 随时调用的思维武器库

---

## 📈 本周复盘

[[2025-W36]] - 本周工作总结与思考

---

## 💡 快速链接

- [[今日笔记]] - 打开今日的Daily Note
- [[99 - SYSTEM/Templates/Schedule Snippets]] - 代码片段库
- [[40 - OPERATIONS/Projects]] - 所有项目概览

---

## 🎨 使用说明

1. **每日早晨**: 首先查看`逾期警报`，处理紧急任务
2. **规划当日**: 基于`今日待办`和`本周计划`制定当日重点
3. **工作中**: 在相应项目笔记中创建新任务，会自动聚合到这里
4. **晚间复盘**: 查看任务完成情况，准备明日计划

> **提示**: 这是一个动态文档，所有数据都会实时更新。建议将其设为Obsidian首页或常用标签页。