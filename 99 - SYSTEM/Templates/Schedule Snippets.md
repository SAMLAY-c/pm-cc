# Schedule Snippets - 日程管理代码片段

> 这些Dataview代码片段可以在任何地方调用，自动生成动态任务列表

---

## 📋 今日待办 (Today's Briefing)

```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due.toISODate() === dv.date('today').toISODate())
    .sort(t => t.priority, 'desc'), false)
```

---

## 📅 本周作战计划 (This Week's Plan)

```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due.weekyear === dv.date('today').weekyear && t.due.weekNumber === dv.date('today').weekNumber)
    .groupBy(t => t.due.toFormat("DDDD, yyyy-MM-dd"))
    .sort(g => g.key, 'asc'), false)
```

---

## ❗ 逾期警报 (Overdue Alerts)

```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due < dv.date('today'))
    .sort(t => t.due, 'asc'), false)
```

---

## 🎯 高优先级任务

```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.priority && t.priority.includes('high'))
    .sort(t => t.due, 'asc'), false)
```

---

## 📊 项目任务统计

```dataview
TABLE 
  length(filter(rows.file.tasks, (t) => !t.completed)) as "待办",
  length(filter(rows.file.tasks, (t) => t.completed)) as "已完成",
  length(rows.file.tasks) as "总数"
FROM #项目
```

---

## 🏷️ 按标签分类的任务

```dataviewjs
const tags = ['#项目管理', '#用户研究', '#产品迭代', '#个人成长'];
for (let tag of tags) {
    dv.header(3, tag);
    dv.taskList(dv.pages()
        .filter(p => p.file.tags && p.file.tags.includes(tag))
        .file.tasks
        .where(t => !t.completed && t.due)
        .sort(t => t.due, 'asc'), false);
}
```

---

## 🔍 搜索特定项目的任务

```dataviewjs
const searchTerm = "项目C"; // 修改为你的项目名称
dv.taskList(dv.pages().file.tasks
    .where(t => t.text.includes(searchTerm) && !t.completed)
    .sort(t => t.due, 'asc'), false)
```

---

## 📈 任务完成趋势（最近7天）

```dataviewjs
const tasks = dv.pages().file.tasks
    .where(t => t.completed && t.completionDate)
    .where(t => t.completionDate >= dv.date('today').minus({days: 7}));

const dailyCompletions = {};
for (let i = 0; i < 7; i++) {
    const date = dv.date('today').minus({days: i}).toFormat('yyyy-MM-dd');
    dailyCompletions[date] = 0;
}

tasks.forEach(t => {
    const date = t.completionDate.toFormat('yyyy-MM-dd');
    if (dailyCompletions.hasOwnProperty(date)) {
        dailyCompletions[date]++;
    }
});

dv.table(['日期', '完成任务数'], 
    Object.entries(dailyCompletions)
        .sort((a, b) => a[0].localeCompare(b[0]))
        .reverse()
);
```

---

## 使用说明

1. **复制粘贴**: 将需要的代码片段复制到任何笔记中
2. **自动更新**: Dataview会自动实时更新结果
3. **自定义**: 修改搜索条件、日期范围等参数
4. **组合使用**: 在指挥中心Canvas中组合多个片段

> **提示**: 将这些代码片段嵌入到你的每日笔记或项目看板中，实现自动化任务管理！