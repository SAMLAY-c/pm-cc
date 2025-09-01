# Obsidian日程管理系统设置指南

## 插件安装与配置

### 1. 必需插件
- **Tasks**: 任务管理核心插件
- **Dataview**: 数据查询和自动化引擎
- **Templater**: 模板动态生成（已安装）
- **Full Calendar**: 可视化日历视图（可选）
- **Kanban**: 看板视图（可选）

### 2. Tasks插件配置
在设置中启用以下功能：
- ✅ 自动完成链接
- ✅ 在任务中显示日期
- ✅ 支持自定义元数据

### 3. 任务原子化格式约定

```markdown
- [ ] 任务描述 📅 YYYY-MM-DD priority:: high/medium/low  проекта [[关联项目笔记]]
```

**字段说明：**
- `📅 YYYY-MM-DD`: 截止日期
- `priority:: high/medium/low`: 优先级
- `проекта [[...]]`: 关联项目（关键链接）

**示例：**
```markdown
```

## Dataview自动化代码片段

### 今日待办 (Today's Briefing)
```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due.toISODate() === dv.date('today').toISODate())
    .sort(t => t.priority, 'desc'), false)
```

### 本周作战计划 (This Week's Plan)
```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due.weekyear === dv.date('today').weekyear && t.due.weekNumber === dv.date('today').weekNumber)
    .groupBy(t => t.due.toFormat("DDDD, yyyy-MM-dd"))
    .sort(g => g.key, 'asc'), false)
```

### 逾期警报 (Overdue Alerts)
```dataviewjs
dv.taskList(dv.pages().file.tasks
    .where(t => !t.completed && t.due && t.due < dv.date('today'))
    .sort(t => t.due, 'asc'), false)
```

### 项目任务聚合 (Project Tasks)
```dataviewjs
const projectName = "项目C：利用AI进行周报摘要";
dv.taskList(dv.pages().file.tasks
    .where(t => t.text.includes(projectName) && !t.completed)
    .sort(t => t.due, 'asc'), false)
```