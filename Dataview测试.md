# Dataview测试页面

## 测试MVP原则的Dataview查询

```dataview
TABLE 项目名称 as "相关项目", MVP范围 as "最小功能", 验证结果 as "学习成果"
FROM #项目 
WHERE contains(file.inlinks, [[MVP原则]])
```

## 测试SMART法则的Dataview查询

```dataview
TABLE task_goal as "我设定的SMART目标", outcome as "最终结果"
FROM #项目 
WHERE contains(file.inlinks, [[SMART法则]])
```

## 测试WBS任务分解法的Dataview查询

```dataview
TABLE 项目名称 as "相关项目", 分解层级 as "WBS深度", 任务数量 as "任务数"
FROM #项目 
WHERE contains(file.inlinks, [[WBS任务分解法]])
```

## 测试SWOT分析法的Dataview查询

```dataview
TABLE 项目名称 as "相关项目", 分析对象 as "SWOT对象", 战略决策 as "制定策略"
FROM #项目 
WHERE contains(file.inlinks, [[SWOT分析法]])
```

## 测试所有包含思维模型链接的项目

```dataview
TABLE file.link as "项目", related_models as "应用的思维模型"
FROM #项目
WHERE related_models != null
```

## 简单测试：显示所有链接到MVP原则的文件

```dataview
LIST file.link
FROM [[MVP原则]]
```

## 简单测试：显示所有项目文件

```dataview
LIST file.link
FROM #项目
```