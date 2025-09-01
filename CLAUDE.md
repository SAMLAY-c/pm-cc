# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an Obsidian vault designed as a "cognitive enhancement engine" for product management work, specifically focused on AI product development. The vault is structured around a workflow that transforms information into actionable insights through systematic interrogation and connection.

## Core Philosophy

**From Information Storage to Insight Creation**: The primary purpose is not to store information, but to process and interrogate it to create unique insights. The workflow emphasizes:
- Never paste AI content directly - always interrogate and break down into atomic components
- Create bidirectional links between all concepts
- Use structured templates for consistent knowledge processing
- Leverage automation through Dataview and Templater plugins

## Directory Architecture

The vault follows a process-driven organization:

- **00 - INBOX**: Single entry point for all incoming information (must be processed daily)
- **10 - COCKPIT**: Daily workspace with command center and time-based organization
- **20 - ATOMS**: Processed knowledge components (Concepts, People, Sources, Models)
- **30 - INCUBATOR**: Space for insight generation through MOCs and Canvas work
- **40 - OPERATIONS**: Active projects and meetings with bidirectional links to knowledge
- **99 - SYSTEM**: Templates and configuration files

## Required Plugin Configuration

### Core Plugins (must be installed and configured)
- **Tasks**: Task management with specific format requirements
- **Dataview**: Automation engine for dynamic queries
- **Templater**: Template generation with JavaScript support
- **Full Calendar**: Visual calendar integration (optional)
- **Kanban**: Visual task management (optional)

### Tasks Plugin Configuration
```
Task Format: - [ ] Task description 📅 YYYY-MM-DD priority:: high/medium/low  проекта [[Project Link]]
Required Settings:
- Auto-complete links: enabled
- Show due dates: enabled
- Support custom metadata: enabled
```

### Dataview Plugin Configuration
```
Required Settings:
- Enable JavaScript Queries: enabled
- Enable Inline Field Parsing: enabled
- Refresh Interval: 1000ms
```

## Knowledge Processing Workflow

### Step 1: Interrogation Process
When processing new information, never paste directly. Instead:

#### **Phase 1: Create Core Topic Note**
Use the **核心议题模板** to structure initial thinking:
````markdown
# [[核心议题名称]]

## AI的论点拆解
- [[论点1：XXX]]
- [[论点2：XXX]] 
- [[论点3：XXX]]

## 核心问题
> [!question] 我需要解决什么问题？

## 相关背景
> [!info] 这个议题的背景是什么？

## 初步想法
> [!idea] 我的初步思考方向

## 关联项目/人员
- [[项目A]]
- [[同事B]]
- [[会议C]]
````

#### **Phase 2: Deep Interrogation of Each Argument**
For each AI argument, create a dedicated note using the **论点质询模板**:
````markdown
# [[论点名称]]

## AI的观点
> [!quote] AI的观点
> (AI关于这个论点的核心解释，1-2句就够了)

## 我的质询
> [!question] 我的质询
> 质疑AI的观点，关联到具体工作场景
> - 这个观点在我们的XX场景下成立吗？
> - 定义是什么？衡量标准是什么？

## 潜在风险
> [!danger] 潜在风险
> - 如果这个观点是错的，会怎样？
> - 实际应用中可能遇到什么问题？
> - 我们的现有流程能否支持？

## 我的联想
> [!idea] 我的联想
> - 这让我想起了什么？
> - 与其他项目、人员、会议的关联
> - 可以与[[XX项目]]结合吗？
> - 需要在[[下次会议]]中讨论吗？

## 行动项
> [!todo] 需要做什么
> - [ ] 与[[相关人员]]讨论
> - [ ] 调研[[相关技术]]
> - [ ] 在[[项目文档]]中记录
````

#### **Phase 3: Create Thinking Canvas**
Use the **思维碰撞Canvas模板** to visualize connections:
```
[核心议题笔记] 
    ↓
[论点1] ← → [论点2]
    ↓         ↓
[风险分析] ← → [机会分析]
    ↓         ↓
[相关项目] ← → [人员关联]
    ↓         ↓
[行动项1] ← → [行动项2]
```

### Step 2: Command Center Usage
The `My Command Center.md` file serves as the central dashboard with automated Dataview queries for:
- Overdue alerts
- Today's tasks
- Weekly planning
- Project status overview

### Step 3: Template System
All content creation uses standardized templates located in `99 - SYSTEM/Templates/`:

#### **Core Templates**
- **Daily notes**: Cognitive workflow with morning briefing and evening review
- **Meeting notes**: Structured with action items and decisions
- **Project notes**: Full project lifecycle with model integration
- **PRD documents**: Product requirements with technical specifications

#### **Thinking Templates (from TEMPLATES.md)**
- **核心议题模板**: For structuring initial thinking on complex topics
- **论点质询模板**: For deep interrogation of AI arguments and viewpoints
- **思维碰撞Canvas模板**: For visualizing connections between ideas
- **每日思考检查模板**: For daily reflection and insight tracking
- **项目关联模板**: For linking projects to related knowledge and stakeholders

#### **Template Usage Guidelines**
1. **Template Invocation**: Set up template shortcuts in Obsidian
2. **Naming Convention**: Use double-link format `[[名称]]` consistently
3. **Tag Usage**: Add tags like `#质询中` `#已洞察` to core topics
4. **Regular Review**: Weekly check of relationship graphs for new connections

**Remember**: Templates are thinking scaffolds, not constraints. Adjust flexibly based on actual needs.

## File Naming Conventions

- **Daily Notes**: `10 - COCKPIT/Daily/YYYY-MM-DD.md`
- **Meeting Notes**: `40 - OPERATIONS/Meetings/YYYY-MM-DD - Meeting Name.md`
- **Project Notes**: `40 - OPERATIONS/Projects/Project Name.md`
- **Thinking Models**: `20 - ATOMS/Models/Model Name.md`
- **MOCs**: `30 - INCUBATOR/MOCs/Topic MOC.md`

## Automation and Integration

### Dataview Queries
The system relies heavily on Dataview for automation. Key queries include:
- Task aggregation by date and priority
- Project progress tracking
- Model application tracking
- Cross-reference analysis

### Templater Scripts
JavaScript templates in `99 - SYSTEM/Templates/` automate:
- Daily note creation with date-based links
- Meeting note generation with prompt-based input
- Project note creation with standard structure

## Content Creation Rules

1. **No Direct Pasting**: Never paste AI content directly - always interrogate and break down
2. **Atomic Links**: All concepts must be linked bidirectionally
3. **Context Integration**: Every piece of information must connect to specific work contexts
4. **Structured Interrogation**: Use the callout template for all knowledge processing
5. **Daily Processing**: Empty INBOX daily and process into atomic components

## Key Files to Understand

- `OBSIDIAN_RULES.md`: Core thinking methodology and workflow rules
- `10 - COCKPIT/My Command Center.md`: Central dashboard with automation
- `99 - SYSTEM/Templates/`: All template files and automation scripts
- `30 - INCUBATOR/MOCs/产品经理思维模型MOC.md`: Example of a mature knowledge map

## Intelligent Content Creation Workflow

When a user asks a question or requests help, follow this intelligent workflow to determine the appropriate response and content creation strategy:

### Step 1: Analyze User Intent
Identify the user's primary need from their query:

#### **Meeting/Event Related** → Create Meeting Note
**Indicators**: "会议", "meeting", "讨论", "discuss", "沟通", "同步", "review"
- **Location**: `40 - OPERATIONS/Meetings/YYYY-MM-DD - Meeting Name.md`
- **Template**: `99 - SYSTEM/Templates/会议模板.md`
- **Key Elements**: Date, time, participants, agenda, action items, decisions

#### **Project/Task Related** → Create Project Note
**Indicators**: "项目", "project", "功能", "feature", "需求", "requirement", "开发", "develop"
- **Location**: `40 - OPERATIONS/Projects/Project Name.md`
- **Template**: `99 - SYSTEM/Templates/项目模板.md`
- **Key Elements**: Goals, timeline, tasks, stakeholders, applied models

#### **Knowledge/Learning Related** → Create Thinking Process
**Indicators**: "学习", "learn", "理解", "understand", "解释", "explain", "分析", "analyze", "思考", "thinking", "质询", "interrogate"
- **Location**: `30 - INCUBATOR/` (for thinking processes) or `20 - ATOMS/` (for final concepts)
- **Template**: Use thinking templates from `TEMPLATES.md`:
  - **核心议题模板**: For complex topics requiring deep analysis
  - **论点质询模板**: For interrogating AI arguments and viewpoints
  - **思维碰撞Canvas模板**: For visualizing idea connections
- **Key Elements**: AI argument breakdown, structured interrogation, risk analysis, action items, bidirectional links
- **Process**: 
  1. Create core topic note with 核心议题模板
  2. Break down AI arguments into separate 论点质询模板 notes
  3. Create 思维碰撞Canvas to visualize connections
  4. Link to related projects, people, and meetings

#### **Daily/Planning Related** → Create Daily Note
**Indicators**: "今天", "today", "计划", "plan", "安排", "schedule", "任务", "tasks"
- **Location**: `10 - COCKPIT/Daily/YYYY-MM-DD.md`
- **Template**: `99 - SYSTEM/Templates/每日笔记模板.md`
- **Key Elements**: Daily goals, scheduled events, work log, evening review

#### **Strategic Thinking Related** → Create MOC or Thinking Canvas
**Indicators**: "策略", "strategy", "规划", "planning", "梳理", "organize", "总结", "summary", "思维碰撞", "thinking canvas"
- **Location**: `30 - INCUBATOR/MOCs/` (for MOCs) or `30 - INCUBATOR/Canvas/` (for thinking canvases)
- **Template**: 
  - **MOC**: `99 - SYSTEM/Templates/MOC模板.md`
  - **Thinking Canvas**: Use 思维碰撞Canvas模板 from `TEMPLATES.md`
- **Key Elements**: 
  - **MOC**: Topic mapping, connections, dynamic queries
  - **Thinking Canvas**: Visual connection mapping, risk/opportunity analysis, action items
- **Process**: 
  1. Create central topic note
  2. Map related arguments and concepts
  3. Visualize connections and relationships
  4. Generate insights and action items

#### **PRD/Documentation Related** → Create PRD Document
**Indicators**: "PRD", "文档", "document", "需求文档", "产品文档", "说明书", "manual", "规范", "specification"
- **Location**: `40 - OPERATIONS/Projects/[Project Name] - PRD.md`
- **Template**: `99 - SYSTEM/Templates/PRD模板.md`
- **Key Elements**: Product requirements, user stories, acceptance criteria, technical specs

### Step 2: Provide Intelligent Response

Based on the identified intent, provide a comprehensive response that includes:

#### **For Meeting Notes:**
```markdown
我理解你需要记录一个会议。我将为你创建一个结构化的会议纪要，包含：
- 会议基本信息（时间、地点、参会人员）
- 会议目标和讨论要点
- 关键决策和行动项
- 相关项目和概念的链接

创建位置：`40 - OPERATIONS/Meetings/YYYY-MM-DD - [会议名称].md`
```

#### **For Project Notes:**
```markdown
我理解你在启动一个新项目。我将为你创建一个项目笔记，包含：
- 项目目标和背景
- 应用相关的思维模型（如SMART法则）
- 任务分解和时间线
- 相关会议和人员的链接
- 自动化的进度跟踪

创建位置：`40 - OPERATIONS/Projects/[项目名称].md`
```

#### **For Knowledge Processing:**
```markdown
我理解你想深入学习这个概念。我将为你创建一个完整的思考流程，包含：

**Phase 1: 核心议题分析**
- 创建核心议题笔记，结构化AI的论点拆解
- 明确核心问题和相关背景
- 建立初步想法和关联关系

**Phase 2: 深度质询过程**
- 为每个AI论点创建专门的质询笔记
- 系统性质疑每个观点，关联到具体工作场景
- 分析潜在风险和实践应用挑战
- 建立与其他项目和人员的关联

**Phase 3: 思维碰撞整合**
- 创建思维碰撞Canvas，可视化所有论点连接
- 整合风险分析和机会识别
- 生成具体行动项和决策建议

**Phase 4: 项目关联**
- 将洞察与现有项目建立链接
- 识别相关人员和会议
- 记录风险评估和机会识别

创建位置：`30 - INCUBATOR/` 和 `20 - ATOMS/`
使用模板：`TEMPLATES.md` 中的完整思考模板体系
```

#### **For Daily Planning:**
```markdown
我理解你需要规划今天的工作。我将为你创建今日认知作战室：
- 晨间校准（三大核心目标）
- 实时战术记录区域
- 信息处理和原子化提醒
- 晚间复盘和洞察合成
- 自动化的反向链接检查

创建位置：`10 - COCKPIT/Daily/YYYY-MM-DD.md`
```

#### **For PRD Documentation:**
```markdown
我理解你需要创建PRD文档。我将为你创建一个结构化的产品需求文档，包含：
- 产品背景和目标
- 用户故事和需求分析
- 功能规格和验收标准
- 技术实现方案
- 项目里程碑和风险评估
- 相关思维模型的应用（如用户故事地图、MVP原则）

创建位置：`40 - OPERATIONS/Projects/[项目名称] - PRD.md`
```

### Step 3: Execute and Create

After providing the explanation, create the content using the appropriate template and:
1. **Use Templater syntax** for dynamic content (dates, links)
2. **Apply the interrogation process** for knowledge content
3. **Create bidirectional links** to related existing content
4. **Include relevant thinking models** from the MOC
5. **Add proper tags** for categorization and filtering
6. **Set up Dataview queries** for automation where applicable

### Step 4: Provide Usage Guidance

After creating the content, explain:
- How to use the created note effectively
- How it integrates with the existing knowledge network
- What automated features are available
- How to maintain and update the content

## 智能日程管理功能

### 用户日程输入与自动分类

当用户输入日程信息时，系统会自动判断是今日任务还是未来待办事项，并添加到相应的位置：

#### **日程输入触发词**
"日程", "安排", "计划", "reminder", "schedule", "todo", "待办", "提醒"

#### **智能分类逻辑**
1. **今日任务** → 添加到当前每日笔记的"今日关键日程"部分
2. **未来待办** → 创建带日期的任务并添加到相应日期的每日笔记

#### **日程处理流程**

**Step 1: 分析日程信息**
- 识别时间关键词（今天、明天、下周、具体日期）
- 判断任务类型（会议、任务、提醒）
- 提取关键信息和优先级

**Step 2: 智能分类**
```javascript
// 日期判断逻辑
if (日程包含今天关键词 || 日期为今天) {
    添加到今日笔记的"今日关键日程"部分
} else if (日程包含未来日期) {
    创建带日期的任务格式：- [ ] 任务描述 📅 YYYY-MM-DD priority:: high/medium/low
    添加到对应日期的每日笔记
}
```

**Step 3: 自动创建链接**
- 会议任务：自动创建会议笔记链接
- 项目任务：链接到相关项目笔记
- 概念任务：链接到相关概念或模型笔记

#### **日程格式示例**

**今日会议日程：**
```markdown
- [[YYYY-MM-DD - 产品评审会议]] 14:00-15:00
- [[YYYY-MM-DD - 技术方案讨论]] 16:00-17:00
```

**未来待办任务：**
```markdown
- [ ] 完成产品需求文档撰写 📅 2024-01-15 priority:: high  проекта [[产品优化项目]]
- [ ] 准备季度汇报材料 📅 2024-01-20 priority:: medium  проекта [[Q4汇报]]
```

#### **Templater脚本支持**

创建 `99 - SYSTEM/Templates/日程处理模板.md`：
```markdown
<%*
// 获取用户输入的日程信息
const scheduleInput = await tp.system.prompt("请输入日程信息（包含时间和任务描述）：");

// 简单的日期识别逻辑
const today = tp.date.now("YYYY-MM-DD");
const tomorrow = tp.date.now("YYYY-MM-DD", 1);

let targetDate = today;
let taskDescription = scheduleInput;

// 识别日期关键词
if (scheduleInput.includes("明天") || scheduleInput.includes("tomorrow")) {
    targetDate = tomorrow;
} else if (scheduleInput.includes("今天") || scheduleInput.includes("今天")) {
    targetDate = today;
}

// 创建任务格式
const taskFormat = `- [ ] ${taskDescription} 📅 ${targetDate} priority:: medium`;

// 如果是今天，添加到当前笔记
if (targetDate === today) {
    // 添加到今日日程部分
    const currentContent = tp.file.content;
    const updatedContent = currentContent.replace(
        /🗓️ 今日关键日程 \(Scheduled Events\):\*[^-]*- \[\[<% tp\.date\.now\("YYYY-MM-DD"\) %> - 会议名称1\]\]/,
        `🗓️ 今日关键日程 (Scheduled Events):\n- ${taskFormat}\n- [[${today} - 会议名称1]]`
    );
    await tp.file.update(updatedContent);
} else {
    // 创建未来日期的每日笔记
    const futureNotePath = `10 - COCKPIT/Daily/${targetDate}.md`;
    await tp.file.create_new(futureNotePath, targetDate);
}

%>
```

## Working with This System

When asked to create or modify content:
1. **Analyze Intent**: Use the intelligent workflow above to determine content type
2. **Check for Schedule Input**: If user mentions日程/安排/计划等， use the schedule management workflow
3. **Explain Strategy**: Tell the user what you're creating and why
4. **Execute Creation**: Use appropriate templates and follow interrogation process
5. **Provide Guidance**: Explain how to use and maintain the content
6. **Ensure Integration**: Create bidirectional links and automation

The goal is to create a living knowledge network that generates unique insights through systematic processing and connection of information, while providing users with intelligent, context-aware assistance.

## 思考模板使用指南

### 模板选择策略

#### **当面对复杂议题时**
使用 **核心议题模板**：
- 需要拆解AI的多个论点
- 涉及多个相关项目或人员
- 需要系统化思考框架

#### **当需要深入质询时**
使用 **论点质询模板**：
- 对AI的某个观点有疑问
- 需要评估实际应用风险
- 想要建立具体的工作关联

#### **当需要可视化思考时**
使用 **思维碰撞Canvas模板**：
- 多个论点之间有复杂关系
- 需要分析风险和机会
- 要生成具体的行动项

#### **当进行项目整合时**
使用 **项目关联模板**：
- 将思考成果与具体项目结合
- 需要评估项目风险和机会
- 要制定基于洞察的决策

### 模板使用最佳实践

1. **循序渐进**: 从核心议题开始，逐步深入到论点质询
2. **保持链接**: 确保所有相关概念、项目、人员都有双向链接
3. **定期回顾**: 使用每日思考检查模板进行反思和洞察提取
4. **灵活调整**: 根据实际需要调整模板结构，不被框架束缚

### 标签系统

为思考过程添加标签以便追踪：
- `#质询中` - 正在进行深度质询
- `#已洞察` - 已经获得重要洞察
- `#需验证` - 需要实践验证的想法
- `#已应用` - 已经应用到实际工作中

### 思考成果的应用

1. **项目决策**: 将质询结果应用到项目决策中
2. **会议准备**: 用思考成果准备会议讨论要点
3. **知识分享**: 将成熟的洞察分享给团队成员
4. **持续迭代**: 基于实践反馈不断完善思考框架