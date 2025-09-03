# Obsidian Canvas JSON 格式规范

## 标准模板

文件名：`canvas.canvas`（扩展名必须是 `.canvas`）

```json
{
  "nodes": [
    {
      "id": "node-id-1",
      "type": "text",
      "text": "这是一个文本卡片",
      "x": 100,
      "y": 100,
      "width": 250,
      "height": 150,
      "color": "1"
    },
    {
      "id": "node-id-2",
      "type": "file",
      "file": "笔记文件名.md",
      "x": 400,
      "y": 100,
      "width": 300,
      "height": 200
    },
    {
      "id": "node-id-3",
      "type": "link",
      "url": "https://obsidian.md",
      "x": 100,
      "y": 300,
      "width": 300,
      "height": 180
    },
    {
      "id": "node-id-4",
      "type": "image",
      "file": "assets/image.png",
      "x": 450,
      "y": 350,
      "width": 200,
      "height": 200
    }
  ],
  "edges": [
    {
      "id": "edge-id-1",
      "fromNode": "node-id-1",
      "fromSide": "right",
      "toNode": "node-id-2",
      "toSide": "left",
      "label": "连接说明文字"
    }
  ]
}
```

## 字段说明（JSON 结构详解）

| 字段名 | 类型 | 说明 |
|---|---|---|
| `nodes` | 数组 | 所有卡片节点 |
| `edges` | 数组 | 所有连接线 |

### Node 字段（每个卡片）

| 字段名 | 类型 | 说明 |
|---|---|---|
| `id` | 字符串 | 卡片唯一标识符 |
| `type` | 字符串 | 卡片类型：`text`、`file`、`link`、`image`、`group` |
| `text` | 字符串 | 仅 `type=text` 时使用，卡片内容 |
| `file` | 字符串 | 仅 `type=file` 或 `type=image` 时使用，对应 Vault 中的文件路径 |
| `url` | 字符串 | 仅 `type=link` 时使用，网页链接 |
| `x`, `y` | 数字 | 卡片左上角坐标 |
| `width`, `height` | 数字 | 卡片宽高 |
| `color` | 字符串 | 卡片颜色编号（1-6 对应 Obsidian 内置颜色） |

### Edge 字段（每条连接线）

| 字段名 | 类型 | 说明 |
|---|---|---|
| `fromNode` | 字符串 | 起点卡片 ID |
| `toNode` | 字符串 | 终点卡片 ID |
| `fromSide`, `toSide` | 字符串 | 连接方向：`top`、`bottom`、`left`、`right` |
| `label` | 字符串 | 连接线上的文字说明 |

## 使用方法

1. 新建一个文本文件，粘贴上述 JSON
2. 保存为 `xxx.canvas`（扩展名必须是 `.canvas`）
3. 放入你的 Obsidian Vault 根目录或任意子目录
4. 打开 Obsidian，即可在文件列表中看到并打开这个 Canvas

## 支持的卡片类型

### 1. 文本卡片 (`type: "text"`)
```json
{
  "id": "text-node",
  "type": "text",
  "text": "这是文本内容",
  "x": 100,
  "y": 100,
  "width": 200,
  "height": 100
}
```

### 2. 文件卡片 (`type: "file"`)
```json
{
  "id": "file-node",
  "type": "file",
  "file": "path/to/note.md",
  "x": 100,
  "y": 100,
  "width": 300,
  "height": 200
}
```

### 3. 链接卡片 (`type: "link"`)
```json
{
  "id": "link-node",
  "type": "link",
  "url": "https://example.com",
  "x": 100,
  "y": 100,
  "width": 300,
  "height": 180
}
```

### 4. 图片卡片 (`type: "image"`)
```json
{
  "id": "image-node",
  "type": "image",
  "file": "path/to/image.png",
  "x": 100,
  "y": 100,
  "width": 200,
  "height": 200
}
```

### 5. 分组卡片 (`type: "group"`)
```json
{
  "id": "group-node",
  "type": "group",
  "label": "分组名称",
  "x": 100,
  "y": 100,
  "width": 400,
  "height": 300
}
```

## 颜色编号对照

| 编号 | 颜色 |
|---|---|
| "1" | 灰色 |
| "2" | 红色 |
| "3" | 橙色 |
| "4" | 黄色 |
| "5" | 绿色 |
| "6" | 蓝色 |