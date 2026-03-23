# Knowledge Base Agent

企业内部知识库问答 Agent，基于 RAG + Claude Function Calling 实现智能路由。

## 技术栈

- **Python + FastAPI** — API 服务
- **Anthropic Claude API** — 意图识别（Function Calling）+ 回答生成
- **ChromaDB** — 向量检索（all-MiniLM-L6-v2 embedding）
- **SQLite** — 请求日志与 trace 存储

## 架构

```
用户提问 → Claude Function Calling 意图分类
  ├─ rag_search   → ChromaDB top3 检索 → Claude 生成回答
  ├─ url_fetch    → 实时爬取网页      → Claude 生成回答
  └─ chitchat     → 直接返回闲聊回复
```

每次请求自动写入 SQLite trace 日志，支持评测分析。

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
export ANTHROPIC_API_KEY=sk-ant-xxxxx

# 3. 启动服务
uvicorn app.main:app --reload
```

服务启动后访问 http://localhost:8000/docs 查看 Swagger 文档。

## API 接口

### 数据摄入

```bash
# 上传 PDF/Word 文档
curl -X POST http://localhost:8000/ingest/document \
  -F "file=@员工手册.pdf"

# 导入 URL 内容
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# 查看知识库统计
curl http://localhost:8000/ingest/stats
```

### 知识问答

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的请假制度是什么？"}'
```

返回格式：
```json
{
  "answer": "根据公司制度，年假为每年15天，需提前3个工作日申请。",
  "sources": ["员工手册.pdf"],
  "confidence": "high",
  "fallback": false
}
```

### 日志查询

```bash
# 列表（分页）
curl "http://localhost:8000/traces/?limit=10&offset=0"

# 单条查询
curl http://localhost:8000/traces/{request_id}
```

## 评测

```bash
# 1. 从 trace 日志导入评测集
python eval/evaluate.py import

# 2a. 人工标注（交互式）
python eval/evaluate.py annotate

# 2b. 自动评分（基于 confidence/fallback 启发式）
python eval/evaluate.py auto-score

# 3. 生成评测报告
python eval/evaluate.py report

# 查看状态
python eval/evaluate.py status
```

评测指标：答案准确率（1-5分）、来源命中率、降级触发率、平均延迟。

## 部署到 Railway

1. 将代码推送到 GitHub
2. 在 Railway 中创建项目，连接 GitHub 仓库
3. 设置环境变量 `ANTHROPIC_API_KEY`
4. Railway 自动检测 `Procfile` 并部署

## 项目结构

```
knowledge-base-agent/
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 全局配置
│   ├── models/schemas.py       # Pydantic 模型
│   ├── routers/
│   │   ├── ingest.py           # 数据摄入 API
│   │   ├── query.py            # 问答 API
│   │   └── traces.py           # 日志查询 API
│   └── services/
│       ├── chunker.py          # 文本分块
│       ├── parser.py           # 文档解析 + URL 爬取
│       ├── vectorstore.py      # ChromaDB 向量存储
│       ├── llm.py              # Claude API 调用
│       ├── query_engine.py     # 查询编排引擎
│       ├── response_builder.py # 统一响应构建
│       └── trace_logger.py     # SQLite 日志
├── eval/
│   └── evaluate.py             # 评测脚本
├── requirements.txt
├── Procfile                    # Railway 部署
└── .env.example
```
