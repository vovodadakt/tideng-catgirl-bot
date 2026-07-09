# 搜索管线

猫娘 AI 模型的搜索问答模块。合并后的模型同时用于关键词提取和猫娘回答。

## 架构

```
用户问题 -> 关键词提取 -> 百度百科(Browser) + 萌娘百科(API) -> <500c? -> Bing -> 猫娘回答
```

## 运行

```bash
python inference/search/search_pipeline.py
```

## 搜索源

| 来源 | 方式 | 说明 |
|------|------|-------------|
| 百度百科 | Playwright 浏览器 | 绕过服务器 403 |
| 萌娘百科 | API | ACG 知识 |
| Bing | Playwright 深度阅读 | R1 < 500c 时触发 |

## 版本

- **V22**: 双模型（基膜提取关键词 + LoRA 回答），最稳定
- **V24**: 单模型 + CPU offload，VRAM 优化
