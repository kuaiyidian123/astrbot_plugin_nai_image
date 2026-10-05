# AI 生图 (NovelAI/吐司)

AstrBot 插件：发送 `/生图 中文描述`，先由大模型把中文转换成英文绘画 tag，再调用 **NovelAI** 生成二次元插画并发送到聊天。
另支持 `/生图 吐司 中文描述` 改走 **吐司（TAMS / tusi.cn）** 工作流模板接口生图。

> 本插件由 `astrbot_plugin_netease_music`（网易云点歌）的「AI 生图」模块拆分而来，独立成一个插件。

## 功能

- 🎨 `/生图 中文描述`：大模型转英文绘画 tag → NovelAI 生图（支持 V4.5 / V5）
- 🍞 `/生图 吐司 中文描述`：改走吐司（TAMS）工作流模板接口生图
- 🧰 `/生图 吐司API [工具名] 中文描述`：走吐司 OpenAPI（AI Tool），支持文生图/图生图/视频/图像编辑；在消息里附图可作为参考图
- 📋 `/生图工具`：列出吐司 OpenAPI 当前可用的全部 AI 工具
- 📖 `/生图帮助`：查看当前配置、可用转换模型 ID 与今日剩余次数
- 🔢 每人每日生图次数上限，**生图成功才计数**，避免网络/配置问题白扣次数
- 🚦 同一用户生图进行中会拦截重复触发，防止误触烧额度
- 🌐 支持 HTTP 代理，便于访问 NovelAI / 吐司接口

## 安装

1. 将 `astrbot_plugin_nai_image` 目录放入 AstrBot 的 `data/plugins/` 下；
2. 重启 AstrBot 或在 WebUI 插件管理中重载；
3. 在插件配置中填写 NovelAI（或吐司）相关参数后即可使用。

依赖：`aiohttp`（一般 AstrBot 环境已自带）。

## 指令说明

| 指令 | 说明 |
| --- | --- |
| `/生图 中文描述` | 走 NovelAI 生图 |
| `/生图 吐司 中文描述` | 走吐司（TAMS）生图（需先填吐司 API Key 与模板 ID） |
| `/生图 吐司API [工具名] 中文描述` | 走吐司 OpenAPI（需先填 Access Key）；不写工具名时用后台默认工具；附图可作为参考图 |
| `/生图工具` | 列出吐司 OpenAPI 当前可用的全部 AI 工具（名称/用途/算力） |
| `/生图帮助` | 查看用法、当前配置、可用转换模型 ID 与今日剩余次数 |

示例：

```
/生图 蓝发少女站在樱花树下，微笑，逆光
/生图 吐司 赛博朋克风格的机械猫，霓虹灯背景
/生图 吐司API anime_lab_wai_illustrious 樱花树下的猫娘
/生图 吐司API （附图）把这张图变成动漫风格
```

## 配置说明

### 通用

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 开启「/生图」 | 开 | 关闭后 `/生图` 不可用（后台可随时开关） |
| 生图 tag 转换模型 ID | 空 | 用于把中文转成英文绘画 tag 的对话模型 ID；留空用当前默认对话模型。可用 ID 见 `/生图帮助` |
| 每人每日生图次数上限 | 5 | 填 0 表示不限制 |
| 额外负面词 | 空 | 追加到负面提示词的固定内容，多个用英文逗号分隔，如 `extra fingers, watermark` |
| HTTP代理地址 | 空 | 访问生图接口的代理，如 `http://127.0.0.1:7890`，留空不使用 |

### NovelAI

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 生图 API 地址 | `https://image.novelai.net/ai/generate-image` | 官方主机；第三方中转填其中转地址 |
| 生图 API Key | 空 | NovelAI 持久化 Token（形如 `pst-...`） |
| 生图模型 | `nai-diffusion-4-5-full` | 常用：`nai-diffusion-4-5-full` / `nai-diffusion-4-5-curated` / `nai-diffusion-5-full` / `nai-diffusion-3` |
| 生图尺寸 | `832x1216` | 格式 `宽x高`，自动对齐 64 的倍数并限制总像素 |
| 生图步数 | 28 | 1~50，越高越慢，超过 28 步多数档位额外消耗 Anlas |
| 生图 CFG | 5.0 | 提示词引导强度，常用 4~7 |
| 生图采样器 | `k_euler_ancestral` | 不确定用默认 |

> API Key 获取：登录 [novelai.net](https://novelai.net) → 账号设置 → 生成持久化 Token（`pst-...`，注意不是登录用的 access token）。

### 吐司（TAMS）

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 吐司接口地址 | `https://cn.tensorart.net` | 国内站；国际站可用 `https://ap-east-1.tensorart.cloud` |
| 吐司 API Key | 空 | 在 [tams.tusiart.com/apps](https://tams.tusiart.com/apps) 创建应用后生成 |
| 吐司模板(ID) | 空 | 想使用的工作流模板 ID（站点地址中的数字 ID）；留空则 `/生图 吐司` 不可用 |
| 吐司提示词字段名 | 空 | 留空自动识别（匹配含 prompt / 提示词 / 关键词 的字段），失败时手动填写 |
| 吐司负面词字段名 | 空 | 留空自动识别（匹配含 negative / 负面 / 反向 的字段），无该字段会自动跳过 |

### 吐司 OpenAPI（AI Tool）

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 吐司 OpenAPI Access Key | 空 | 在 tusi.cn / tensor.art 个人资料页获取；以 `ak_tusi` 开头自动走国内站 `openapi.tusiart.cn`，其余走 `openapi.tensor.art` |
| 吐司 OpenAPI 接口地址 | 空 | 留空按 Key 前缀自动选择，需要固定站点时手动填写 |
| 吐司 OpenAPI 默认工具 | `anime_lab_wai_illustrious` | `/生图 吐司API 描述` 未指定工具时使用；可发 `/生图工具` 查看全部工具 |

> OpenAPI 渠道的工具是**动态发现**的（`tool/list`），文档里的文生图/视频/图像编辑工具都会出现在 `/生图工具` 里；任务参数由对话模型按工具定义自动填充。

## 常见问题

- **提示「尚未配置 NovelAI API Key」**：在后台插件配置里填写「生图 API Key」。
- **提示「无法连接到生图接口」**：确认 API 地址完整有效；若域名能 ping 通仍失败，多为网络阻断，请在「HTTP代理地址」填写本地代理。
- **转换 tag 失败（额度不足 / 鉴权失败 / 限流）**：提示中会点明原因，可把「生图 tag 转换模型 ID」换成其他可用对话模型。
- **吐司提示「未找到模板」/「没有可填写的参数」**：检查「吐司模板(ID)」是否正确，或在「吐司提示词字段名」手动填写字段名。
