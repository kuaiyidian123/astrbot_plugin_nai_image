"""
astrbot_plugin_nai_image - AI 生图插件（NovelAI / 吐司）

从 astrbot_plugin_netease_music（网易云点歌）的「AI 生图」模块拆分而来，独立成一个插件：
- /生图 中文描述          → 大模型把中文转英文绘画 tag → 调用 NovelAI 生图
- /生图 吐司 中文描述     → 改走吐司（TAMS / tusi.cn）工作流模板接口
- /生图帮助               → 查看当前配置、可用转换模型与今日剩余次数
"""

import os
import re
import json
import time
import random
import hashlib
import asyncio
import tempfile
import zipfile
import urllib.parse
from io import BytesIO
from typing import Dict, List, Optional, Any

# ============= AstrBot API =============
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger

# ============= 第三方依赖 =============
import aiohttp


# ============ NovelAI 生图（/生图） ============
NAI_DEFAULT_API_URL = "https://image.novelai.net/ai/generate-image"
# 中文描述 -> NovelAI 英文绘画 tag 的系统提示词（规则来自使用者指定）
NAI_TAG_SYSTEM_PROMPT = (
    "你是NovelAI绘画tag生成器，把下面中文描述转换成英文绘画关键词，逗号分隔。\n"
    "输出格式：\n"
    "正向tag：xxx\n"
    "负面tag：xxx\n"
    "规则：\n"
    "1. 只输出绘画关键词，不要完整长句子\n"
    "2. 二次元插画风格，masterpiece, best quality放最前面\n"
    "3. 负面词固定包含：lowres, bad anatomy, bad hands, extra limbs, deformed, blurry, ugly\n"
)
# 固定必须出现的负面词（大模型漏掉时自动补齐）
NAI_REQUIRED_NEGATIVE = [
    "lowres", "bad anatomy", "bad hands", "extra limbs", "deformed", "blurry", "ugly",
]
# 生图总像素上限（超过等比缩小，避免超出 NovelAI 单图上限被拒）
NAI_MAX_PIXELS = 1024 * 3072
NAI_SAMPLER_OPTIONS = [
    "k_euler_ancestral", "k_euler", "k_dpmpp_2m", "k_dpmpp_2s_ancestral",
    "k_dpmpp_sde", "ddim", "k_heun", "k_dpm_2", "k_dpm_2_ancestral",
]
NAI_REQUEST_TIMEOUT = 180          # 生图请求总超时（秒）
NAI_MAX_IMAGE_BYTES = 32 * 1024 * 1024  # 单次响应体上限（32MB），防止异常响应撑爆内存
MAX_NAI_PROMPT_CHARS = 500         # 用户输入的描述长度上限


# ============ 吐司（TAMS / tusi.cn）生图 ============
TUSI_DEFAULT_BASE_URL = "https://cn.tensorart.net"
# 模板里提示词/负面词字段的自动识别关键词（可用配置项手动指定覆盖）
TUSI_PROMPT_KEYS = ("prompt", "提示词", "提示语", "关键词", "描述")
TUSI_NEGATIVE_KEYS = ("negative", "负面", "反向", "负向")
TUSI_API_TIMEOUT = 60                   # 单次接口请求超时（秒）
TUSI_JOB_TIMEOUT = 300                  # 提交后等待作业完成的整体上限（秒）
TUSI_POLL_INTERVAL = 4                  # 作业状态轮询间隔（秒）
TUSI_MAX_JSON_BYTES = 2 * 1024 * 1024   # 接口 JSON 响应上限
TUSI_MAX_IMAGE_BYTES = 32 * 1024 * 1024  # 结果图片上限


def _split_tusi_prefix(text: str) -> tuple:
    """识别「/生图 吐司 xxx」中的渠道前缀，返回 (是否走吐司, 剩余描述)

    「吐司」后必须跟空白或分隔符才算前缀，避免「吐司面包」这类描述被误判。
    """
    raw = (text or "").strip()
    m = re.match(r"^吐司(?:\s+|[,，:：]+\s*)(.+)$", raw)
    if m:
        return True, m.group(1).strip()
    if raw == "吐司":
        return True, ""
    return False, raw


def _parse_nai_size(value, default: tuple = (832, 1216)) -> tuple:
    """解析生图尺寸 "宽x高"（自动对齐到 64 的倍数，并限制总像素），非法时返回默认值"""
    text = str(value or "").strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{2,5})[x*×](\d{2,5})", text)
    if not m:
        return default
    w = max(64, (int(m.group(1)) + 63) // 64 * 64)
    h = max(64, (int(m.group(2)) + 63) // 64 * 64)
    if w * h > NAI_MAX_PIXELS:
        ratio = (NAI_MAX_PIXELS / (w * h)) ** 0.5
        w = max(64, int(w * ratio) // 64 * 64)
        h = max(64, int(h * ratio) // 64 * 64)
    return (w, h)


def _describe_llm_error(err: Exception) -> str:
    """把对话模型的报错转成可读提示（额度不足/鉴权失败/限流等常见原因单独点出）"""
    detail = " ".join(str(err or "").split())
    low = detail.lower()
    if "balance" in low or "insufficient" in low or "quota" in low or "欠费" in detail:
        tip = "原因：模型额度不足（insufficient balance），请充值或换一个对话模型"
    elif "401" in detail or "unauthorized" in low or "invalid api key" in low:
        tip = "原因：模型鉴权失败（API Key 无效或已过期）"
    elif "403" in detail or "forbidden" in low or "permission" in low:
        tip = "原因：模型拒绝了本次请求（403）"
    elif "429" in detail or "rate limit" in low or "too many requests" in low:
        tip = "原因：模型请求过于频繁（被限流），请稍后再试"
    elif "timeout" in low or "timed out" in low:
        tip = "原因：调用模型超时，请稍后重试"
    else:
        tip = "原因：调用模型时出错"
    if detail:
        tip += f"\n原始错误：{detail[:160]}{'…' if len(detail) > 160 else ''}"
    tip += "\n（可在后台把「生图 tag 转换模型 ID」换成其他可用模型）"
    return tip


# ============ 常量 ============
DEFAULT_CONFIG = {
    "enable_image_gen": True,
    "nai_api_url": NAI_DEFAULT_API_URL,
    "nai_api_key": "",
    "nai_model": "nai-diffusion-4-5-full",
    "nai_size": "832x1216",
    "nai_steps": 28,
    "nai_scale": 5.0,
    "nai_sampler": "k_euler_ancestral",
    "nai_negative_extra": "",
    "nai_tag_provider": "",
    "nai_daily_limit": 5,
    "tusi_base_url": TUSI_DEFAULT_BASE_URL,
    "tusi_api_key": "",
    "tusi_template_id": "",
    "tusi_prompt_field": "",
    "tusi_negative_field": "",
    "http_proxy": ""
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# 临时文件命名前缀（独立命名空间，避免与其它程序同前缀文件互相误删）
TEMP_FILE_PREFIX = "nai_image_"


@register(
    "astrbot_plugin_nai_image",
    "kuaiyidian123",
    "AI 生图插件：/生图 中文描述 走 NovelAI，/生图 吐司 描述 走吐司（tusi.cn），由大模型先把中文转成英文绘画 tag",
    "1.0.0",
)
class NaiImagePlugin(Star):
    """AI 生图插件（NovelAI / 吐司）"""

    def __init__(self, context: Context, config=None):
        super().__init__(context)

        user_config = dict(config) if config else {}
        self.config = {**DEFAULT_CONFIG, **user_config}

        # 持久化目录：优先使用框架接口（兼容 ASTRBOT_ROOT / 桌面运行时），失败回退到 cwd 约定
        try:
            data_dir = str(StarTools.get_data_dir("astrbot_plugin_nai_image"))
        except Exception as e:
            logger.warning(f"获取框架数据目录失败，回退到默认路径: {e}")
            data_dir = os.path.join(
                os.getcwd(), 'data', 'plugin_data', 'astrbot_plugin_nai_image'
            )
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

        self._img_quota: Dict[str, Dict[str, Any]] = {}  # /生图 每日次数统计（按用户隔离）
        self._img_busy: set = set()                      # 正在生图的用户，防重复触发烧额度
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._last_temp_cleanup = 0.0  # 上次临时文件清理时间戳（节流用）
        self._session_lock = asyncio.Lock()  # 保护 HTTP 会话创建，避免并发重复建连

        logger.info("AI 生图插件初始化完成")

    # ==================== 生命周期 ====================

    async def terminate(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
        logger.info("AI 生图插件已卸载")

    async def _get_http_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                )
            return self._http_session

    def _get_proxy(self) -> Optional[str]:
        return self.config.get("http_proxy") or None

    # ==================== 通用工具 ====================

    def _get_user_id(self, event: AstrMessageEvent) -> str:
        """获取发送者唯一 ID（QQ 号）"""
        try:
            sender = event.message_obj.sender
            for attr in ('user_id', 'id'):
                val = getattr(sender, attr, None)
                if val:
                    return str(val)
        except Exception:
            pass
        return ""

    def _get_session_key(self, event: AstrMessageEvent) -> str:
        """会话（按「会话 + 用户」隔离）标识"""
        return f"session_{event.session_id}_user_{self._get_user_id(event)}"

    @staticmethod
    def _strip_command(text: str, command: str) -> str:
        """去掉消息开头的唤醒前缀与命令名，返回参数文本"""
        t = (text or "").strip()
        while t.startswith("/"):
            t = t[1:].strip()
        if t.startswith(command):
            t = t[len(command):].strip()
        while t.startswith("/"):
            t = t[1:].strip()
        return re.sub(r'\s+', ' ', t)

    # ==================== 生图：配置与限额 ====================

    def _nai_enabled(self) -> bool:
        """「/生图」功能开关"""
        return bool(self.config.get("enable_image_gen", False))

    def _nai_cfg(self) -> Dict[str, Any]:
        """解析并收敛生图配置（后台填错也不会直接崩，统一夹紧到安全范围）"""
        try:
            steps = int(self.config.get("nai_steps", 28))
        except (TypeError, ValueError):
            steps = 28
        try:
            scale = float(self.config.get("nai_scale", 5))
        except (TypeError, ValueError):
            scale = 5.0
        try:
            limit = int(self.config.get("nai_daily_limit", 5))
        except (TypeError, ValueError):
            limit = 5
        sampler = str(self.config.get("nai_sampler") or "").strip()
        if sampler not in NAI_SAMPLER_OPTIONS:
            sampler = "k_euler_ancestral"
        w, h = _parse_nai_size(self.config.get("nai_size"))
        return {
            "url": str(self.config.get("nai_api_url") or "").strip() or NAI_DEFAULT_API_URL,
            "key": str(self.config.get("nai_api_key") or "").strip(),
            "model": str(self.config.get("nai_model") or "").strip() or "nai-diffusion-4-5-full",
            "width": w,
            "height": h,
            "steps": min(max(steps, 1), 50),
            "scale": min(max(scale, 0.0), 10.0),
            "sampler": sampler,
            "extra_negative": str(self.config.get("nai_negative_extra") or "").strip(),
            "limit": max(0, limit),
        }

    # ---------- 每日次数限制 ----------

    def _img_quota_state(self, event: AstrMessageEvent) -> Dict[str, Any]:
        """取该用户今日的生图计数（跨天自动重置，条目过多时清理旧记录）"""
        key = self._get_user_id(event) or self._get_session_key(event)
        today = time.strftime("%Y-%m-%d")
        rec = self._img_quota.get(key)
        if not rec or rec.get("date") != today:
            rec = {"date": today, "count": 0}
            self._img_quota[key] = rec
            if len(self._img_quota) > 1000:
                for k in [k for k, v in self._img_quota.items()
                          if v.get("date") != today]:
                    del self._img_quota[k]
        return rec

    def _img_quota_check(self, event: AstrMessageEvent, limit: int) -> Optional[str]:
        """次数校验；limit<=0 表示不限次数。超限返回提示文案，否则 None"""
        if limit <= 0:
            return None
        if self._img_quota_state(event)["count"] >= limit:
            return f"⚠️ 今日生图次数已用完（{limit} 次/天），请明天再试"
        return None

    def _img_quota_commit(self, event: AstrMessageEvent, limit: int) -> None:
        """生图成功后计一次数（失败不计，避免配置/网络问题白扣用户次数）"""
        if limit > 0:
            self._img_quota_state(event)["count"] += 1

    def _img_quota_remaining(self, event: AstrMessageEvent, limit: int) -> int:
        """今日剩余次数；-1 表示不限"""
        if limit <= 0:
            return -1
        return max(0, limit - self._img_quota_state(event)["count"])

    # ---------- 中文描述 -> 英文 tag ----------

    async def _resolve_tag_provider(self):
        """取用于 tag 转换的对话模型：先按后台填的 ID，失败回退当前默认模型"""
        provider_id = str(self.config.get("nai_tag_provider") or "").strip()
        if provider_id:
            prov = None
            try:
                prov = self.context.get_provider_by_id(provider_id)
            except Exception as e:
                logger.warning(f"按 ID 获取对话模型失败: {e}")
            if prov is not None:
                return prov, None
            logger.warning(f"未找到对话模型「{provider_id}」，回退到 AstrBot 当前默认模型")
        try:
            prov = await self.context.get_using_provider_async()
        except Exception as e:
            logger.warning(f"获取默认对话模型失败: {e}")
            prov = None
        if prov is None:
            return None, (
                "❌ 未找到可用的对话模型\n"
                "请在 AstrBot 中配置模型，或在后台把「生图 tag 转换模型 ID」改成本实例已有的模型 ID"
            )
        return prov, None

    @staticmethod
    def _parse_tag_output(text: str) -> tuple:
        """从大模型输出中解析「正向tag / 负面tag」"""
        raw = text or ""
        pos = ""
        neg = ""
        m = re.search(r"正向\s*tag\s*[：:]\s*(.+)", raw, re.IGNORECASE)
        if m:
            pos = m.group(1).strip()
        m = re.search(r"负面\s*tag\s*[：:]\s*(.+)", raw, re.IGNORECASE)
        if m:
            neg = m.group(1).strip()
        return pos, neg

    @staticmethod
    def _normalize_tags(pos: str, neg: str, extra_negative: str) -> tuple:
        """规整 tag：正向补齐 masterpiece/best quality 打头，负面强制补全固定词"""
        pos = re.sub(r"\s+", " ", pos or "").strip(" ,，")
        if pos and "masterpiece" not in pos.lower():
            pos = f"masterpiece, best quality, {pos}"
        elif not pos:
            pos = "masterpiece, best quality"

        parts = [p.strip() for p in re.sub(r"\s+", " ", neg or "").split(",") if p.strip()]
        lower = {p.lower() for p in parts}
        for word in NAI_REQUIRED_NEGATIVE:
            if word.lower() not in lower:
                parts.append(word)
                lower.add(word.lower())
        for word in str(extra_negative or "").split(","):
            word = word.strip()
            if word and word.lower() not in lower:
                parts.append(word)
                lower.add(word.lower())
        return pos, ", ".join(parts)

    async def _convert_to_tags(self, text: str) -> tuple:
        """调用对话模型把中文描述转成英文绘画 tag，返回 (正向, 负面, 错误文案)"""
        prov, err = await self._resolve_tag_provider()
        if err:
            return "", "", err
        try:
            resp = await prov.text_chat(
                prompt=f"中文描述：{text}",
                system_prompt=NAI_TAG_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(f"调用对话模型转换绘画 tag 失败: {e}")
            return "", "", f"❌ 调用对话模型失败\n{_describe_llm_error(e)}"
        raw = str(getattr(resp, "completion_text", "") or "").strip()
        if not raw:
            return "", "", "❌ 对话模型没有返回内容，请稍后重试"
        pos, neg = self._parse_tag_output(raw)
        if not pos:
            logger.warning(f"绘画 tag 解析失败，原始输出: {raw[:200]}")
            return "", "", "❌ 关键词转换结果格式异常，请重试或换个描述"
        return pos, neg, None

    # ---------- NovelAI 接口 ----------

    @staticmethod
    async def _read_capped(resp, max_bytes: int) -> Optional[bytes]:
        """流式读取响应体并限制上限，超限返回 None（防止异常响应撑爆内存）"""
        buf = bytearray()
        async for chunk in resp.content.iter_chunked(64 * 1024):
            buf.extend(chunk)
            if len(buf) > max_bytes:
                return None
        return bytes(buf)

    @staticmethod
    def _extract_image(data: bytes) -> Optional[bytes]:
        """从 API 响应里取出图片字节：兼容 ZIP 包（官方返回格式）与裸 PNG/JPEG"""
        if not data:
            return None
        if data[:2] == b"PK":
            try:
                with zipfile.ZipFile(BytesIO(data)) as zf:
                    for name in sorted(zf.namelist()):
                        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            return zf.read(name)
            except Exception as e:
                logger.warning(f"解压 NovelAI 返回的 ZIP 失败: {e}")
            return None
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return data
        if data[:2] == b"\xff\xd8":
            return data
        return None

    @staticmethod
    def _nai_error_msg(status: int, body: bytes) -> str:
        """把接口错误转成好懂的中文提示（优先透出服务端 message）"""
        detail = ""
        try:
            obj = json.loads(body.decode("utf-8", "ignore"))
            if isinstance(obj, dict):
                detail = str(obj.get("message") or obj.get("error") or "").strip()
        except Exception:
            detail = ""
        if status == 401:
            return "❌ NovelAI Token 无效或已过期，请检查后台「生图 API Key」"
        if status == 402:
            return "❌ NovelAI Anlas 额度不足，请先补充额度"
        if status == 429:
            return "⚠️ NovelAI 请求过于频繁（被限流），请稍后再试"
        tail = f"\n{detail}" if detail else ""
        return f"❌ 生图失败（HTTP {status}）{tail}"

    async def _nai_generate(self, positive: str, negative: str,
                            cfg: Dict[str, Any]) -> tuple:
        """调用 NovelAI 生图接口，返回 (图片字节, 错误文案)"""
        payload = {
            "input": positive,
            "model": cfg["model"],
            "action": "generate",
            "parameters": {
                "params_version": 3,
                "width": cfg["width"],
                "height": cfg["height"],
                "scale": cfg["scale"],
                "sampler": cfg["sampler"],
                "steps": cfg["steps"],
                "n_samples": 1,
                "ucPreset": 0,
                "qualityToggle": True,
                "dynamic_thresholding": False,
                "controlnet_strength": 1,
                "legacy": False,
                "add_original_image": False,
                "cfg_rescale": 0,
                "noise_schedule": "native",
                "legacy_v3_extend": False,
                "skip_cfg_above_sigma": None,
                "use_coords": False,
                "seed": random.randint(0, 4294967295),
                "negative_prompt": negative,
                "sm": False,
                "sm_dyn": False,
                # V4.5/V5 必须同时带这两项，缺任意一个服务端会直接返回 500
                "v4_prompt": {
                    "caption": {"base_caption": positive, "char_captions": []},
                    "use_coords": False,
                    "use_order": True,
                },
                "v4_negative_prompt": {
                    "caption": {"base_caption": negative, "char_captions": []},
                    "use_coords": False,
                    "use_order": True,
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {cfg['key']}",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        }
        try:
            session = await self._get_http_session()
            async with session.post(
                cfg["url"], json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=NAI_REQUEST_TIMEOUT),
                proxy=self._get_proxy(),
            ) as resp:
                body = await self._read_capped(resp, NAI_MAX_IMAGE_BYTES)
                if body is None:
                    return None, "❌ 生成结果过大，已中止接收"
                if resp.status != 200:
                    logger.warning(f"NovelAI 生图失败 HTTP {resp.status}")
                    return None, self._nai_error_msg(resp.status, body)
            image = self._extract_image(body)
            if not image:
                logger.warning(f"NovelAI 返回内容无法识别，长度 {len(body)}")
                return None, "❌ 未能从接口响应中解析出图片，请检查 API 地址是否正确"
            return image, None
        except asyncio.TimeoutError:
            return None, "⏰ 生图超时，请稍后重试"
        except aiohttp.ClientConnectionError as e:
            host = urllib.parse.urlparse(cfg["url"]).netloc or cfg["url"]
            logger.error(f"NovelAI 生图连接失败 {host}: {e}")
            return None, (
                f"❌ 无法连接到生图接口：{host}\n"
                "请确认后台「生图 API 地址」完整有效；若地址本身没问题\n"
                "（例如域名能 ping 通），多为网络阻断所致，可在后台\n"
                "「HTTP 代理」填写本地代理地址（如 http://127.0.0.1:7890）。"
            )
        except Exception as e:
            logger.error(f"NovelAI 生图请求失败: {e}")
            return None, "❌ 生图请求失败，请稍后重试"

    # ---------- 吐司（TAMS）接口 ----------

    def _tusi_cfg(self) -> Dict[str, Any]:
        """解析吐司（TAMS）生图配置"""
        return {
            "base_url": (str(self.config.get("tusi_base_url") or "").strip()
                         or TUSI_DEFAULT_BASE_URL).rstrip("/"),
            "api_key": str(self.config.get("tusi_api_key") or "").strip(),
            "template_id": str(self.config.get("tusi_template_id") or "").strip(),
            "prompt_field": str(self.config.get("tusi_prompt_field") or "").strip(),
            "negative_field": str(self.config.get("tusi_negative_field") or "").strip(),
        }

    def _tusi_ready(self) -> bool:
        """吐司是否已配置好（API Key 与模板 ID 都填了才可用）"""
        cfg = self._tusi_cfg()
        return bool(cfg["api_key"] and cfg["template_id"])

    @staticmethod
    def _pick_tusi_attr(attrs: List[Dict[str, Any]], explicit: str,
                        keywords: tuple, exclude: tuple = ()) -> Optional[Dict[str, Any]]:
        """在模板字段里定位目标字段：优先用配置指定的名称，否则按关键词自动识别"""
        if explicit:
            target = explicit.lower()
            for attr in attrs:
                if str(attr.get("fieldName") or "").strip() == explicit:
                    return attr
            for attr in attrs:
                if target in str(attr.get("fieldName") or "").lower():
                    return attr
            return None
        for attr in attrs:
            name = str(attr.get("fieldName") or "").lower()
            if not name:
                continue
            if any(k in name for k in keywords) and not any(x in name for x in exclude):
                return attr
        return None

    @staticmethod
    def _tusi_error_msg(status: int, body: bytes) -> str:
        """把吐司接口错误转成可读提示"""
        detail = ""
        try:
            obj = json.loads(body.decode("utf-8", "ignore"))
            if isinstance(obj, dict):
                detail = str(obj.get("message") or obj.get("msg") or obj.get("error") or "").strip()
        except Exception:
            detail = ""
        if status == 401:
            return "❌ 吐司 API Key 无效或已过期，请检查后台「吐司 API Key」"
        if status == 403:
            return "❌ 吐司拒绝访问（403），请确认该应用权限或算力余额"
        if status == 404:
            return "❌ 未找到该吐司模板，请检查后台「吐司模板(ID)」是否正确"
        if status == 429:
            return "⚠️ 吐司请求过于频繁（被限流），请稍后再试"
        tail = f"\n{detail}" if detail else ""
        return f"❌ 吐司生图失败（HTTP {status}）{tail}"

    async def _tusi_download_image(self, session, url: str) -> tuple:
        """下载吐司生成的结果图片，返回 (图片字节, 错误文案)"""
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=TUSI_API_TIMEOUT),
                proxy=self._get_proxy(),
            ) as resp:
                if resp.status != 200:
                    return None, f"❌ 下载吐司生成结果失败（HTTP {resp.status}）"
                body = await self._read_capped(resp, TUSI_MAX_IMAGE_BYTES)
                if body is None:
                    return None, "❌ 生成结果过大，已中止接收"
            if not body:
                return None, "❌ 吐司生成结果为空"
            return body, None
        except Exception as e:
            logger.error(f"下载吐司生成结果失败: {e}")
            return None, "❌ 下载吐司生成结果失败，请稍后重试"

    async def _tusi_wait_job(self, session, base: str, headers: Dict[str, str],
                             job_id: str) -> tuple:
        """轮询吐司作业直到完成，返回 (图片字节, 错误文案)"""
        deadline = time.monotonic() + TUSI_JOB_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(TUSI_POLL_INTERVAL)
            try:
                async with session.get(
                    f"{base}/v1/jobs/{job_id}", headers=headers,
                    timeout=aiohttp.ClientTimeout(total=TUSI_API_TIMEOUT),
                    proxy=self._get_proxy(),
                ) as resp:
                    body = await self._read_capped(resp, TUSI_MAX_JSON_BYTES)
                    if body is None:
                        continue
                    if resp.status != 200:
                        logger.warning(f"吐司查询作业失败 HTTP {resp.status}")
                        return None, self._tusi_error_msg(resp.status, body)
                data = json.loads(body.decode("utf-8", "ignore")) or {}
            except Exception as e:
                # 单次轮询失败不致命，等下一轮继续查
                logger.warning(f"吐司查询作业异常（将继续重试）: {e}")
                continue

            job = data.get("job") or {}
            status = str(job.get("status") or "").upper()
            if status == "SUCCESS":
                images = ((job.get("successInfo") or {}).get("images")) or []
                if not images:
                    return None, "❌ 吐司作业已完成，但没有返回图片"
                url = str((images[0] or {}).get("url") or "").strip()
                if not url:
                    return None, "❌ 吐司返回的图片地址为空"
                return await self._tusi_download_image(session, url)
            if status == "FAILED":
                info = job.get("failedInfo") or job.get("message") or job.get("error") or ""
                detail = (json.dumps(info, ensure_ascii=False)
                          if isinstance(info, (dict, list)) else str(info))
                detail = " ".join(detail.split())[:200]
                logger.warning(f"吐司作业失败: {detail}")
                return None, f"❌ 吐司生图失败{f'{chr(10)}{detail}' if detail else ''}"
            # WAITING / RUNNING 等状态：继续等待
        return None, "⏰ 吐司生图等待超时，请稍后重试"

    async def _tusi_generate(self, positive: str, negative: str,
                             cfg: Dict[str, Any]) -> tuple:
        """调用吐司（TAMS）工作流模板接口生图，返回 (图片字节, 错误文案)"""
        base = cfg["base_url"]
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        }
        try:
            session = await self._get_http_session()
            # 1) 取模板参数定义（各模板字段不同，直接问接口最稳）
            async with session.get(
                f"{base}/v1/workflows/{cfg['template_id']}", headers=headers,
                timeout=aiohttp.ClientTimeout(total=TUSI_API_TIMEOUT),
                proxy=self._get_proxy(),
            ) as resp:
                body = await self._read_capped(resp, TUSI_MAX_JSON_BYTES)
                if body is None:
                    return None, "❌ 吐司模板信息过大，已中止接收"
                if resp.status != 200:
                    logger.warning(f"吐司取模板失败 HTTP {resp.status}")
                    return None, self._tusi_error_msg(resp.status, body)
            try:
                template = json.loads(body.decode("utf-8", "ignore")) or {}
            except Exception as e:
                logger.warning(f"吐司模板响应解析失败: {e}")
                return None, "❌ 吐司返回的模板信息无法解析，请检查接口地址"

            attrs = [dict(a) for a in (((template.get("fields") or {}).get("fieldAttrs")) or [])]
            if not attrs:
                return None, "❌ 该吐司模板没有可填写的参数，请更换「吐司模板(ID)」"

            prompt_attr = self._pick_tusi_attr(
                attrs, cfg["prompt_field"], TUSI_PROMPT_KEYS, TUSI_NEGATIVE_KEYS
            )
            if prompt_attr is None:
                return None, (
                    "❌ 未能识别吐司模板中的提示词字段\n"
                    "请在后台「吐司提示词字段名」中手动填写该字段名"
                )
            prompt_attr["fieldValue"] = positive
            negative_attr = self._pick_tusi_attr(
                attrs, cfg["negative_field"], TUSI_NEGATIVE_KEYS
            )
            if negative_attr is not None:
                negative_attr["fieldValue"] = negative

            # 2) 提交作业
            payload = {
                "requestId": hashlib.md5(
                    f"{cfg['template_id']}{time.time()}".encode("utf-8")
                ).hexdigest(),
                "templateId": cfg["template_id"],
                "fields": {"fieldAttrs": attrs},
            }
            async with session.post(
                f"{base}/v1/jobs/workflow/template", json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=TUSI_API_TIMEOUT),
                proxy=self._get_proxy(),
            ) as resp:
                body = await self._read_capped(resp, TUSI_MAX_JSON_BYTES)
                if body is None:
                    return None, "❌ 吐司返回内容过大，已中止接收"
                if resp.status != 200:
                    logger.warning(f"吐司提交作业失败 HTTP {resp.status}")
                    return None, self._tusi_error_msg(resp.status, body)
            try:
                created = json.loads(body.decode("utf-8", "ignore")) or {}
            except Exception as e:
                logger.warning(f"吐司提交响应解析失败: {e}")
                return None, "❌ 吐司提交作业失败，返回内容无法解析"
            job_id = str(((created.get("job") or {}).get("id")) or "").strip()
            if not job_id:
                logger.warning(f"吐司未返回作业 ID: {str(created)[:200]}")
                return None, "❌ 吐司未返回作业 ID，请稍后重试"

            # 3) 轮询到出图
            return await self._tusi_wait_job(session, base, headers, job_id)
        except aiohttp.ClientConnectionError as e:
            host = urllib.parse.urlparse(base).netloc or base
            logger.error(f"吐司连接失败 {host}: {e}")
            return None, (
                f"❌ 无法连接到吐司接口：{host}\n"
                "请检查后台「吐司接口地址」是否正确；若被网络阻断，"
                "可在后台「HTTP 代理」填写本地代理地址。"
            )
        except asyncio.TimeoutError:
            return None, "⏰ 吐司生图超时，请稍后重试"
        except Exception as e:
            logger.error(f"吐司生图请求失败: {e}")
            return None, "❌ 吐司生图请求失败，请稍后重试"

    # ==================== 指令：/生图 ====================

    @filter.command("生图")
    async def cmd_image_gen(self, event: AstrMessageEvent):
        """/生图 中文描述 → 大模型转英文绘画 tag → 生图并发送

        加「吐司」前缀（/生图 吐司 描述）则改走吐司（TAMS）接口。
        """
        if not self._nai_enabled():
            yield event.plain_result("⚠️「/生图」功能已被管理员关闭")
            return

        raw = self._strip_command(event.message_str, "生图")
        use_tusi, text = _split_tusi_prefix(raw)
        if not text:
            yield event.plain_result(
                "用法：/生图 中文描述\n"
                "      /生图 吐司 中文描述（走吐司接口）\n"
                "示例：/生图 蓝发少女站在樱花树下，微笑，逆光\n"
                "查看配置与可用模型：/生图帮助"
            )
            return
        if len(text) > MAX_NAI_PROMPT_CHARS:
            text = text[:MAX_NAI_PROMPT_CHARS]

        # 按渠道取配置
        nai_cfg = self._nai_cfg()
        tusi_cfg = self._tusi_cfg() if use_tusi else None
        if use_tusi:
            if not (tusi_cfg["api_key"] and tusi_cfg["template_id"]):
                yield event.plain_result(
                    "⚠️ 吐司生图尚未配置完整\n"
                    "请在后台填写「吐司 API Key」与「吐司模板(ID)」，详见 /生图帮助"
                )
                return
        elif not nai_cfg["key"]:
            yield event.plain_result(
                "⚠️ 尚未配置 NovelAI API Key\n请联系管理员在后台插件配置中填写「生图 API Key」"
            )
            return

        denied = self._img_quota_check(event, nai_cfg["limit"])
        if denied:
            yield event.plain_result(denied)
            return

        busy_key = self._get_user_id(event) or self._get_session_key(event)
        if busy_key in self._img_busy:
            yield event.plain_result("⏳ 你上一次生图还在进行中，请稍候再试")
            return

        self._img_busy.add(busy_key)
        try:
            yield event.plain_result("🎨 正在把描述转换成绘画关键词…")
            positive, negative, err = await self._convert_to_tags(text)
            if err:
                yield event.plain_result(err)
                return
            positive, negative = self._normalize_tags(
                positive, negative, nai_cfg["extra_negative"]
            )

            if use_tusi:
                yield event.plain_result(
                    f"🍞 已提交到吐司（模板 {tusi_cfg['template_id']}），排队/生成中，请稍候…"
                )
                image, err = await self._tusi_generate(positive, negative, tusi_cfg)
            else:
                yield event.plain_result(
                    f"🖌️ 正在生成图片（{nai_cfg['width']}x{nai_cfg['height']} · "
                    f"{nai_cfg['steps']} 步），通常需要十几秒，请稍候…"
                )
                image, err = await self._nai_generate(positive, negative, nai_cfg)
            if err:
                yield event.plain_result(err)
                return

            # 仅在成功后计数，避免网络/配置问题白扣用户次数
            self._img_quota_commit(event, nai_cfg["limit"])
            image_path = self._bytes_to_tempfile(image, ".png", "aigen")
            yield event.image_result(image_path)
            self._schedule_tempfile_cleanup(image_path)
        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result("❌ 生图失败，请稍后重试")
        finally:
            self._img_busy.discard(busy_key)

    @filter.command("生图帮助")
    async def cmd_image_gen_help(self, event: AstrMessageEvent):
        """/生图帮助 → 查看生图用法、当前配置与可选用的对话模型 ID"""
        cfg = self._nai_cfg()
        lines = ["🎨 AI 生图", "━━━━━━━━━━━━"]
        if self._nai_enabled():
            lines.append("用法：/生图 中文描述")
            lines.append("示例：/生图 蓝发少女站在樱花树下，微笑，逆光")
            if self._tusi_ready():
                lines.append("走吐司接口：/生图 吐司 中文描述")
        else:
            lines.append("⛔ 当前已被管理员关闭，请在后台插件配置中开启")
        lines.append("")
        lines.append("【NovelAI】")
        lines.append(f"模型：{cfg['model']}")
        lines.append(f"接口：{cfg['url']}")
        lines.append(
            f"尺寸：{cfg['width']}x{cfg['height']} · {cfg['steps']} 步 · "
            f"CFG {cfg['scale']:g} · {cfg['sampler']}"
        )
        lines.append(f"API Key：{'已配置' if cfg['key'] else '未配置'}")

        tusi = self._tusi_cfg()
        tusi_ok = bool(tusi["api_key"] and tusi["template_id"])
        lines.append("")
        lines.append("【吐司】")
        lines.append(f"状态：{'已配置' if tusi_ok else '未配置（需填 API Key 与模板 ID）'}")
        lines.append(f"接口：{tusi['base_url']}")
        lines.append(f"模板：{tusi['template_id'] or '（未填写）'}")

        provider_id = str(self.config.get("nai_tag_provider") or "").strip()
        lines.append("")
        lines.append(f"转换模型：{provider_id or '（AstrBot 当前默认对话模型）'}")
        if cfg["limit"] > 0:
            lines.append(
                f"今日剩余：{self._img_quota_remaining(event, cfg['limit'])}/{cfg['limit']} 次"
            )
        else:
            lines.append("次数限制：不限")

        try:
            providers = list(self.context.get_all_providers())
        except Exception as e:
            logger.warning(f"获取对话模型列表失败: {e}")
            providers = []
        if providers:
            lines.append("")
            lines.append("可选转换模型 ID（填到后台「生图 tag 转换模型 ID」）：")
            for i, prov in enumerate(providers[:20], 1):
                try:
                    meta = prov.meta()
                    lines.append(f"{i}. {meta.id}　（{meta.type} / {meta.model}）")
                except Exception:
                    continue

        yield event.plain_result("\n".join(lines))

    # ==================== 临时文件管理 ====================

    # 临时文件清理策略（仅清理本插件产生的文件，且保留 1 小时）
    TEMP_CLEANUP_INTERVAL = 3600  # 两次扫描的最小间隔（秒），避免频繁遍历目录
    TEMP_FILE_MAX_AGE = 3600      # 文件保留时长（秒），超过才删除

    def _cleanup_temp_files(self):
        """节流清理系统临时目录中由本插件产生、且修改时间超过 1 小时的文件"""
        now = time.time()
        if now - self._last_temp_cleanup < self.TEMP_CLEANUP_INTERVAL:
            return
        self._last_temp_cleanup = now
        try:
            tmp_dir = tempfile.gettempdir()
            for name in os.listdir(tmp_dir):
                # 严格按前缀匹配，绝不触碰其它程序的文件
                if not name.startswith(TEMP_FILE_PREFIX):
                    continue
                path = os.path.join(tmp_dir, name)
                try:
                    if (os.path.isfile(path)
                            and now - os.path.getmtime(path) > self.TEMP_FILE_MAX_AGE):
                        os.remove(path)
                except OSError:
                    continue
        except OSError:
            pass

    def _bytes_to_tempfile(self, data: bytes, suffix: str, tag: str = "file") -> str:
        self._cleanup_temp_files()
        tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"{TEMP_FILE_PREFIX}{tag}_{int(time.time() * 1000)}{suffix}"
        )
        with open(tmp_path, 'wb') as f:
            f.write(data)
        return tmp_path

    @staticmethod
    def _remove_tempfile(path: str) -> None:
        """删除本插件产生的临时文件（仅限本插件前缀，避免误删其它程序文件）"""
        if not path:
            return
        try:
            name = os.path.basename(path)
            if name.startswith(TEMP_FILE_PREFIX) and os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass

    def _schedule_tempfile_cleanup(self, path: str, delay: int = 180) -> None:
        """延迟删除临时文件：等到框架真正发送完毕后再清理，避免高峰期堆积"""
        if not path:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.call_later(delay, self._remove_tempfile, path)
        except RuntimeError:
            pass
