
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardRemove,
    ReplyKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    PicklePersistence,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Optional rate limiter (requires extra: python-telegram-bot[rate-limiter])
try:
    from telegram.ext import AIORateLimiter  # type: ignore
except Exception:
    AIORateLimiter = None  # type: ignore

load_dotenv()

TG_TOKEN = os.environ.get("TG_TOKEN", "")
ADMIN_IDS = {
    int(x.strip())
    for x in os.environ.get("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}
PANEL_BASE = os.environ.get("PANEL_BASE", "").rstrip("/")
PANEL_USERNAME = os.environ.get("PANEL_USERNAME", "")
PANEL_PASSWORD = os.environ.get("PANEL_PASSWORD", "")
VERIFY_TLS = os.environ.get("VERIFY_TLS", "1") not in {"0", "false", "False"}
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "")
DEFAULT_TRAFFIC_GB = int(os.environ.get("DEFAULT_TRAFFIC_GB", "30"))
DEFAULT_DAYS = int(os.environ.get("DEFAULT_DAYS", "30"))
DEFAULT_LIMIT_IP = int(os.environ.get("DEFAULT_LIMIT_IP", "0"))

if not TG_TOKEN:
    raise SystemExit("Set TG_TOKEN in environment")
if not PANEL_BASE:
    raise SystemExit("Set PANEL_BASE in environment")

# ---- Helpers -----------------------------------------------------------------

def _kb(rows: List[List[Tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text=t, callback_data=d) for t, d in row] for row in rows]
    )

async def _reply_err(update: Update, ctx: ContextTypes.DEFAULT_TYPE, msg: str):
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(f"❌ {msg}")
    else:
        await update.effective_message.reply_text(f"❌ {msg}")

# ---- 3x-ui API client --------------------------------------------------------

@dataclass
class XUIPanel:
    base: str
    username: str
    password: str
    verify_tls: bool = True

    def __post_init__(self):
        self.base = self.base.rstrip("/") + "/"
        self.login_path_env = os.environ.get("PANEL_LOGIN_PATH", "").strip("/")
        self.cookie_names = {"session", "3x-ui", "x-ui", "X-UI-SESSION", "3x-ui-session"}
        self.client: Optional[httpx.AsyncClient] = None
        self._last_login_ts = 0.0

    async def _ensure_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base,
                verify=self.verify_tls,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (XUI-Bot)", "Accept": "application/json, */*"},
            )

    async def login(self) -> None:
        await self._ensure_client()
        jar_keys = set(self.client.cookies.keys())
        if (self.cookie_names & jar_keys) and (time.time() - self._last_login_ts < 120):
            return
        self.client.cookies.clear()
        paths = []
        if getattr(self, "login_path_env", ""):
            paths.append(self.login_path_env)
        paths.extend(["login", "login/"])
        for _path in paths:
            try:
                await self.client.post(
                    _path,
                    follow_redirects=True,
                    data={"username": self.username, "password": self.password},
                    headers={"Accept": "application/json"},
                )
                jar_keys = set(self.client.cookies.keys())
                if self.cookie_names & jar_keys:
                    self._last_login_ts = time.time()
                    break
            except Exception:
                pass
        else:
            raise RuntimeError("Login failed: no session cookie returned")

    async def _req(self, method: str, path: str, **kw) -> Dict[str, Any]:
        await self.login()
        headers = kw.pop("headers", {})
        headers.setdefault("Accept", "application/json")
        r = await self.client.request(method, path, headers=headers, **kw)
        if r.status_code in (401, 403):
            self.client.cookies.clear()
            self._last_login_ts = 0.0
            await self.login()
            r = await self.client.request(method, path, headers=headers, **kw)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"Panel non-JSON response at {path}")
        if not isinstance(data, dict) or not data.get("success", True):
            raise RuntimeError(f"Panel error: {data}")
        return data

    async def inbounds_list(self) -> List[Dict[str, Any]]:
        data = await self._req("GET", "panel/api/inbounds/list", headers={"Accept": "application/json"})
        return data.get("obj", [])

    async def inbound_get(self, inbound_id: int) -> Dict[str, Any]:
        data = await self._req("GET", f"panel/api/inbounds/get/{inbound_id}", headers={"Accept": "application/json"})
        return data.get("obj", {})

    async def add_client(
        self,
        inbound_id: int,
        *,
        uuid_str: Optional[str] = None,
        email: str = "",
        enable: bool = True,
        limit_ip: int = 0,
        total_gb: int = 0,
        expiry_ts_ms: int = 0,
        flow: str = "",
        sub_id: str = "",
        tg_id: str = "",
        comment: str = "",
    ) -> Dict[str, Any]:
        if uuid_str is None:
            try:
                new_uuid = await self.get_new_uuid()
                uuid_str = new_uuid
            except Exception:
                uuid_str = str(uuid.uuid4())
        client_obj = {
            "id": uuid_str,
            "alterId": 0,
            "email": email,
            "limitIp": int(limit_ip),
            "totalGB": int(total_gb),
            "expiryTime": int(expiry_ts_ms),
            "enable": bool(enable),
            "tgId": tg_id,
            "subId": sub_id,
            "comment": comment,
            "flow": flow,
        }
        settings_str = json.dumps({"clients": [client_obj]}, separators=(",", ":"))
        payload = {"id": inbound_id, "settings": settings_str}
        return await self._req(
            "POST",
            "panel/api/inbounds/addClient",
            data=payload,
            headers={"Accept": "application/json"},
        )

    async def update_client(self, inbound_id: int, client_uuid: str, updated_client: Dict[str, Any]) -> Dict[str, Any]:
        settings_str = json.dumps({"clients": [updated_client]}, separators=(",", ":"))
        payload = {"id": inbound_id, "settings": settings_str}
        return await self._req(
            "POST",
            f"panel/api/inbounds/updateClient/{client_uuid}",
            data=payload,
            headers={"Accept": "application/json"},
        )

    async def delete_client(self, inbound_id: int, client_id: str) -> Dict[str, Any]:
        return await self._req(
            "POST",
            f"panel/api/inbounds/{inbound_id}/delClient/{client_id}",
            headers={"Accept": "application/json"},
        )

    async def reset_client_traffic(self, inbound_id: int, email: str) -> Dict[str, Any]:
        return await self._req(
            "POST",
            f"panel/api/inbounds/{inbound_id}/resetClientTraffic/{email}",
            headers={"Accept": "application/json"},
        )

    async def clear_client_ips(self, email: str) -> Dict[str, Any]:
        return await self._req(
            "POST",
            f"panel/api/inbounds/clearClientIps/{email}",
            headers={"Accept": "application/json"},
        )

    async def onlines(self) -> List[str]:
        data = await self._req("POST", "panel/api/inbounds/onlines", headers={"Accept": "application/json"})
        return data.get("obj", [])

    async def get_new_uuid(self) -> str:
        data = await self._req("GET", "panel/api/server/getNewUUID", headers={"Accept": "application/json"})
        return data.get("obj")

# ---- VLESS URL composer ------------------------------------------------------

class VlessURL:
    @staticmethod
    def human_remark(email: str, inbound_remark: str) -> str:
        name = email or inbound_remark or "client"
        return re.sub(r"\s+", "_", name)[:40]

    @staticmethod
    def compose(inbound: Dict[str, Any], client_uuid: str, email: str, public_host: str) -> str:
        port = inbound.get("port")
        remark = inbound.get("remark", "")
        network = "tcp"
        sni = None
        host_header = None
        type_param = None
        path = None
        service_name = None
        security = "none"
        flow = None
        pbk = None
        sid = None

        try:
            stream = json.loads(inbound.get("streamSettings", "{}"))
        except json.JSONDecodeError:
            stream = {}
        try:
            settings = json.loads(inbound.get("settings", "{}"))
        except json.JSONDecodeError:
            settings = {}

        if stream:
            network = stream.get("network", network)
            security = stream.get("security", security)
            if network == "ws":
                ws = stream.get("wsSettings", {}) or stream.get("wssettings", {})
                path = ws.get("path") or "/"
                headers = ws.get("headers", {}) or {}
                host_header = headers.get("Host") or headers.get("host")
                type_param = "ws"
            elif network == "grpc":
                grpc = stream.get("grpcSettings", {}) or stream.get("grpcsettings", {})
                service_name = grpc.get("serviceName")
                type_param = "grpc"
            elif network == "http":
                http = stream.get("httpSettings", {})
                path = (http.get("path") or ["/"])[0] if isinstance(http.get("path"), list) else http.get("path")
                host_header = (http.get("host") or [None])[0] if isinstance(http.get("host"), list) else http.get("host")
                type_param = "http"

            if security == "reality":
                reality = stream.get("realitySettings", {})
                sni = (reality.get("serverNames") or [None])[0]
                pbk = reality.get("publicKey")
                sid = (reality.get("shortIds") or [None])[0]
                flow = "xtls-rprx-vision"
            elif security == "tls":
                tls = stream.get("tlsSettings", {}) or {}
                sni = tls.get("serverName") or tls.get("serverName")

        if security in {"tls", "reality"} and not sni:
            sni = host_header or public_host

        q = {"encryption": "none"}
        if security in {"tls", "reality"}:
            q["security"] = security
        if sni:
            q["sni"] = sni
        if network in {"ws", "grpc", "http"}:
            q["type"] = network if not type_param else type_param
        if path:
            q["path"] = path
        if host_header:
            q["host"] = host_header
        if service_name:
            q["serviceName"] = service_name
        if security == "reality":
            if pbk:
                q["pbk"] = pbk
            if sid:
                q["sid"] = sid
            q["fp"] = "chrome"
            q["flow"] = flow or "xtls-rprx-vision"

        from urllib.parse import urlencode, quote
        qs = urlencode({k: v for k, v in q.items() if v is not None})
        tag = quote(VlessURL.human_remark(email, remark))
        return f"vless://{client_uuid}@{public_host}:{port}?{qs}#{tag}"

# ---- Payments / Reminders ----------------------------------------------------

NOVOSIBIRSK_TZ = ZoneInfo("Asia/Novosibirsk")
MONTHLY_FEE_RUB = 200  # стоимость подписки в рублях за 1 месяц

def month_key(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(NOVOSIBIRSK_TZ)
    return dt.strftime("%Y-%m")

def ensure_botdata_defaults(app):
    bd = app.bot_data
    bd.setdefault("users", {})
    bd.setdefault("pay_text", (
        "🧾 <b>Счёт за подписку</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "👤 {name}\n"
        "📆 Период: <code>{period}</code>\n"
        "💳 Статус: <b>ожидает оплаты</b>\n\n"
        "Пожалуйста, оплатите подписку. Спасибо!"
    ))
    bd.setdefault("overdue_prefix", "⚠️ <b>Напоминание об оплате</b>\n\n")
    bd.setdefault("pay_status", {})
    bd.setdefault("paid_until", {})  # user_id -> "YYYY-MM" (включительно)
    bd.setdefault("pay_targets", "all")
    bd.setdefault("pay_schedule", {"day": 10, "prelist": "09:00", "remind": "10:00", "tz": "Asia/Novosibirsk"})
    return bd

def non_admin_users(bot_data: dict) -> dict[int, dict]:
    return {uid: u for uid, u in bot_data.get("users", {}).items() if not u.get("is_admin")}

def pay_target_set(bot_data: dict) -> set[int]:
    """
    pay_targets:
      - "all" => all non-admin users are included
      - list[int] => explicit allow-list of user ids
    """
    pu = non_admin_users(bot_data)
    targets = bot_data.get("pay_targets", "all")
    if targets == "all" or targets is None:
        return set(pu.keys())
    try:
        return {int(x) for x in targets if int(x) in pu}
    except Exception:
        return set(pu.keys())

def set_pay_targets(bot_data: dict, targets: set[int]):
    pu = non_admin_users(bot_data)
    if targets == set(pu.keys()):
        bot_data["pay_targets"] = "all"
    else:
        bot_data["pay_targets"] = sorted(list(targets))


def _ym_to_ints(ym: str) -> tuple[int, int]:
    y, m = ym.split("-", 1)
    return int(y), int(m)

def _ints_to_ym(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"

def add_months_ym(ym: str, months: int) -> str:
    """Add N months to YYYY-MM, returning YYYY-MM."""
    y, m = _ym_to_ints(ym)
    total = (y * 12 + (m - 1)) + months
    ny = total // 12
    nm = (total % 12) + 1
    return _ints_to_ym(ny, nm)

def current_period_from_botdata(bot_data: dict) -> str:
    sched = bot_data.get("pay_schedule", {}) or {}
    tz_name = sched.get("tz", "Asia/Novosibirsk")
    tz = ZoneInfo(tz_name)
    return month_key(datetime.now(tz))

def is_prepaid(bot_data: dict, uid: int, period: str) -> bool:
    """True if user is prepaid for 'period' (YYYY-MM) based on paid_until."""
    until = (bot_data.get("paid_until", {}) or {}).get(uid)
    if not until:
        return False
    # Lexicographic compare works for YYYY-MM
    return str(until) >= str(period)


def apply_prepay(bot_data: dict, uid: int, amount_rub: int) -> tuple[int, int, str, str]:
    """
    Apply prepayment to user.
    Returns: (months, remainder, new_until_ym, current_period_ym)
    """
    months = amount_rub // MONTHLY_FEE_RUB
    remainder = amount_rub % MONTHLY_FEE_RUB
    if months <= 0:
        raise ValueError(f"Минимум {MONTHLY_FEE_RUB} ₽ за 1 месяц")

    period = current_period_from_botdata(bot_data)
    paid_until = (bot_data.get("paid_until", {}) or {}).get(uid)

    if paid_until and str(paid_until) >= str(period):
        new_until = add_months_ym(str(paid_until), months)
    else:
        new_until = add_months_ym(str(period), months - 1)

    # Ensure user is in targets if using explicit list
    pu = non_admin_users(bot_data)
    if uid in pu and bot_data.get("pay_targets", "all") != "all":
        targets = pay_target_set(bot_data)
        targets.add(uid)
        set_pay_targets(bot_data, targets)

    bot_data.setdefault("paid_until", {})[uid] = new_until
    bot_data.setdefault("pay_status", {}).setdefault(period, {})[uid] = "paid"
    return months, remainder, new_until, period


def build_prepay_user_menu(bot_data: dict, page: int = 0, page_size: int = 10, query: str = "") -> InlineKeyboardMarkup:
    pu = non_admin_users(bot_data)
    items = []
    q = (query or "").strip().lower()
    for uid, u in pu.items():
        name = u.get("name") or f"ID {uid}"
        uname = (u.get("username") or "").lower()
        if q and (q not in name.lower()) and (q not in uname):
            continue
        items.append((uid, name, u.get("username")))

    items.sort(key=lambda x: x[1].lower())
    total = len(items)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, pages - 1))
    start = page * page_size
    chunk = items[start:start + page_size]

    rows = []
    for uid, name, uname in chunk:
        label = name
        if uname:
            label = f"{name} ({uname})"
        if len(label) > 40:
            label = label[:37] + "..."
        rows.append([InlineKeyboardButton(label, callback_data=f"prepay:select:{uid}")])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"prepay:page:{page-1}"))
    if page < pages - 1:
        nav.append(InlineKeyboardButton("Вперёд ➡️", callback_data=f"prepay:page:{page+1}"))
    if nav:
        rows.append(nav)

    if q:
        rows.append([InlineKeyboardButton("🧹 Сбросить поиск", callback_data="prepay:clear")])

    rows.append([InlineKeyboardButton("❌ Отмена", callback_data="prepay:cancel")])
    return InlineKeyboardMarkup(rows)
def format_pay_text(tpl: str, name: str, period: str) -> tuple[str, str | None]:
    """
    Safe formatting for admin-editable templates.

    Supports:
      - Named placeholders: {name}, {period}
      - Positional placeholders: {0}, {1}
      - Empty positional placeholders: {} {} (mapped to 0,1,...)

    If user wants literal braces, they must escape as {{ and }}.
    """
    try:
        # Provide both positional and named args.
        return tpl.format(name, period, name=name, period=period), None
    except Exception as e:
        return tpl, f"{type(e).__name__}: {e}"
def render_payment_message(bot_data: dict, user: dict, period: str, prefix: str = ""):
    tpl = bot_data.get("pay_text") or (
        "🧾 <b>Счёт за подписку</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "👤 {name}\n"
        "📆 Период: <code>{period}</code>\n"
        "💳 Статус: <b>ожидает оплаты</b>\n\n"
        "Пожалуйста, оплатите подписку. Спасибо!"
    )
    name = user.get("name") or "Пользователь"
    formatted, err = format_pay_text(tpl, name=name, period=period)
    if err:
        # fallback to a safe default template, but keep admin template unchanged
        fallback = (
            "🧾 <b>Счёт за подписку</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "👤 {name}\n"
            "📆 Период: <code>{period}</code>\n"
            "💳 Статус: <b>ожидает оплаты</b>\n\n"
            "Пожалуйста, оплатите подписку. Спасибо!"
        )
        formatted, _ = format_pay_text(fallback, name=name, period=period)
        formatted = (
            "⚠️ <b>Шаблон уведомления сломан</b>\n"
            f"<code>{err}</code>\n\n"
            "Используйте {name}/{period} или {0}/{1} или {}.\n"
            "Для фигурных скобок ставьте {{ и }}.\n\n"
            + formatted
        )
    text = f"{prefix}{formatted}"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔔 Напомнить позже", callback_data=f"pay:remind:{user['id']}"),
         InlineKeyboardButton("✅ Оплачено", callback_data=f"pay:paid:{user['id']}")]
    ])
    return text, kb


# ---- Telegram Bot ------------------------------------------------------------

PANEL = XUIPanel(
    base=PANEL_BASE,
    username=PANEL_USERNAME,
    password=PANEL_PASSWORD,
    verify_tls=VERIFY_TLS,
)

# Conversation states
(ADD_SELECT_INBOUND, ADD_EMAIL, ADD_TRAFFIC, ADD_DAYS, ADD_LIMITIP, ADD_CONFIRM,
 VLESS_SELECT_INBOUND, VLESS_SELECT_CLIENT,
 DEL_SELECT_INBOUND, DEL_SELECT_CLIENT,
 TOGGLE_SELECT_INBOUND, TOGGLE_SELECT_CLIENT,
 RESET_SELECT_INBOUND, RESET_SELECT_CLIENT,
 DISABLE_SELECT_INBOUND, DISABLE_SELECT_CLIENT,
 ENABLE_SELECT_INBOUND, ENABLE_SELECT_CLIENT,
 DISABLEALL_SELECT_INBOUND, DISABLEALL_CONFIRM,
 ENABLEALL_SELECT_INBOUND, ENABLEALL_CONFIRM,
) = range(22)
PREPAY_PICK_USER = 2001
PREPAY_ENTER_AMOUNT = 2002



def admin_only(func):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else 0
        if uid not in ADMIN_IDS:
            await _reply_err(update, ctx, "Доступ запрещён (не в ADMIN_IDS)")
            return
        return await func(update, ctx)
    return wrapper


# --- Basic user commands
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bd = ensure_botdata_defaults(ctx.application)
    bd["users"].setdefault(user.id, {"name": user.full_name, "username": ("@"+user.username) if user.username else None, "is_admin": user.id in ADMIN_IDS})
    kb = ReplyKeyboardMarkup([["/info", "/contact"]], resize_keyboard=True)
    if user.id in ADMIN_IDS:
        await update.message.reply_text(
            "👋 <b>Привет, админ!</b>\n"
            "Ниже — полный список возможностей бота. Команды сгруппированы, чтобы было проще ориентироваться.\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🧾 <b>Оплаты и уведомления</b>\n\n"
            "• /paylist — список оплат за текущий период (кнопки: Оплатил/Не оплатил + ⚙️ Получатели)\n"
            "• /paytext — посмотреть/изменить шаблон уведомления об оплате\n"
            "  └ можно так: <code>/paytext ...текст...</code> или ответом на сообщение командой /paytext\n"
            "  └ плейсхолдеры: <code>{name}</code> и <code>{period}</code> (можно <code>{0}/{1}</code> или <code>{}</code>)\n"
            "• /payschedule — показать/изменить дату и время\n"
            "  └ пример: <code>/payschedule day=10 prelist=09:00 remind=10:00 tz=Asia/Novosibirsk</code>\n"
            "  └ prelist = обновление списка и отправка /paylist админам\n"
            "  └ remind = рассылка уведомлений пользователям\n"
            "• /prepay — предоплата по сумме (200 ₽ = 1 месяц, уведомления на оплаченные месяцы не приходят)\n"
            "  └ удобно: <code>/prepay</code> → выбрать юзера кнопкой → ввести сумму\n"
            "  └ быстро: ответом <code>/prepay 1000</code> или <code>/prepay @user 1000</code>\n"
            "• /broadcast — разовое сообщение всем пользователям (текст после команды или ответом)\n\n"
            "🧩 <b>3x-ui / X-UI управление</b>\n\n"
            "• /inbounds — список инбаундов\n"
            "• /clients — список клиентов в выбранном инбаунде\n"
            "• /add — добавить клиента (диалог)\n"
            "• /del — удалить клиента\n"
            "• /toggle — включить/выключить клиента\n"
            "• /disable — выключить конкретного клиента\n"
            "• /enable — включить конкретного клиента\n"
            "• /reset — сброс трафика клиента\n"
            "• /vless — получить VLESS URL клиента\n"
            "• /online — кто сейчас онлайн (если поддерживается панелью)\n"
            "• /disableall — массово выключить всех в инбаунде\n"
            "• /enableall — массово включить всех в инбаунде\n"
            "\n👥 <b>Как это работает по оплатам</b>\n\n"
            "• Каждый месяц бот делает авто-обновление статусов на <b>10 число</b> в <b>09:00</b> (по умолчанию, Новосибирск).\n"
            "• В <b>10:00</b> (по умолчанию) рассылаются уведомления тем, кто в списке получателей и не предоплачен.\n"
            "• Если ты в /paylist нажимаешь «Не оплатил ❌» — пользователю уходит напоминание <b>сразу</b>.\n"
            "⚠️ <b>Если планировщик не работает</b>\n"
            "Нужен JobQueue: установи <code>pip install \"python-telegram-bot[job-queue]\"</code> и перезапусти бота.\n"
            "Команды для обычных пользователей ограничены: /info и /contact.\n",
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
        )
    else:
        await update.message.reply_text(
            "Добро пожаловать в моего бота.\nТут ты получаешь ежемсячные уведомления об оплате. \n\n(Просто удобная напоминалка и ничего лишнего\n"
            "Доступно: /info и /contact"
            "\n\n⚠️❗️Бот еще находтся в режиме тестировки.⚠️❗️\n\n",
            reply_markup=kb
        )

async def info_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ℹ️ Информация: подписка активна. Уведомления приходят 10 числа каждого месяца.\nБот запоминает кто заплатил. Это и мне и Вам легче😉")

async def contact_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_IDS:
        await update.message.reply_text("Администратор не задан.")
        return
    admin_id = next(iter(ADMIN_IDS))
    await update.message.reply_text(f"✉️ Написать администратору: @dan_sava", parse_mode=ParseMode.HTML)

# --- Payment callbacks and admin list
async def cb_pay(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data.split(":")  # pay:action:user_id
    if len(data) != 3:
        return
    action, uid = data[1], int(data[2])
    bd = ensure_botdata_defaults(ctx.application)
    user = bd.get("users", {}).get(uid, {"id": uid, "name": "Пользователь"})
    period = month_key()
    status = bd.setdefault("pay_status", {}).setdefault(period, {})
    if action == "remind":
        def send_reminder(ctx2):
            txt, kb = render_payment_message(bd, {"id":uid,"name":user.get("name","Пользователь")}, period)
            ctx2.bot.send_message(chat_id=uid, text=txt, reply_markup=kb, parse_mode=ParseMode.HTML)
        jq = ctx.application.job_queue
        if jq is None:
            await q.edit_message_text("❌ Планировщик недоступен (нет JobQueue).", parse_mode=ParseMode.HTML)
            return
        jq.run_once(lambda c: send_reminder(c), when=5*60*60, name=f"pay_remind_{uid}")
        await q.edit_message_text("⏰ Напоминание придёт через 5 часов.", parse_mode=ParseMode.HTML)
    elif action == "paid":
        status[uid] = "paid"
        for admin_id in ADMIN_IDS:
            try:
                await ctx.bot.send_message(chat_id=admin_id, text=f"💰 Пользователь <a href='tg://user?id={uid}'>{user.get('name','Без имени')}</a> отметил оплату за {period}.", parse_mode=ParseMode.HTML)
            except Exception:
                pass
        await q.edit_message_text("✅ Спасибо! Оплата отмечена.", parse_mode=ParseMode.HTML)

@admin_only
async def paylist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    period = month_key()
    pu = non_admin_users(bd)
    targets = pay_target_set(bd)
    users = {uid: u for uid, u in pu.items() if uid in targets}
    status = bd.setdefault("pay_status", {}).setdefault(period, {})
    if not users:
        await update.message.reply_text("Пользователи не найдены")
        return
    lines = []
    buttons = [[InlineKeyboardButton('⚙️ Получатели', callback_data='paytargets:menu')]]
    for uid, u in users.items():
        st = status.get(uid, "unpaid")
        until = (bd.get("paid_until", {}) or {}).get(uid)
        if until and str(until) >= str(period):
            mark = f"✅ <i>до {until}</i>"
            status[uid] = "paid"
        else:
            mark = "✅" if st == "paid" else "❌"
        link = f"<a href='tg://user?id={uid}'>{u.get('name','Без имени')}</a>"
        lines.append(f"{link} — {mark}")
        buttons.append([
            InlineKeyboardButton(f"Оплатил ✅", callback_data=f"payset:paid:{uid}"),
            InlineKeyboardButton(f"Не оплатил ❌", callback_data=f"payset:unpaid:{uid}"),
        ])
    text = "🧾 <b>Список оплат</b> — " + period + "\n" + "\n".join(lines)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(buttons))

async def cb_payset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, action, uid = q.data.split(":")
    uid = int(uid)
    bd = ensure_botdata_defaults(ctx.application)
    period = month_key()
    status = bd.setdefault("pay_status", {}).setdefault(period, {})
    if action == "paid":
        status[uid] = "paid"
        await q.edit_message_text("✅ Отмечено как оплачено", parse_mode=ParseMode.HTML)
    else:
        status[uid] = "unpaid"
        # admin confirmed НЕ оплатил -> отменяем предоплату/кредит
        try:
            bd.setdefault("paid_until", {}).pop(uid, None)
        except Exception:
            pass
        user = bd["users"].get(uid, {"name":"Пользователь","id":uid})
        prefix = bd.get("overdue_prefix", "")
        txt, kb = render_payment_message(bd, {"id":uid,"name":user.get("name","Пользователь")}, period, prefix=prefix)
        try:
            await ctx.bot.send_message(chat_id=uid, text=txt, reply_markup=kb, parse_mode=ParseMode.HTML)
        except Exception:
            pass
        await q.edit_message_text("❌ Отмечено как не оплачено (пользователю отправлено повторное напоминание)", parse_mode=ParseMode.HTML)


async def cb_paytargets(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    bd = ensure_botdata_defaults(ctx.application)
    pu = non_admin_users(bd)
    targets = pay_target_set(bd)

    parts = q.data.split(":")  # paytargets:action[:uid]
    action = parts[1] if len(parts) > 1 else "menu"

    async def show_menu():
        rows = [
            [
                InlineKeyboardButton("✅ Включить всех", callback_data="paytargets:all_on"),
                InlineKeyboardButton("🚫 Исключить всех", callback_data="paytargets:all_off"),
            ]
        ]
        for uid, u in pu.items():
            included = uid in targets
            icon = "✅" if included else "🚫"
            rows.append([InlineKeyboardButton(f"{icon} {u.get('name','Без имени')}", callback_data=f"paytargets:toggle:{uid}")])
        rows.append([InlineKeyboardButton("⬅️ Назад к /paylist", callback_data="paytargets:back")])
        text = (
            "⚙️ <b>Получатели уведомлений об оплате</b>\n"
            "Нажимай на пользователя, чтобы добавить/убрать из списка.\n\n"
            "✅ — в списке, 🚫 — исключён"
        )
        await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(rows))

    if action == "menu":
        await show_menu()
        return


    if action == "all_on":
        # include all non-admin users
        set_pay_targets(bd, set(pu.keys()))
        period = month_key()
        st = bd.setdefault("pay_status", {}).setdefault(period, {})
        for uid in pu.keys():
            st.setdefault(uid, "unpaid")
        targets = pay_target_set(bd)
        await show_menu()
        return

    if action == "all_off":
        # exclude everyone
        set_pay_targets(bd, set())
        period = month_key()
        # optional: keep status, but clear for cleanliness
        if period in bd.get("pay_status", {}):
            bd["pay_status"][period] = {}
        targets = pay_target_set(bd)
        await show_menu()
        return

    if action == "toggle" and len(parts) == 3:
        uid = int(parts[2])
        if uid not in pu:
            await q.answer("Пользователь не найден", show_alert=True)
            return

        # If currently "all" and admin excludes someone -> convert to explicit set
        if bd.get("pay_targets", "all") == "all":
            targets = set(pu.keys())

        if uid in targets:
            targets.remove(uid)
            period = month_key()
            if period in bd.get("pay_status", {}):
                bd["pay_status"][period].pop(uid, None)
        else:
            targets.add(uid)
            period = month_key()
            bd.setdefault("pay_status", {}).setdefault(period, {}).setdefault(uid, "unpaid")

        set_pay_targets(bd, targets)
        targets = pay_target_set(bd)
        await show_menu()
        return

    if action == "back":
        period = month_key()
        users = {uid: u for uid, u in pu.items() if uid in targets}
        status = bd.setdefault("pay_status", {}).setdefault(period, {})
        if not users:
            await q.edit_message_text("Список оплат пуст (нет получателей).", parse_mode=ParseMode.HTML)
            return

        lines = []
        buttons = [[InlineKeyboardButton('⚙️ Получатели', callback_data='paytargets:menu')]]
        for uid, u in users.items():
            st = status.get(uid, "unpaid")
            until = (bd.get("paid_until", {}) or {}).get(uid)
            if until and str(until) >= str(period):
                mark = f"✅ <i>до {until}</i>"
                status[uid] = "paid"
            else:
                mark = "✅" if st == "paid" else "❌"
            link = f"<a href='tg://user?id={uid}'>{u.get('name','Без имени')}</a>"
            lines.append(f"{link} — {mark}")
            buttons.append([
                InlineKeyboardButton("Оплатил ✅", callback_data=f"payset:paid:{uid}"),
                InlineKeyboardButton("Не оплатил ❌", callback_data=f"payset:unpaid:{uid}"),
            ])
        text = "🧾 <b>Список оплат</b> — " + period + "\n" + "\n".join(lines)
        await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(buttons))
        return

@admin_only
async def paytext(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    args = update.message.text.split(maxsplit=1)
    if len(args) == 2:
        test, err = format_pay_text(args[1], name="Test User", period="2099-12")
        if err:
            await update.message.reply_text(
                "❌ Шаблон не сохранён, потому что в нём ошибка форматирования.\n"
                f"<code>{err}</code>\n\n"
                "Разрешённые плейсхолдеры: {name}, {period} (или {0}/{1} или {} ).\n"
                "Чтобы вывести фигурные скобки — используйте {{ и }}.",
                parse_mode=ParseMode.HTML,
            )
            return
        bd["pay_text"] = args[1]
        await update.message.reply_text("✍️ Текст уведомления обновлён.")
    elif update.message.reply_to_message:
        new_tpl = update.message.reply_to_message.text_html or update.message.reply_to_message.text or bd["pay_text"]
        test, err = format_pay_text(new_tpl, name="Test User", period="2099-12")
        if err:
            await update.message.reply_text(
                "❌ Шаблон не сохранён, потому что в нём ошибка форматирования.\n"
                f"<code>{err}</code>\n\n"
                "Разрешённые плейсхолдеры: {name}, {period} (или {0}/{1} или {} ).\n"
                "Чтобы вывести фигурные скобки — используйте {{ и }}.",
                parse_mode=ParseMode.HTML,
            )
            return
        bd["pay_text"] = new_tpl
        await update.message.reply_text("✍️ Текст уведомления обновлён (из ответа).")
    else:
        await update.message.reply_text("Текущий текст:\n\n" + bd["pay_text"], parse_mode=ParseMode.HTML)

@admin_only
async def payschedule(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    jq = ctx.application.job_queue
    if jq is None:
        await update.message.reply_text(
            "❌ Планировщик (JobQueue) недоступен.\n"
            "Установи зависимости: <code>pip install \"python-telegram-bot[job-queue]\"</code>\n"
            "или хотя бы <code>pip install apscheduler</code>, затем перезапусти бота.",
            parse_mode=ParseMode.HTML,
        )
        return
    text = update.message.text
    parts = text.split()[1:]
    changed = False
    mapping = {"day": None, "prelist": None, "remind": None, "tz": None}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in mapping and v:
                mapping[k] = v
                changed = True
    if changed:
        sched = bd.setdefault("pay_schedule", {"day": 10, "prelist": "09:00", "remind": "10:00", "tz": "Asia/Novosibirsk"})
        if mapping["day"]:
            try:
                sched["day"] = max(1, min(28, int(mapping["day"])))
            except Exception:
                pass
        if mapping["prelist"]:
            sched["prelist"] = mapping["prelist"]
        if mapping["remind"]:
            sched["remind"] = mapping["remind"]
        if mapping["tz"]:
            sched["tz"] = mapping["tz"]
        # reschedule jobs
        try:
            for name in ("monthly_prelist", "monthly_payment"):
                for job in jq.get_jobs_by_name(name):
                    job.schedule_removal()
        except Exception:
            pass
        try:
            tz = ZoneInfo(sched.get("tz", "Asia/Novosibirsk"))
            hh, mm = map(int, sched.get("prelist", "09:00").split(":"))
            pre_time = dtime(hour=hh, minute=mm, tzinfo=tz)
            hh2, mm2 = map(int, sched.get("remind", "10:00").split(":"))
            rem_time = dtime(hour=hh2, minute=mm2, tzinfo=tz)
            jq.run_monthly(job_monthly_prelist, time=pre_time, day=sched.get("day",10), name='monthly_prelist', timezone=tz)
            jq.run_monthly(job_monthly_payment, time=rem_time, day=sched.get("day",10), name='monthly_payment', timezone=tz)
        except Exception as e:
            await update.message.reply_text(f"Ошибка планировщика: {e}")
            return
        await update.message.reply_text("🗓️ График обновлён.")
    else:
        sched = bd.get("pay_schedule", {})
        await update.message.reply_text(
            "Текущий график:\n"
            f"• День месяца: <b>{sched.get('day',10)}</b>\n"
            f"• Обновление списка: <b>{sched.get('prelist','09:00')}</b>\n"
            f"• Напоминания: <b>{sched.get('remind','10:00')}</b>\n"
            f"• Таймзона: <code>{sched.get('tz','Asia/Novosibirsk')}</code>\n\n"
            "Пример: <code>/payschedule day=10 prelist=09:00 remind=10:00 tz=Asia/Novosibirsk</code>",
            parse_mode=ParseMode.HTML
        )



@admin_only
async def prepay(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /prepay — предоплата. Если указать сумму в ответе на пользователя или с user_id/@username — применит сразу.
    Если без аргументов — откроет меню выбора пользователя.
    """
    bd = ensure_botdata_defaults(ctx.application)
    args = update.message.text.split()

    # Quick modes:
    # 1) reply: /prepay 1000
    if update.message.reply_to_message and len(args) == 2:
        target_uid = update.message.reply_to_message.from_user.id
        amount_str = args[1]
        return await _prepay_apply_and_report(update, ctx, bd, target_uid, amount_str)

    # 2) direct: /prepay <user_id|@username> 1000
    if len(args) >= 3:
        target = args[1]
        amount_str = args[2]
        target_uid = None
        if target.isdigit():
            target_uid = int(target)
        elif target.startswith("@"):
            for uid, u in bd.get("users", {}).items():
                if (u.get("username") or "").lower() == target.lower():
                    target_uid = int(uid)
                    break
        else:
            t2 = "@" + target
            for uid, u in bd.get("users", {}).items():
                if (u.get("username") or "").lower() == t2.lower():
                    target_uid = int(uid)
                    break
        if not target_uid:
            await update.message.reply_text("Пользователь не найден. Убедись, что он нажимал /start, или укажи user_id.")
            return ConversationHandler.END
        return await _prepay_apply_and_report(update, ctx, bd, target_uid, amount_str)

    # Interactive mode:
    ctx.user_data["prepay_page"] = 0
    ctx.user_data["prepay_query"] = ""
    kb = build_prepay_user_menu(bd, page=0, query="")
    await update.message.reply_text(
        "💳 <b>Предоплата</b>\nВыбери пользователя из списка.\n\n"
        "Можно написать в чат часть имени или @username — список отфильтруется.",
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
    )
    return PREPAY_PICK_USER


async def _prepay_apply_and_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE, bd: dict, target_uid: int, amount_str: str):
    try:
        amount = int(str(amount_str).replace("₽", "").strip())
    except Exception:
        await update.message.reply_text("Сумма должна быть числом (в рублях), например 1000")
        return ConversationHandler.END

    if amount <= 0:
        await update.message.reply_text("Сумма должна быть > 0")
        return ConversationHandler.END

    # Ensure user exists in db
    u = bd.get("users", {}).get(target_uid, {"name": f"ID {target_uid}"})
    name = u.get("name", f"ID {target_uid}")

    try:
        months, remainder, new_until, period = apply_prepay(bd, target_uid, amount)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")
        return ConversationHandler.END

    msg = (
        f"✅ Предоплата принята\n"
        f"👤 Пользователь: <a href='tg://user?id={target_uid}'>{name}</a>\n"
        f"💳 Сумма: <b>{amount} ₽</b>\n"
        f"📦 Месяцев: <b>{months}</b> (по {MONTHLY_FEE_RUB} ₽)\n"
    )
    if remainder:
        msg += f"🪙 Остаток: <b>{remainder} ₽</b> (не учтён)\n"
    msg += f"🗓 Оплачено до: <b>{new_until}</b> (включительно)"

    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    # Notify user once
    try:
        await ctx.bot.send_message(
            chat_id=target_uid,
            text=(
                f"✅ <b>Оплата получена!</b>\n\n"
                f"Сумма: <b>{amount} ₽</b>\n"
                f"Подписка активна до: <b>{new_until}</b> (включительно).\n\n"
                f"Спасибо!"
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass

    return ConversationHandler.END


async def prepay_pick_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    bd = ensure_botdata_defaults(ctx.application)

    parts = q.data.split(":")
    action = parts[1] if len(parts) > 1 else ""

    if action == "cancel":
        await q.edit_message_text("Отменено ✅")
        return ConversationHandler.END

    if action == "clear":
        ctx.user_data["prepay_query"] = ""
        ctx.user_data["prepay_page"] = 0
        kb = build_prepay_user_menu(bd, page=0, query="")
        await q.edit_message_reply_markup(reply_markup=kb)
        return PREPAY_PICK_USER

    if action == "page" and len(parts) == 3:
        page = int(parts[2])
        ctx.user_data["prepay_page"] = page
        query = ctx.user_data.get("prepay_query", "")
        kb = build_prepay_user_menu(bd, page=page, query=query)
        await q.edit_message_reply_markup(reply_markup=kb)
        return PREPAY_PICK_USER

    if action == "select" and len(parts) == 3:
        uid = int(parts[2])
        ctx.user_data["prepay_uid"] = uid
        u = bd.get("users", {}).get(uid, {"name": f"ID {uid}"})
        name = u.get("name", f"ID {uid}")
        await q.edit_message_text(
            f"👤 Выбран: <a href='tg://user?id={uid}'>{name}</a>\n\n"
            f"Введи сумму в рублях (например 1000). Цена месяца: {MONTHLY_FEE_RUB} ₽.\n"
            f"Можно отправить число одним сообщением.",
            parse_mode=ParseMode.HTML,
        )
        return PREPAY_ENTER_AMOUNT

    return PREPAY_PICK_USER


async def prepay_pick_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Filter menu by text while in PREPAY_PICK_USER."""
    bd = ensure_botdata_defaults(ctx.application)
    query = (update.message.text or "").strip()
    ctx.user_data["prepay_query"] = query
    ctx.user_data["prepay_page"] = 0
    kb = build_prepay_user_menu(bd, page=0, query=query)
    await update.message.reply_text(
        f"🔎 Фильтр: <code>{query}</code>\nВыбери пользователя:",
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
    )
    return PREPAY_PICK_USER


async def prepay_amount(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = ctx.user_data.get("prepay_uid")
    if not uid:
        await update.message.reply_text("Не выбран пользователь. Запусти /prepay заново.")
        return ConversationHandler.END
    amount_str = (update.message.text or "").strip()
    # Reuse the same apply+report logic
    return await _prepay_apply_and_report(update, ctx, bd, int(uid), amount_str)

@admin_only
async def broadcast(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    users = [uid for uid, u in bd["users"].items() if not u.get("is_admin")]
    if update.message.reply_to_message:
        text = update.message.reply_to_message.text_html or update.message.reply_to_message.text
    else:
        args = update.message.text.split(maxsplit=1)
        if len(args) < 2:
            await update.message.reply_text("Пришлите текст после команды или ответом на сообщение.")
            return
        text = args[1]
    sent = 0
    for uid in users:
        try:
            await ctx.bot.send_message(chat_id=uid, text=text, parse_mode=ParseMode.HTML)
            sent += 1
        except Exception:
            pass
    await update.message.reply_text(f"📣 Разослано: {sent}")

# --- Inbounds / clients management (сокращено для фокуса на оплатах) ---

@admin_only
async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Same as /start, but used when admin requests help
    await start(update, ctx)

@admin_only
async def inbounds(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    if not inbs:
        await update.message.reply_text("Инбаунды не найдены")
        return
    lines = []
    for ib in inbs:
        lines.append(f"ID {ib['id']}: {ib.get('remark','')} • {ib.get('protocol')} • порт {ib.get('port')}")
    await update.message.reply_text("\n".join(lines))

@admin_only
async def clients_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"cl:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Выберите инбаунд:", reply_markup=kb)

async def clients_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, _, iid = q.data.split(":")
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    if not clients:
        await q.edit_message_text("Клиентов нет")
        return ConversationHandler.END
    lines = []
    for c in clients:
        status = "✅" if c.get("enable", True) else "⛔"
        total = c.get("totalGB", 0)
        exp = c.get("expiryTime", 0)
        exp_txt = time.strftime("%Y-%m-%d", time.localtime(exp/1000)) if exp else "∞"
        gb = (total // (1024**3)) if isinstance(total, int) and total else 0
        lines.append(f"{status} {c.get('email','(без имени)')} • {c.get('id')} • {gb} GB • до {exp_txt}")
    await q.edit_message_text("\n".join(lines))
    return ConversationHandler.END

@admin_only
async def add_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"add:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Инбаунд для нового клиента:", reply_markup=kb)
    return ADD_SELECT_INBOUND

async def add_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["add_inbound_id"] = int(iid)
    await q.edit_message_text("Укажите email/имя клиента (без пробелов):")
    return ADD_EMAIL

async def add_set_email(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    email = update.message.text.strip()
    if " " in email:
        await update.message.reply_text("Без пробелов, пожалуйста. Введите снова:")
        return ADD_EMAIL
    ctx.user_data["add_email"] = email
    await update.message.reply_text(f"Трафик ГБ (по умолчанию {DEFAULT_TRAFFIC_GB}, 0 — безлимит):")
    return ADD_TRAFFIC

async def add_set_traffic(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    gb = DEFAULT_TRAFFIC_GB if txt == "" else int(txt)
    ctx.user_data["add_total_gb"] = gb
    await update.message.reply_text(f"Срок в днях (по умолчанию {DEFAULT_DAYS}, 0 — без срока):")
    return ADD_DAYS

async def add_set_days(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    days = int(update.message.text.strip() or DEFAULT_DAYS)
    ctx.user_data["add_days"] = days
    await update.message.reply_text(f"Лимит IP (по умолчанию {DEFAULT_LIMIT_IP}, 0 — без лимита):")
    return ADD_LIMITIP

async def add_set_limitip(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    limit_ip = int(update.message.text.strip() or DEFAULT_LIMIT_IP)
    ctx.user_data["add_limit_ip"] = limit_ip
    email = ctx.user_data["add_email"]
    gb = ctx.user_data["add_total_gb"]
    days = ctx.user_data["add_days"]
    limit = ctx.user_data["add_limit_ip"]
    await update.message.reply_text(
        f"Подтвердите:\nEmail: {email}\nТрафик: {gb} GB\nСрок: {days} дн\nЛимит IP: {limit}\nОтправьте 'да' или 'нет'",
    )
    return ADD_CONFIRM

async def add_confirm(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.message.text.strip().lower() not in {"да", "yes", "y", "ok", "ага"}:
        await update.message.reply_text("Отменено", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    iid = ctx.user_data["add_inbound_id"]
    email = ctx.user_data["add_email"]
    gb = ctx.user_data["add_total_gb"]
    total_bytes = 0 if gb == 0 else gb * 1024**3
    days = ctx.user_data["add_days"]
    expiry_ms = 0 if days == 0 else int((time.time() + days * 86400) * 1000)
    limit_ip = ctx.user_data["add_limit_ip"]
    await PANEL.add_client(iid, email=email, total_gb=total_bytes, expiry_ts_ms=expiry_ms, limit_ip=limit_ip)
    await update.message.reply_text("✅ Клиент добавлен")
    return ConversationHandler.END

@admin_only
async def vless_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"vl:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Выберите инбаунд:", reply_markup=kb)
    return VLESS_SELECT_INBOUND

async def vless_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["vl_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    kb = _kb([[(f"{c.get('email','')} ({c.get('id')[:8]})", f"vl:cl:{c.get('id')}")] for c in clients])
    await q.edit_message_text("Выберите клиента:", reply_markup=kb)
    return VLESS_SELECT_CLIENT

async def vless_pick_client(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, cuuid = q.data.split(":")
    iid = ctx.user_data["vl_inbound_id"]
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    client = next((c for c in clients if c.get("id") == cuuid), None)
    if not client:
        await q.edit_message_text("Клиент не найден")
        return ConversationHandler.END
    public_host = PUBLIC_HOST or httpx.URL(PANEL.base).host
    link = VlessURL.compose(ib, client_uuid=cuuid, email=client.get("email",""), public_host=public_host)
    await q.edit_message_text(f"<code>{link}</code>", parse_mode=ParseMode.HTML)
    return ConversationHandler.END

@admin_only
async def del_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"del:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Выберите инбаунд:", reply_markup=kb)
    return DEL_SELECT_INBOUND

async def del_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["del_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    kb = _kb([[(f"{c.get('email','')} ({c.get('id')[:8]})", f"del:cl:{c.get('id')}")] for c in clients])
    await q.edit_message_text("Кого удалить?", reply_markup=kb)
    return DEL_SELECT_CLIENT

async def del_pick_client(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, cuuid = q.data.split(":")
    iid = ctx.user_data["del_inbound_id"]
    await PANEL.delete_client(int(iid), cuuid)
    await q.edit_message_text("🗑️ Удалён")
    return ConversationHandler.END

@admin_only
async def toggle_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"tg:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Инбаунд:", reply_markup=kb)
    return TOGGLE_SELECT_INBOUND

async def toggle_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["tg_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    kb = _kb([[(f"{('⛔' if not c.get('enable',True) else '✅')} {c.get('email','')} ({c.get('id')[:8]})", f"tg:cl:{c.get('id')}")] for c in clients])
    await q.edit_message_text("Кого переключить?", reply_markup=kb)
    return TOGGLE_SELECT_CLIENT

async def toggle_pick_client(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, cuuid = q.data.split(":")
    iid = ctx.user_data["tg_inbound_id"]
    ib = await PANEL.inbound_get(int(iid))
    s = json.loads(ib.get("settings", "{}"))
    clients = s.get("clients", [])
    client = next((c for c in clients if c.get("id") == cuuid), None)
    if not client:
        await q.edit_message_text("Не найден")
        return ConversationHandler.END
    client["enable"] = not client.get("enable", True)
    await PANEL.update_client(int(iid), cuuid, client)
    await q.edit_message_text("Переключено")
    return ConversationHandler.END

@admin_only
async def reset_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"rs:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Инбаунд:", reply_markup=kb)
    return RESET_SELECT_INBOUND

async def reset_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["rs_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    s = json.loads(ib.get("settings", "{}"))
    clients = s.get("clients", [])
    kb = _kb([[(f"{c.get('email','')} ({c.get('id')[:8]})", f"rs:cl:{c.get('email')}")] for c in clients])
    await q.edit_message_text("Чей трафик сбросить?", reply_markup=kb)
    return RESET_SELECT_CLIENT

async def reset_pick_client(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, email = q.data.split(":")
    iid = ctx.user_data["rs_inbound_id"]
    await PANEL.reset_client_traffic(int(iid), email)
    await q.edit_message_text("Трафик сброшен")
    return ConversationHandler.END

@admin_only
async def disable_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    if not inbs:
        await update.message.reply_text("Инбаунды не найдены")
        return ConversationHandler.END
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"dis:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Инбаунд:", reply_markup=kb)
    return DISABLE_SELECT_INBOUND

async def disable_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["dis_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    if not clients:
        await q.edit_message_text("Клиентов нет")
        return ConversationHandler.END
    kb = _kb([[(f"{c.get('email','')} ({c.get('id')[:8]})", f"dis:cl:{c.get('id')}")] for c in clients])
    await q.edit_message_text("Кого выключить?", reply_markup=kb)
    return DISABLE_SELECT_CLIENT

async def disable_pick_client(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, cuuid = q.data.split(":")
    iid = ctx.user_data["dis_inbound_id"]
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    client = next((c for c in clients if c.get("id") == cuuid), None)
    if not client:
        await q.edit_message_text("Клиент не найден")
        return ConversationHandler.END
    if client.get("enable", True) is False:
        await q.edit_message_text("Уже выключен")
        return ConversationHandler.END
    client["enable"] = False
    await PANEL.update_client(int(iid), cuuid, client)
    await q.edit_message_text("⛔ Клиент выключен")
    return ConversationHandler.END

@admin_only
async def enable_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    if not inbs:
        await update.message.reply_text("Инбаунды не найдены")
        return ConversationHandler.END
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"en:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Инбаунд:", reply_markup=kb)
    return ENABLE_SELECT_INBOUND

async def enable_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["en_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    if not clients:
        await q.edit_message_text("Клиентов нет")
        return ConversationHandler.END
    kb = _kb([[(f"{c.get('email','')} ({c.get('id')[:8]})", f"en:cl:{c.get('id')}")] for c in clients])
    await q.edit_message_text("Кого включить?", reply_markup=kb)
    return ENABLE_SELECT_CLIENT

async def enable_pick_client(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, cuuid = q.data.split(":")
    iid = ctx.user_data["en_inbound_id"]
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    client = next((c for c in clients if c.get("id") == cuuid), None)
    if not client:
        await q.edit_message_text("Клиент не найден")
        return ConversationHandler.END
    if client.get("enable", True) is True:
        await q.edit_message_text("Уже включен")
        return ConversationHandler.END
    client["enable"] = True
    await PANEL.update_client(int(iid), cuuid, client)
    await q.edit_message_text("✅ Клиент включен")
    return ConversationHandler.END

@admin_only
async def disableall_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    if not inbs:
        await update.message.reply_text("Инбаунды не найдены")
        return ConversationHandler.END
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"disa:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Выберите инбаунд для массового выключения:", reply_markup=kb)
    return DISABLEALL_SELECT_INBOUND

async def disableall_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["disa_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    to_disable = [c for c in clients if c.get("enable", True)]
    ctx.user_data["disa_targets"] = [c.get("id") for c in to_disable]
    count = len(to_disable)
    await q.edit_message_text(f"Найдено клиентов к выключению: {count}. Напишите 'да' для подтверждения или что угодно для отмены.")
    return DISABLEALL_CONFIRM

async def disableall_confirm(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.message.text.strip().lower() not in {"да", "yes", "y", "ok", "ага"}:
        await update.message.reply_text("Отменено")
        return ConversationHandler.END
    iid = ctx.user_data.get("disa_inbound_id")
    targets = ctx.user_data.get("disa_targets", [])
    ib = await PANEL.inbound_get(int(iid))
    s = json.loads(ib.get("settings","{}"))
    clients = s.get("clients", [])
    done = 0
    for cuuid in targets:
        c = next((x for x in clients if x.get("id")==cuuid), None)
        if not c:
            continue
        c["enable"] = False
        try:
            await PANEL.update_client(int(iid), cuuid, c)
            done += 1
        except Exception:
            pass
    await update.message.reply_text(f"⛔ Выключено клиентов: {done}")
    return ConversationHandler.END

@admin_only
async def enableall_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    inbs = await PANEL.inbounds_list()
    if not inbs:
        await update.message.reply_text("Инбаунды не найдены")
        return ConversationHandler.END
    kb = _kb([[(f"{ib['id']} · {ib.get('remark','')} ({ib.get('protocol')})", f"ena:ib:{ib['id']}")] for ib in inbs])
    await update.message.reply_text("Выберите инбаунд для массового включения:", reply_markup=kb)
    return ENABLEALL_SELECT_INBOUND

async def enableall_pick_inbound(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    _, _, iid = q.data.split(":")
    ctx.user_data["ena_inbound_id"] = int(iid)
    ib = await PANEL.inbound_get(int(iid))
    try:
        s = json.loads(ib.get("settings", "{}"))
        clients = s.get("clients", [])
    except json.JSONDecodeError:
        clients = []
    to_enable = [c for c in clients if not c.get("enable", True)]
    ctx.user_data["ena_targets"] = [c.get("id") for c in to_enable]
    count = len(to_enable)
    await q.edit_message_text(f"Найдено клиентов к включению: {count}. Напишите 'да' для подтверждения или что угодно для отмены.")
    return ENABLEALL_CONFIRM

async def enableall_confirm(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.message.text.strip().lower() not in {"да", "yes", "y", "ok", "ага"}:
        await update.message.reply_text("Отменено")
        return ConversationHandler.END
    iid = ctx.user_data.get("ena_inbound_id")
    targets = ctx.user_data.get("ena_targets", [])
    ib = await PANEL.inbound_get(int(iid))
    s = json.loads(ib.get("settings","{}"))
    clients = s.get("clients", [])
    done = 0
    for cuuid in targets:
        c = next((x for x in clients if x.get("id")==cuuid), None)
        if not c:
            continue
        c["enable"] = True
        try:
            await PANEL.update_client(int(iid), cuuid, c)
            done += 1
        except Exception:
            pass
    await update.message.reply_text(f"✅ Включено клиентов: {done}")
    return ConversationHandler.END

@admin_only
async def online(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        emails = await PANEL.onlines()
        if not emails:
            await update.message.reply_text("Никто не онлайн")
            return
        await update.message.reply_text("Онлайн:\n" + "\n".join(emails))
    except Exception:
        await update.message.reply_text("Эндпоинт onlines недоступен на вашей версии панели")

# --- Error handler
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    try:
        text = f"Ошибка: {err}"
        if isinstance(update, Update):
            if update.callback_query:
                await update.callback_query.edit_message_text(f"❌ {text}")
            elif update.effective_message:
                await update.effective_message.reply_text(f"❌ {text}")
    finally:
        print("Exception in handler:", err)

# --- Jobs
async def job_monthly_prelist(context: ContextTypes.DEFAULT_TYPE):
    app = context.application
    bd = ensure_botdata_defaults(app)
    sched = bd.get("pay_schedule", {})
    tz = ZoneInfo(sched.get("tz", "Asia/Novosibirsk"))
    period = month_key(datetime.now(tz))
    pu = non_admin_users(bd)
    targets = pay_target_set(bd)
    users = {uid: u for uid, u in pu.items() if uid in targets}
    bd.setdefault("pay_status", {})[period] = {uid: ("paid" if is_prepaid(bd, uid, period) else "unpaid") for uid in users.keys()}
    if not ADMIN_IDS:
        return
    lines = []
    buttons = []
    for uid, u in users.items():
        link = f"<a href='tg://user?id={uid}'>{u.get('name','Без имени')}</a>"
        until = (bd.get("paid_until", {}) or {}).get(uid)
        if until and str(until) >= str(period):
            lines.append(f"{link} — ✅ <i>до {until}</i>")
        else:
            lines.append(f"{link} — ❌")
        buttons.append([
            InlineKeyboardButton(f"Оплатил ✅", callback_data=f"payset:paid:{uid}"),
            InlineKeyboardButton(f"Не оплатил ❌", callback_data=f"payset:unpaid:{uid}"),
        ])
    text = "🧾 <b>Список оплат</b> — " + period + "\n" + "\n".join(lines)
    for admin_id in ADMIN_IDS:
        try:
            await context.bot.send_message(chat_id=admin_id, text=text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(buttons))
        except Exception:
            pass

async def job_monthly_payment(context: ContextTypes.DEFAULT_TYPE):
    app = context.application
    bd = ensure_botdata_defaults(app)
    sched = bd.get("pay_schedule", {})
    tz = ZoneInfo(sched.get("tz", "Asia/Novosibirsk"))
    period = month_key(datetime.now(tz))
    pu = non_admin_users(bd)
    targets = pay_target_set(bd)
    users = {uid: u for uid, u in pu.items() if uid in targets}
    status = bd.setdefault("pay_status", {}).setdefault(period, {})
    for uid, u in users.items():
        # paid if prepaid for this month
        if is_prepaid(bd, uid, period):
            status[uid] = "paid"
            continue
        status.setdefault(uid, "unpaid")
        if status.get(uid) == "paid":
            continue
        try:
            txt, kb = render_payment_message(bd, {"id":uid,"name":u.get("name","Пользователь")}, period)
            await context.bot.send_message(chat_id=uid, text=txt, reply_markup=kb, parse_mode=ParseMode.HTML)
        except Exception:
            pass

# --- App
async def _post_init(app):
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        print("Webhook deleted (if any). Switching to polling.")
        bd = ensure_botdata_defaults(app)
        sched = bd.get("pay_schedule", {"day":10, "prelist":"09:00", "remind":"10:00", "tz":"Asia/Novosibirsk"})
        jq = app.job_queue
        if jq is None:
            print("JobQueue is None -> install python-telegram-bot[job-queue] or apscheduler to enable scheduling.")
            return
        tz = ZoneInfo(sched.get("tz","Asia/Novosibirsk"))
        # Prelist job
        hh, mm = map(int, sched.get("prelist","09:00").split(":"))
        pre_time = dtime(hour=hh, minute=mm, tzinfo=tz)
        jq.run_monthly(job_monthly_prelist, time=pre_time, day=sched.get("day",10), name='monthly_prelist', timezone=tz)
        # Reminder job
        hh2, mm2 = map(int, sched.get("remind","10:00").split(":"))
        rem_time = dtime(hour=hh2, minute=mm2, tzinfo=tz)
        jq.run_monthly(job_monthly_payment, time=rem_time, day=sched.get("day",10), name='monthly_payment', timezone=tz)
        print(f"Monthly jobs scheduled: day={sched.get('day',10)} prelist={sched.get('prelist','09:00')} remind={sched.get('remind','10:00')} tz={sched.get('tz','Asia/Novosibirsk')}")
    except Exception as e:
        print("post_init warning:", e)

def main():
    persistence = PicklePersistence(filepath='bot_data.pkl')
    builder = ApplicationBuilder().token(TG_TOKEN).persistence(persistence)
    if AIORateLimiter:
        builder = builder.rate_limiter(AIORateLimiter())
    app = builder.build()

    app.add_error_handler(on_error)

    # Base
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("info", info_cmd))
    app.add_handler(CommandHandler("contact", contact_cmd))

    # Payments
    app.add_handler(CommandHandler("paylist", paylist))
    app.add_handler(CommandHandler("paytext", paytext))
    app.add_handler(CommandHandler("payschedule", payschedule))
    prepay_conv = ConversationHandler(
        entry_points=[CommandHandler("prepay", prepay)],
        states={
            PREPAY_PICK_USER: [
                CallbackQueryHandler(prepay_pick_cb, pattern=r"^prepay:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, prepay_pick_text),
            ],
            PREPAY_ENTER_AMOUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, prepay_amount),
                CallbackQueryHandler(prepay_pick_cb, pattern=r"^prepay:"),
            ],
        },
        fallbacks=[CallbackQueryHandler(prepay_pick_cb, pattern=r"^prepay:cancel$")],
        allow_reentry=True,
    )
    app.add_handler(prepay_conv)
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(CallbackQueryHandler(cb_pay, pattern=r"^pay:"))
    app.add_handler(CallbackQueryHandler(cb_payset, pattern=r"^payset:"))
    app.add_handler(CallbackQueryHandler(cb_paytargets, pattern=r"^paytargets:"))

    # Inbounds/clients
    app.add_handler(CommandHandler("inbounds", inbounds))
    app.add_handler(CommandHandler("clients", clients_entry))
    app.add_handler(CallbackQueryHandler(clients_pick_inbound, pattern=r"^cl:ib:"))

    add_conv = ConversationHandler(
        entry_points=[CommandHandler("add", add_entry)],
        states={
            ADD_SELECT_INBOUND: [CallbackQueryHandler(add_pick_inbound, pattern=r"^add:ib:")],
            ADD_EMAIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_set_email)],
            ADD_TRAFFIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_set_traffic)],
            ADD_DAYS: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_set_days)],
            ADD_LIMITIP: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_set_limitip)],
            ADD_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_confirm)],
        },
        fallbacks=[],
        allow_reentry=True,
    )
    app.add_handler(add_conv)

    vless_conv = ConversationHandler(
        entry_points=[CommandHandler("vless", vless_entry)],
        states={
            VLESS_SELECT_INBOUND: [CallbackQueryHandler(vless_pick_inbound, pattern=r"^vl:ib:")],
            VLESS_SELECT_CLIENT: [CallbackQueryHandler(vless_pick_client, pattern=r"^vl:cl:")],
        },
        fallbacks=[],
    )
    app.add_handler(vless_conv)

    del_conv = ConversationHandler(
        entry_points=[CommandHandler("del", del_entry)],
        states={
            DEL_SELECT_INBOUND: [CallbackQueryHandler(del_pick_inbound, pattern=r"^del:ib:")],
            DEL_SELECT_CLIENT: [CallbackQueryHandler(del_pick_client, pattern=r"^del:cl:")],
        },
        fallbacks=[],
    )
    app.add_handler(del_conv)

    toggle_conv = ConversationHandler(
        entry_points=[CommandHandler("toggle", toggle_entry)],
        states={
            TOGGLE_SELECT_INBOUND: [CallbackQueryHandler(toggle_pick_inbound, pattern=r"^tg:ib:")],
            TOGGLE_SELECT_CLIENT: [CallbackQueryHandler(toggle_pick_client, pattern=r"^tg:cl:")],
        },
        fallbacks=[],
    )
    app.add_handler(toggle_conv)

    reset_conv = ConversationHandler(
        entry_points=[CommandHandler("reset", reset_entry)],
        states={
            RESET_SELECT_INBOUND: [CallbackQueryHandler(reset_pick_inbound, pattern=r"^rs:ib:")],
            RESET_SELECT_CLIENT: [CallbackQueryHandler(reset_pick_client, pattern=r"^rs:cl:")],
        },
        fallbacks=[],
    )
    app.add_handler(reset_conv)

    disable_conv = ConversationHandler(
        entry_points=[CommandHandler("disable", disable_entry)],
        states={
            DISABLE_SELECT_INBOUND: [CallbackQueryHandler(disable_pick_inbound, pattern=r"^dis:ib:")],
            DISABLE_SELECT_CLIENT: [CallbackQueryHandler(disable_pick_client, pattern=r"^dis:cl:")],
        },
        fallbacks=[],
    )
    app.add_handler(disable_conv)

    enable_conv = ConversationHandler(
        entry_points=[CommandHandler("enable", enable_entry)],
        states={
            ENABLE_SELECT_INBOUND: [CallbackQueryHandler(enable_pick_inbound, pattern=r"^en:ib:")],
            ENABLE_SELECT_CLIENT: [CallbackQueryHandler(enable_pick_client, pattern=r"^en:cl:")],
        },
        fallbacks=[],
    )
    app.add_handler(enable_conv)

    disableall_conv = ConversationHandler(
        entry_points=[CommandHandler("disableall", disableall_entry)],
        states={
            DISABLEALL_SELECT_INBOUND: [CallbackQueryHandler(disableall_pick_inbound, pattern=r"^disa:ib:")],
            DISABLEALL_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, disableall_confirm)],
        },
        fallbacks=[],
    )
    app.add_handler(disableall_conv)

    enableall_conv = ConversationHandler(
        entry_points=[CommandHandler("enableall", enableall_entry)],
        states={
            ENABLEALL_SELECT_INBOUND: [CallbackQueryHandler(enableall_pick_inbound, pattern=r"^ena:ib:")],
            ENABLEALL_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, enableall_confirm)],
        },
        fallbacks=[],
    )
    app.add_handler(enableall_conv)

    app.add_handler(CommandHandler("online", online))

    app.post_init = _post_init
    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
