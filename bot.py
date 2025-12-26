# coding: utf-8
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
    """Add N months to YYYY-MM, returning YYYY-MM. months can be negative."""
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
    until = (bot_data.get("paid_until", {}) or {}).get(uid)
    if not until:
        return False
    return str(until) >= str(period)

def apply_prepay(bot_data: dict, uid: int, amount_rub: int) -> tuple[int, int, str, str]:
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
        label = f"{name} ({uname})" if uname else name
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
    try:
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

(ADD_SELECT_INBOUND, ADD_EMAIL, ADD_TRAFFIC, ADD_DAYS, ADD_LIMITIP, ADD_CONFIRM,
 VLESS_SELECT_INBOUND, VLESS_SELECT_CLIENT,
 DEL_SELECT_INBOUND, DEL_SELECT_CLIENT,
 TOGGLE_SELECT_INBOUND, TOGGLE_SELECT_CLIENT,
 RESET_SELECT_INBOUND, RESET_SELECT_CLIENT,
 DISABLE_SELECT_INBOUND, DISABLE_SELECT_CLIENT,
 ENABLE_SELECT_INBOUND, ENABLE_SELECT_CLIENT,
 DISABLEALL_SELECT_INBOUND, DISABLEALL_CONFIRM,
 ENABLEALL_SELECT_INBOUND, ENABLEALL_CONFIRM) = range(22)

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


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bd = ensure_botdata_defaults(ctx.application)
    bd["users"].setdefault(user.id, {"name": user.full_name, "username": ("@"+user.username) if user.username else None, "is_admin": user.id in ADMIN_IDS})
    kb = ReplyKeyboardMarkup([["/info", "/contact"]], resize_keyboard=True)
    if user.id in ADMIN_IDS:
        await update.message.reply_text(
            "👋 <b>Привет, админ!</b>\n"
            "Ниже — полный список возможностей бота.\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🧾 <b>Оплаты и уведомления</b>\n\n"
            "• /paylist — список оплат за текущий период\n"
            "• /paytext — изменить шаблон уведомления\n"
            "• /payschedule — изменить дату/время\n"
            "• /prepay — предоплата по сумме\n"
            "• /prepayinfo — посмотреть «оплачено до»\n"
            "• /prepayminus — отнять N месяцев (откатить ошибку)\n"
            "• /prepayset — поставить точный месяц YYYY-MM\n"
            "• /prepayclear — удалить предоплату\n"
            "• /broadcast — рассылка\n\n"
            "🧩 <b>3x-ui / X-UI</b>\n"
            "• /inbounds, /clients, /add, /del, /toggle, /disable, /enable, /reset, /vless, /online\n\n"
            "⚠️ Если планировщик не работает: <code>pip install \"python-telegram-bot[job-queue]\"</code>\n",
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
        )
    else:
        await update.message.reply_text(
            "Добро пожаловать.\nТут ты получаешь ежемeсячные уведомления об оплате.\n\nДоступно: /info и /contact",
            reply_markup=kb
        )

async def info_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ℹ️ Информация: подписка активна после оплаты. Уведомления приходят 10 числа.")

async def contact_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_IDS:
        await update.message.reply_text("Администратор не задан.")
        return
    await update.message.reply_text("✉️ Написать администратору: @dan_sava", parse_mode=ParseMode.HTML)


async def cb_pay(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data.split(":")
    if len(data) != 3:
        return
    action, uid = data[1], int(data[2])
    bd = ensure_botdata_defaults(ctx.application)
    user = bd.get("users", {}).get(uid, {"id": uid, "name": "Пользователь"})
    period = month_key()
    status = bd.setdefault("pay_status", {}).setdefault(period, {})
    if action == "remind":
        async def send_reminder(ctx2):
            txt, kb = render_payment_message(bd, {"id":uid,"name":user.get("name","Пользователь")}, period)
            await ctx2.bot.send_message(chat_id=uid, text=txt, reply_markup=kb, parse_mode=ParseMode.HTML)
        jq = ctx.application.job_queue
        if jq is None:
            await q.edit_message_text("❌ Планировщик недоступен (нет JobQueue).", parse_mode=ParseMode.HTML)
            return
        jq.run_once(send_reminder, when=5*60*60, name=f"pay_remind_{uid}")
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

    parts = q.data.split(":")
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
        set_pay_targets(bd, set(pu.keys()))
        period = month_key()
        st = bd.setdefault("pay_status", {}).setdefault(period, {})
        for uid in pu.keys():
            st.setdefault(uid, "unpaid")
        targets = pay_target_set(bd)
        await show_menu()
        return

    if action == "all_off":
        set_pay_targets(bd, set())
        period = month_key()
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
            "и перезапусти бота.",
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
            jq.run_monthly(job_monthly_prelist, when=pre_time, day=sched.get("day",10), name='monthly_prelist')
            jq.run_monthly(job_monthly_payment, when=rem_time, day=sched.get("day",10), name='monthly_payment')
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


# ---- PREPAY undo tools (admin) -----------------------------------------------

def _parse_ym(s: str) -> str | None:
    s = (s or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2))
    if mo < 1 or mo > 12:
        return None
    return f"{y:04d}-{mo:02d}"

def _resolve_uid_from_args_or_reply(update: Update, bd: dict) -> int | None:
    if update.message and update.message.reply_to_message:
        return update.message.reply_to_message.from_user.id
    if not update.message:
        return None
    args = update.message.text.split()
    if len(args) < 2:
        return None
    target = args[1].strip()
    if target.isdigit():
        return int(target)
    if not target.startswith("@"):
        target = "@" + target
    target = target.lower()
    for uid, u in (bd.get("users", {}) or {}).items():
        if (u.get("username") or "").lower() == target:
            return int(uid)
    return None

def _sync_current_period_status(bd: dict, uid: int):
    period = current_period_from_botdata(bd)
    st = bd.setdefault("pay_status", {}).setdefault(period, {})
    st[uid] = "paid" if is_prepaid(bd, uid, period) else "unpaid"

@admin_only
async def prepayinfo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    if not uid:
        await update.message.reply_text("Использование: /prepayinfo (ответом) или /prepayinfo <user_id|@username>")
        return
    u = bd.get("users", {}).get(uid, {"name": f"ID {uid}"})
    name = u.get("name", f"ID {uid}")
    until = (bd.get("paid_until", {}) or {}).get(uid)
    period = current_period_from_botdata(bd)
    msg = (
        f"👤 <a href='tg://user?id={uid}'>{name}</a>\n"
        f"📆 Текущий период: <b>{period}</b>\n"
        f"🗓 Оплачено до: <b>{until or '—'}</b>\n"
        f"✅ Предоплачен сейчас: <b>{'да' if is_prepaid(bd, uid, period) else 'нет'}</b>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

@admin_only
async def prepayset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    args = update.message.text.split()
    ym_arg = args[1] if (update.message.reply_to_message and len(args) >= 2) else (args[2] if len(args) >= 3 else None)
    ym = _parse_ym(ym_arg or "")
    if not uid or not ym:
        await update.message.reply_text("Использование: /prepayset 2026-03 (ответом) или /prepayset <user_id|@username> 2026-03")
        return
    bd.setdefault("paid_until", {})[uid] = ym
    _sync_current_period_status(bd, uid)
    await update.message.reply_text(f"✅ Обновлено: оплачено до <b>{ym}</b>", parse_mode=ParseMode.HTML)

@admin_only
async def prepayminus(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    args = update.message.text.split()
    n_arg = args[1] if (update.message.reply_to_message and len(args) >= 2) else (args[2] if len(args) >= 3 else None)
    if not uid or not n_arg or not str(n_arg).lstrip("-").isdigit():
        await update.message.reply_text("Использование: /prepayminus 5 (ответом) или /prepayminus <user_id|@username> 5")
        return
    n = int(n_arg)
    if n <= 0:
        await update.message.reply_text("Число месяцев должно быть > 0")
        return
    until = (bd.get("paid_until", {}) or {}).get(uid)
    if not until:
        await update.message.reply_text("У пользователя нет предоплаты (paid_until пуст).")
        return
    new_until = add_months_ym(str(until), -n)
    period = current_period_from_botdata(bd)
    if str(new_until) < str(period):
        bd.setdefault("paid_until", {}).pop(uid, None)
    else:
        bd.setdefault("paid_until", {})[uid] = new_until
    _sync_current_period_status(bd, uid)
    now_until = (bd.get("paid_until", {}) or {}).get(uid)
    await update.message.reply_text(f"✅ Готово: было <b>{until}</b> → стало <b>{now_until or '—'}</b>", parse_mode=ParseMode.HTML)

@admin_only
async def prepayclear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    if not uid:
        await update.message.reply_text("Использование: /prepayclear (ответом) или /prepayclear <user_id|@username>")
        return
    bd.setdefault("paid_until", {}).pop(uid, None)
    _sync_current_period_status(bd, uid)
    await update.message.reply_text("✅ Предоплата очищена (paid_until удалён).")


@admin_only
async def prepay(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    args = update.message.text.split()

    if update.message.reply_to_message and len(args) == 2:
        target_uid = update.message.reply_to_message.from_user.id
        amount_str = args[1]
        return await _prepay_apply_and_report(update, ctx, bd, target_uid, amount_str)

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

    ctx.user_data["prepay_page"] = 0
    ctx.user_data["prepay_query"] = ""
    kb = build_prepay_user_menu(bd, page=0, query="")
    await update.message.reply_text(
        "💳 <b>Предоплата</b>\nВыбери пользователя из списка.\n\n"
        "Можно написать часть имени или @username — список отфильтруется.",
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
            f"Введи сумму в рублях (например 1000). Цена месяца: {MONTHLY_FEE_RUB} ₽.",
            parse_mode=ParseMode.HTML,
        )
        return PREPAY_ENTER_AMOUNT

    return PREPAY_PICK_USER

async def prepay_pick_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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


@admin_only
async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await start(update, ctx)

# --- дальше: inbounds/clients/прочие handlers + jobs + main() ---
# (оставлено как у тебя; в файле по ссылке bot_updated.py — всё целиком)
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
    """Add N months to YYYY-MM, returning YYYY-MM. months can be negative."""
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
    until = (bot_data.get("paid_until", {}) or {}).get(uid)
    if not until:
        return False
    return str(until) >= str(period)

def apply_prepay(bot_data: dict, uid: int, amount_rub: int) -> tuple[int, int, str, str]:
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
        label = f"{name} ({uname})" if uname else name
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
    try:
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

(ADD_SELECT_INBOUND, ADD_EMAIL, ADD_TRAFFIC, ADD_DAYS, ADD_LIMITIP, ADD_CONFIRM,
 VLESS_SELECT_INBOUND, VLESS_SELECT_CLIENT,
 DEL_SELECT_INBOUND, DEL_SELECT_CLIENT,
 TOGGLE_SELECT_INBOUND, TOGGLE_SELECT_CLIENT,
 RESET_SELECT_INBOUND, RESET_SELECT_CLIENT,
 DISABLE_SELECT_INBOUND, DISABLE_SELECT_CLIENT,
 ENABLE_SELECT_INBOUND, ENABLE_SELECT_CLIENT,
 DISABLEALL_SELECT_INBOUND, DISABLEALL_CONFIRM,
 ENABLEALL_SELECT_INBOUND, ENABLEALL_CONFIRM) = range(22)

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


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bd = ensure_botdata_defaults(ctx.application)
    bd["users"].setdefault(user.id, {"name": user.full_name, "username": ("@"+user.username) if user.username else None, "is_admin": user.id in ADMIN_IDS})
    kb = ReplyKeyboardMarkup([["/info", "/contact"]], resize_keyboard=True)
    if user.id in ADMIN_IDS:
        await update.message.reply_text(
            "👋 <b>Привет, админ!</b>\n"
            "Ниже — полный список возможностей бота.\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🧾 <b>Оплаты и уведомления</b>\n\n"
            "• /paylist — список оплат за текущий период\n"
            "• /paytext — изменить шаблон уведомления\n"
            "• /payschedule — изменить дату/время\n"
            "• /prepay — предоплата по сумме\n"
            "• /prepayinfo — посмотреть «оплачено до»\n"
            "• /prepayminus — отнять N месяцев (откатить ошибку)\n"
            "• /prepayset — поставить точный месяц YYYY-MM\n"
            "• /prepayclear — удалить предоплату\n"
            "• /broadcast — рассылка\n\n"
            "🧩 <b>3x-ui / X-UI</b>\n"
            "• /inbounds, /clients, /add, /del, /toggle, /disable, /enable, /reset, /vless, /online\n\n"
            "⚠️ Если планировщик не работает: <code>pip install \"python-telegram-bot[job-queue]\"</code>\n",
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
        )
    else:
        await update.message.reply_text(
            "Добро пожаловать.\nТут ты получаешь ежемeсячные уведомления об оплате.\n\nДоступно: /info и /contact",
            reply_markup=kb
        )

async def info_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ℹ️ Информация: подписка активна после оплаты. Уведомления приходят 10 числа.")

async def contact_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ADMIN_IDS:
        await update.message.reply_text("Администратор не задан.")
        return
    await update.message.reply_text("✉️ Написать администратору: @dan_sava", parse_mode=ParseMode.HTML)


async def cb_pay(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data.split(":")
    if len(data) != 3:
        return
    action, uid = data[1], int(data[2])
    bd = ensure_botdata_defaults(ctx.application)
    user = bd.get("users", {}).get(uid, {"id": uid, "name": "Пользователь"})
    period = month_key()
    status = bd.setdefault("pay_status", {}).setdefault(period, {})
    if action == "remind":
        async def send_reminder(ctx2):
            txt, kb = render_payment_message(bd, {"id":uid,"name":user.get("name","Пользователь")}, period)
            await ctx2.bot.send_message(chat_id=uid, text=txt, reply_markup=kb, parse_mode=ParseMode.HTML)
        jq = ctx.application.job_queue
        if jq is None:
            await q.edit_message_text("❌ Планировщик недоступен (нет JobQueue).", parse_mode=ParseMode.HTML)
            return
        jq.run_once(send_reminder, when=5*60*60, name=f"pay_remind_{uid}")
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

    parts = q.data.split(":")
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
        set_pay_targets(bd, set(pu.keys()))
        period = month_key()
        st = bd.setdefault("pay_status", {}).setdefault(period, {})
        for uid in pu.keys():
            st.setdefault(uid, "unpaid")
        targets = pay_target_set(bd)
        await show_menu()
        return

    if action == "all_off":
        set_pay_targets(bd, set())
        period = month_key()
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
            "и перезапусти бота.",
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
            jq.run_monthly(job_monthly_prelist, when=pre_time, day=sched.get("day",10), name='monthly_prelist')
            jq.run_monthly(job_monthly_payment, when=rem_time, day=sched.get("day",10), name='monthly_payment')
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


# ---- PREPAY undo tools (admin) -----------------------------------------------

def _parse_ym(s: str) -> str | None:
    s = (s or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2))
    if mo < 1 or mo > 12:
        return None
    return f"{y:04d}-{mo:02d}"

def _resolve_uid_from_args_or_reply(update: Update, bd: dict) -> int | None:
    if update.message and update.message.reply_to_message:
        return update.message.reply_to_message.from_user.id
    if not update.message:
        return None
    args = update.message.text.split()
    if len(args) < 2:
        return None
    target = args[1].strip()
    if target.isdigit():
        return int(target)
    if not target.startswith("@"):
        target = "@" + target
    target = target.lower()
    for uid, u in (bd.get("users", {}) or {}).items():
        if (u.get("username") or "").lower() == target:
            return int(uid)
    return None

def _sync_current_period_status(bd: dict, uid: int):
    period = current_period_from_botdata(bd)
    st = bd.setdefault("pay_status", {}).setdefault(period, {})
    st[uid] = "paid" if is_prepaid(bd, uid, period) else "unpaid"

@admin_only
async def prepayinfo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    if not uid:
        await update.message.reply_text("Использование: /prepayinfo (ответом) или /prepayinfo <user_id|@username>")
        return
    u = bd.get("users", {}).get(uid, {"name": f"ID {uid}"})
    name = u.get("name", f"ID {uid}")
    until = (bd.get("paid_until", {}) or {}).get(uid)
    period = current_period_from_botdata(bd)
    msg = (
        f"👤 <a href='tg://user?id={uid}'>{name}</a>\n"
        f"📆 Текущий период: <b>{period}</b>\n"
        f"🗓 Оплачено до: <b>{until or '—'}</b>\n"
        f"✅ Предоплачен сейчас: <b>{'да' if is_prepaid(bd, uid, period) else 'нет'}</b>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

@admin_only
async def prepayset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    args = update.message.text.split()
    ym_arg = args[1] if (update.message.reply_to_message and len(args) >= 2) else (args[2] if len(args) >= 3 else None)
    ym = _parse_ym(ym_arg or "")
    if not uid or not ym:
        await update.message.reply_text("Использование: /prepayset 2026-03 (ответом) или /prepayset <user_id|@username> 2026-03")
        return
    bd.setdefault("paid_until", {})[uid] = ym
    _sync_current_period_status(bd, uid)
    await update.message.reply_text(f"✅ Обновлено: оплачено до <b>{ym}</b>", parse_mode=ParseMode.HTML)

@admin_only
async def prepayminus(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    args = update.message.text.split()
    n_arg = args[1] if (update.message.reply_to_message and len(args) >= 2) else (args[2] if len(args) >= 3 else None)
    if not uid or not n_arg or not str(n_arg).lstrip("-").isdigit():
        await update.message.reply_text("Использование: /prepayminus 5 (ответом) или /prepayminus <user_id|@username> 5")
        return
    n = int(n_arg)
    if n <= 0:
        await update.message.reply_text("Число месяцев должно быть > 0")
        return
    until = (bd.get("paid_until", {}) or {}).get(uid)
    if not until:
        await update.message.reply_text("У пользователя нет предоплаты (paid_until пуст).")
        return
    new_until = add_months_ym(str(until), -n)
    period = current_period_from_botdata(bd)
    if str(new_until) < str(period):
        bd.setdefault("paid_until", {}).pop(uid, None)
    else:
        bd.setdefault("paid_until", {})[uid] = new_until
    _sync_current_period_status(bd, uid)
    now_until = (bd.get("paid_until", {}) or {}).get(uid)
    await update.message.reply_text(f"✅ Готово: было <b>{until}</b> → стало <b>{now_until or '—'}</b>", parse_mode=ParseMode.HTML)

@admin_only
async def prepayclear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    uid = _resolve_uid_from_args_or_reply(update, bd)
    if not uid:
        await update.message.reply_text("Использование: /prepayclear (ответом) или /prepayclear <user_id|@username>")
        return
    bd.setdefault("paid_until", {}).pop(uid, None)
    _sync_current_period_status(bd, uid)
    await update.message.reply_text("✅ Предоплата очищена (paid_until удалён).")


@admin_only
async def prepay(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bd = ensure_botdata_defaults(ctx.application)
    args = update.message.text.split()

    if update.message.reply_to_message and len(args) == 2:
        target_uid = update.message.reply_to_message.from_user.id
        amount_str = args[1]
        return await _prepay_apply_and_report(update, ctx, bd, target_uid, amount_str)

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

    ctx.user_data["prepay_page"] = 0
    ctx.user_data["prepay_query"] = ""
    kb = build_prepay_user_menu(bd, page=0, query="")
    await update.message.reply_text(
        "💳 <b>Предоплата</b>\nВыбери пользователя из списка.\n\n"
        "Можно написать часть имени или @username — список отфильтруется.",
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
            f"Введи сумму в рублях (например 1000). Цена месяца: {MONTHLY_FEE_RUB} ₽.",
            parse_mode=ParseMode.HTML,
        )
        return PREPAY_ENTER_AMOUNT

    return PREPAY_PICK_USER

async def prepay_pick_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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


@admin_only
async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await start(update, ctx)

# --- дальше: inbounds/clients/прочие handlers + jobs + main() ---
# (оставлено как у тебя; в файле по ссылке bot_updated.py — всё целиком)