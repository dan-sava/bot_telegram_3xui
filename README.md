# bot_telegram_3xui
---

## Структура проекта

```
.
├── bot.py              # основной файл бота
├── requirements.txt    # зависимости
├── .env                # переменные окружения (НЕ коммитить)
└── bot_data.pkl        # база данных бота (PicklePersistence) (НЕ коммитить)
```

---

## Установка и запуск (с нуля)

### 1) Установи Python 3.11

**Ubuntu/Debian пример:**
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
```

Проверка:
```bash
python3.11 --version
```

---

### 2) Клонируй репозиторий
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

---

### 3) Создай виртуальное окружение (venv)

**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Проверка:
```bash
python -V
pip -V
```

---

### 4) Установи зависимости

```bash
pip install -r requirements.txt
```

#### Планировщик (важно)
Чтобы `app.job_queue` был доступен, установи extra **job-queue**:

```bash
pip install "python-telegram-bot[job-queue]==21.6"
```

> Рекомендация для GitHub: можно объединить extras и закрепить версию так:
> `python-telegram-bot[rate-limiter,job-queue]==21.6`

---

### 5) Настрой `.env`

Создай файл `.env` (см. `.env.example`) и заполни:

- `TG_TOKEN` — токен бота
- `ADMIN_IDS` — id админов через запятую (числа)
- `PANEL_BASE` — URL панели 3x-ui, например `https://example.com/`
- `PANEL_USERNAME`, `PANEL_PASSWORD`
- `PUBLIC_HOST` — домен/IP для VLESS ссылок (если пусто — берётся host из PANEL_BASE)

---

### 6) Запусти бота

```bash
python bot.py
```

В Telegram:
1) напиши боту `/start`
2) админ увидит расширенную справку и команды

# 3x-ui Telegram Bot (X-UI / 3x-ui) + Оплаты/Напоминания
---   

Telegram‑бот для **управления 3x-ui (X-UI)** и **учёта оплат** (ежемесячные напоминания + предоплата на N месяцев).

## Возможности

### Для пользователей
- `/info` — краткая информация
- `/contact` — как связаться с админом (ссылка/ник)

### Для администратора
**3x-ui / X-UI**
- `/inbounds` — список инбаундов
- `/clients` — список клиентов в инбаунде
- `/add` — добавить клиента (диалог)
- `/del` — удалить клиента
- `/toggle` — включить/выключить клиента
- `/disable` — выключить клиента
- `/enable` — включить клиента
- `/reset` — сбросить трафик клиента
- `/vless` — получить VLESS URL
- `/online` — онлайн‑клиенты (если поддерживается панелью)
- `/disableall` — выключить всех в инбаунде
- `/enableall` — включить всех в инбаунде

**Оплаты**
- `/paylist` — список оплат за текущий период (кнопки «Оплатил/Не оплатил» + ⚙️ «Получатели»)
- `/paytext` — редактирование шаблона уведомления об оплате (HTML поддерживается)
- `/payschedule` — изменить дату/время ежемесячных задач (Новосибирск по умолчанию)
- `/prepay` — предоплата по сумме (авто‑продление на N месяцев, без ежемесячных уведомлений)
- `/broadcast` — разовое сообщение всем пользователям

---

## Требования

- **Python 3.11+**
- Доступ к панели **3x-ui / X-UI** по HTTP(S)
- Telegram Bot Token (BotFather)
- Установленный `pip`

> ⚠️ Для работы планировщика (ежемесячные напоминания) нужен JobQueue (APScheduler).
> См. раздел **Планировщик** ниже.


---

## Переменные окружения (.env)

| Переменная | Обязательная | Пример | Описание |
|---|---:|---|---|
| `TG_TOKEN` | ✅ | `123:ABC...` | токен Telegram бота |
| `ADMIN_IDS` | ✅ | `11111111,22222222` | id админов, числовые |
| `PANEL_BASE` | ✅ | `https://panel.example.com/` | базовый URL 3x-ui |
| `PANEL_USERNAME` | ✅ | `admin` | логин панели |
| `PANEL_PASSWORD` | ✅ | `password` | пароль панели |
| `PANEL_LOGIN_PATH` | ❌ | `login` | кастомный путь логина (если нужен) |
| `VERIFY_TLS` | ❌ | `1` / `0` | проверка TLS сертификата (0 — отключить) |
| `PUBLIC_HOST` | ❌ | `vpn.example.com` | хост для VLESS URL |
| `DEFAULT_TRAFFIC_GB` | ❌ | `0` | дефолт трафика при /add (0 = безлимит) |
| `DEFAULT_DAYS` | ❌ | `0` | дефолт срок в днях при /add (0 = без срока) |
| `DEFAULT_LIMIT_IP` | ❌ | `0` | дефолт лимит IP при /add |

---

## Планировщик (ежемесячные задачи)

По умолчанию:
- **10 числа в 09:00** (Asia/Novosibirsk): обновление статусов и отправка `/paylist` админам
- **10 числа в 10:00**: рассылка напоминаний тем, кто **не оплатил** и **не предоплачен**

Команда:
```text
/payschedule day=10 prelist=09:00 remind=10:00 tz=Asia/Novosibirsk
```

### Типовые ошибки

**1) `JobQueue is None`**
- установи: `pip install "python-telegram-bot[job-queue]==21.6"`
- перезапусти бота

**2) `JobQueue.run_monthly() got an unexpected keyword argument 'time'`**
- Это означает несовпадение версии `python-telegram-bot` и параметров.
- Для **python-telegram-bot 21.x** обычно используется `when=` (а не `time=`), и таймзона берётся из `tzinfo` у `datetime.time`.

---

## Данные и приватность

- `bot_data.pkl` — локальная база (пользователи, шаблоны, статусы оплат, предоплата).
- **Не коммить в GitHub**: файл содержит пользовательские данные.

---

## Рекомендации для GitHub

### `.gitignore`
Добавь исключения:
- `.env`
- `bot_data.pkl`
- `.venv/`
- `__pycache__/`

### `.env.example`
Храни пример переменных без секретов.

---

## Автозапуск (systemd, Linux)

Пример `/etc/systemd/system/xui-bot.service`:

```ini
[Unit]
Description=3x-ui Telegram Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/xui-bot
EnvironmentFile=/opt/xui-bot/.env
ExecStart=/opt/xui-bot/.venv/bin/python /opt/xui-bot/bot.py
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
```

Команды:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now xui-bot
sudo systemctl status xui-bot
```

---

## Лицензия
Добавь свою лицензию (MIT/Apache-2.0/…).
