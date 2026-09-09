"""
bot_notifier.py — ImageInsight Telegram notification module.

Drop this file into your project root (alongside main.py).
Set BOT_TOKEN and CHAT_ID below (or via environment variables).
"""

import os
import time
import logging
import threading
import requests

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "8877108990:AAE_KUOqTJsnmqpG8V6Gd5OsW3q854m1smc")
CHAT_ID   = os.environ.get("TG_CHAT_ID",   "5869143564")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ── Low-level send ─────────────────────────────────────────────────────────────
def _send(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message; returns True on success."""
    try:
        r = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        if not r.ok:
            logger.warning(f"Telegram send failed [{r.status_code}]: {r.text[:200]}")
        return r.ok
    except Exception as e:
        logger.warning(f"Telegram unreachable: {e}")
        return False


def notify(message: str):
    """Public helper — fire-and-forget in a thread so the pipeline never blocks."""
    threading.Thread(target=_send, args=(message,), daemon=True).start()


# ── Pipeline lifecycle helpers ─────────────────────────────────────────────────
def notify_pipeline_start(pod_id: str, iterationid: int, stagingid: int,
                           unprocessed_stores: int):
    _send(
        f"🚀 <b>Pipeline Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🖥  Pod ID       : <code>{pod_id}</code>\n"
        f"🔄  Iteration ID : <code>{iterationid}</code>\n"
        f"📦  Staging ID   : <code>{stagingid}</code>\n"
        f"🏪  Stores queued: <b>{unprocessed_stores}</b>\n"
        f"⏰  Started      : {_ts()}"
    )


def notify_batch_start(batch_number: int, assigned_stores: int,
                       assigned_files: int, pod_id: str):
    _send(
        f"📂 <b>Batch {batch_number} Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏪  Stores  : <b>{assigned_stores}</b>\n"
        f"🖼  Images  : <b>{assigned_files}</b>\n"
        f"🖥  Pod     : <code>{pod_id}</code>\n"
        f"⏰  Time    : {_ts()}"
    )


def notify_batch_complete(batch_number: int, stores: int, images: int,
                           visicooler_records: int, yolo_records: int,
                           remaining_stores: int):
    _send(
        f"✅ <b>Batch {batch_number} Complete</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏪  Stores processed   : <b>{stores}</b>\n"
        f"🖼  Images processed   : <b>{images}</b>\n"
        f"🧊  Visicooler records : <b>{visicooler_records}</b>\n"
        f"🤖  YOLO records       : <b>{yolo_records}</b>\n"
        f"⏳  Remaining stores   : <b>{remaining_stores}</b>\n"
        f"⏰  Time               : {_ts()}"
    )


def notify_batch_failed(batch_number: int, error: str, pod_id: str):
    _send(
        f"🚨 <b>Batch {batch_number} FAILED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🖥  Pod   : <code>{pod_id}</code>\n"
        f"❌  Error : <code>{_truncate(error, 300)}</code>\n"
        f"⏰  Time  : {_ts()}\n"
        f"♻️  Retrying in 10 seconds…"
    )


def notify_download_complete(total: int, failed: int):
    status = "✅" if failed == 0 else "⚠️"
    _send(
        f"{status} <b>Image Download Done</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📥  Downloaded : <b>{total - failed}</b> / {total}\n"
        f"❌  Failed     : <b>{failed}</b>\n"
        f"⏰  Time       : {_ts()}"
    )


def notify_yolo_complete(record_count: int):
    _send(
        f"🤖 <b>YOLO Analysis Complete</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊  Records  : <b>{record_count}</b>\n"
        f"⏰  Time     : {_ts()}"
    )


def notify_db_upload_complete(stagingid: int, record_count: int):
    _send(
        f"🗄 <b>DB Upload Complete</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📦  Staging ID : <code>{stagingid}</code>\n"
        f"📊  Rows       : <b>{record_count}</b>\n"
        f"⏰  Time       : {_ts()}"
    )


def notify_pipeline_complete(pod_id: str, iterationid: int, stagingid: int,
                              total_batches: int, cap_status: str):
    emoji = "✅" if cap_status == "success" else "⚠️"
    _send(
        f"{emoji} <b>Pipeline Complete</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🖥  Pod ID       : <code>{pod_id}</code>\n"
        f"🔄  Iteration ID : <code>{iterationid}</code>\n"
        f"📦  Staging ID   : <code>{stagingid}</code>\n"
        f"📊  Total batches: <b>{total_batches}</b>\n"
        f"🏭  CAP pipeline : <b>{cap_status.upper()}</b>\n"
        f"⏰  Finished     : {_ts()}"
    )


def notify_pipeline_error(stage: str, error: str, pod_id: str):
    _send(
        f"🔴 <b>Pipeline ERROR</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🖥  Pod   : <code>{pod_id}</code>\n"
        f"📍 Stage  : <b>{stage}</b>\n"
        f"❌  Error :\n<code>{_truncate(error, 400)}</code>\n"
        f"⏰  Time  : {_ts()}"
    )


def notify_stale_reset(reset_count: int):
    if reset_count > 0:
        _send(
            f"♻️ <b>Stale Batch Reset</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔁  Files reset : <b>{reset_count}</b>\n"
            f"⏰  Time        : {_ts()}"
        )


def notify_cap_pipeline(status: str, duration_ms: float = 0):
    emoji = "✅" if status == "success" else "⚠️"
    _send(
        f"{emoji} <b>CAP Post-Processing</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊  Status   : <b>{status.upper()}</b>\n"
        f"⏱  Duration : <b>{duration_ms:.0f} ms</b>\n"
        f"⏰  Time     : {_ts()}"
    )


# ── 15-minute heartbeat ────────────────────────────────────────────────────────
class PipelineHeartbeat:
    """
    Sends a status update every 15 minutes while the pipeline is running.

    Usage:
        hb = PipelineHeartbeat(get_status_fn=lambda: {...})
        hb.start()
        ...pipeline runs...
        hb.stop()

    get_status_fn must return a dict with keys:
        batch_number, stores_done, images_done, remaining_stores, current_stage
    """

    INTERVAL = 15 * 60  # 15 minutes

    def __init__(self, get_status_fn):
        self._get_status = get_status_fn
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Heartbeat thread started (15-min interval)")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Heartbeat thread stopped")

    def _run(self):
        while not self._stop_event.wait(self.INTERVAL):
            try:
                s = self._get_status()
                _send(
                    f"💓 <b>Pipeline Heartbeat</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📍 Stage           : <b>{s.get('current_stage', 'Unknown')}</b>\n"
                    f"📦 Batch           : <b>{s.get('batch_number', 0)}</b>\n"
                    f"🏪 Stores done     : <b>{s.get('stores_done', 0)}</b>\n"
                    f"🖼 Images done     : <b>{s.get('images_done', 0)}</b>\n"
                    f"⏳ Remaining stores: <b>{s.get('remaining_stores', '?')}</b>\n"
                    f"⏰ Time            : {_ts()}"
                )
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")


# ── Q&A bot (polling) ─────────────────────────────────────────────────────────
class PipelineQABot:
    """
    Polls Telegram for incoming messages and answers simple questions about
    the running pipeline.  Runs in a background daemon thread.

    Usage:
        bot = PipelineQABot(get_status_fn=lambda: {...})
        bot.start()
        ...pipeline runs...
        bot.stop()

    Same get_status_fn signature as PipelineHeartbeat.
    """

    POLL_INTERVAL = 5  # seconds between getUpdates polls

    def __init__(self, get_status_fn):
        self._get_status = get_status_fn
        self._stop_event = threading.Event()
        self._thread = None
        self._last_update_id = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Q&A bot polling thread started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Q&A bot polling thread stopped")

    def _run(self):
        while not self._stop_event.wait(self.POLL_INTERVAL):
            try:
                self._poll()
            except Exception as e:
                logger.warning(f"Bot poll error: {e}")

    def _poll(self):
        params = {"timeout": 3, "allowed_updates": ["message"]}
        if self._last_update_id is not None:
            params["offset"] = self._last_update_id + 1

        try:
            r = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=8)
            if not r.ok:
                return
            data = r.json()
            for update in data.get("result", []):
                self._last_update_id = update["update_id"]
                msg = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                # Only respond to the configured chat
                if chat_id != str(CHAT_ID):
                    continue

                self._handle_command(text, chat_id)
        except Exception as e:
            logger.warning(f"getUpdates error: {e}")

    def _handle_command(self, text: str, chat_id: str):
        s = self._get_status()
        if "/status" in text or "status" in text:
            reply = (
                f"📊 <b>Pipeline Status</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📍 Stage           : <b>{s.get('current_stage', 'Unknown')}</b>\n"
                f"📦 Batch           : <b>{s.get('batch_number', 0)}</b>\n"
                f"🏪 Stores done     : <b>{s.get('stores_done', 0)}</b>\n"
                f"🖼 Images done     : <b>{s.get('images_done', 0)}</b>\n"
                f"⏳ Remaining stores: <b>{s.get('remaining_stores', '?')}</b>\n"
                f"⏰ As of           : {_ts()}"
            )
        elif "/batches" in text or "batch" in text:
            reply = (
                f"📦 <b>Batch Summary</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ Completed : <b>{s.get('batch_number', 0)}</b>\n"
                f"🏪 Stores    : <b>{s.get('stores_done', 0)}</b>\n"
                f"🖼 Images    : <b>{s.get('images_done', 0)}</b>\n"
                f"⏳ Remaining : <b>{s.get('remaining_stores', '?')}</b>"
            )
        elif "/errors" in text or "error" in text:
            errors = s.get('recent_errors', [])
            if errors:
                lines = "\n".join(f"• {e}" for e in errors[-5:])
                reply = f"🚨 <b>Recent Errors</b>\n━━━━━━━━━━━━━━━━━━━━\n{lines}"
            else:
                reply = "✅ No recent errors recorded."
        elif "/db" in text:
            reply = (
                f"🗄 <b>DB Info</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📦 Staging ID   : <code>{s.get('stagingid', 'N/A')}</code>\n"
                f"🔄 Iteration ID : <code>{s.get('iterationid', 'N/A')}</code>\n"
                f"📊 Rows inserted: <b>{s.get('db_rows_inserted', 0)}</b>"
            )
        elif "/pod" in text:
            reply = (
                f"🖥 <b>Pod Info</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🆔 Pod ID       : <code>{s.get('pod_id', 'N/A')}</code>\n"
                f"🔄 Iteration ID : <code>{s.get('iterationid', 'N/A')}</code>"
            )
        elif "/help" in text or "help" in text or "/start" in text:
            reply = (
                f"🤖 <b>ImageInsight Bot Commands</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"/status  — Current pipeline stage & progress\n"
                f"/batches — How many batches completed\n"
                f"/errors  — Last 5 errors (if any)\n"
                f"/db      — Staging ID, iteration, rows inserted\n"
                f"/pod     — Pod ID and iteration info\n"
                f"/help    — This message"
            )
        else:
            reply = (
                f"❓ Unknown command. Send /help to see available commands."
            )

        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": reply, "parse_mode": "HTML"},
            timeout=10,
        )


# ── Utilities ──────────────────────────────────────────────────────────────────
def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _truncate(s: str, max_len: int) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[:max_len] + "…"