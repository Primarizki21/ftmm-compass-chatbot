#!/usr/bin/env python3
"""
Export GT_BARU course dataset to Google Spreadsheet via Google Sheets API v4.
Spreadsheet ID: 1jR0Hj3NBBvUv3NDfbU_-pPDIWB7EaDfJk5E7-7SoKSM
Sheet Name: GT_BARU (370 rows: 1 header + 369 courses)
"""

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_sheets")

SPREADSHEET_ID = "1jR0Hj3NBBvUv3NDfbU_-pPDIWB7EaDfJk5E7-7SoKSM"
DATASET_PATH = Path("data/ground_truth/gt_baru_courses.json")
GCP_KEYS_PATH = Path("/home/primarizki/.config/google-drive-mcp/gcp-oauth.keys.json")
TOKENS_PATH = Path("/home/primarizki/.config/google-drive-mcp/tokens.json")
SHEET_TITLE = "GT_BARU"


def get_fresh_access_token() -> str:
    if not GCP_KEYS_PATH.exists() or not TOKENS_PATH.exists():
        raise FileNotFoundError("GCP OAuth keys or tokens file not found in ~/.config/google-drive-mcp/")

    with open(GCP_KEYS_PATH, "r", encoding="utf-8") as f:
        keys_data = json.load(f)
    with open(TOKENS_PATH, "r", encoding="utf-8") as f:
        token_data = json.load(f)

    installed = keys_data.get("installed") or keys_data.get("web")
    client_id = installed["client_id"]
    client_secret = installed["client_secret"]

    accounts = token_data.get("accounts", {})
    account = accounts.get("prima") or accounts.get("default")
    if not account:
        raise KeyError("Account 'prima' or 'default' not found in tokens.json")

    refresh_token = account["refreshToken"]

    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    req_data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(token_url, data=req_data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to refresh OAuth token: HTTP {resp.status}")
        res = json.loads(resp.read().decode())
        access_token = res["access_token"]
        logger.info("✓ Fresh OAuth Access Token obtained successfully.")
        return access_token


def execute_sheets_api(method: str, endpoint: str, access_token: str, body: dict | None = None) -> dict:
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}{endpoint}"
    req_data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=req_data, method=method)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read().decode()
        return json.loads(content) if content else {}


def ensure_sheet_exists(access_token: str, sheet_title: str = SHEET_TITLE) -> int:
    """Check if sheet exists; if so clear it, if not create it."""
    meta = execute_sheets_api("GET", "?fields=sheets.properties", access_token)
    sheets = meta.get("sheets", [])

    for s in sheets:
        prop = s.get("properties", {})
        if prop.get("title") == sheet_title:
            sheet_id = prop.get("sheetId")
            logger.info("Sheet '%s' already exists (sheetId=%s). Clearing existing rows...", sheet_title, sheet_id)
            execute_sheets_api("POST", f"/values/{sheet_title}:clear", access_token, body={})
            return sheet_id

    # Create new sheet
    logger.info("Sheet '%s' does not exist. Creating new sheet via batchUpdate...", sheet_title)
    add_req = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": sheet_title,
                        "gridProperties": {
                            "rowCount": 500,
                            "columnCount": 10,
                            "frozenRowCount": 1,
                        },
                    }
                }
            }
        ]
    }
    res = execute_sheets_api("POST", ":batchUpdate", access_token, body=add_req)
    new_sheet_id = res["replies"][0]["addSheet"]["properties"]["sheetId"]
    logger.info("✓ Created sheet '%s' with sheetId=%s", sheet_title, new_sheet_id)
    return new_sheet_id


def upload_dataset_to_sheet(access_token: str, sheet_title: str = SHEET_TITLE):
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d courses from %s", len(data), DATASET_PATH)

    header = [
        "id",
        "question",
        "answer",
        "source_id",
        "source_title",
        "source_section",
        "source_type",
    ]

    rows = [header]
    for item in data:
        row = [
            str(item["id"]),
            str(item["question"]),
            str(item["answer"]),
            str(item["source_id"]),
            str(item["source_title"]),
            str(item["source_section"]),
            str(item["source_type"]),
        ]
        rows.append(row)

    logger.info("Uploading %d total rows (1 header + %d data) to '%s'...", len(rows), len(data), sheet_title)
    body = {
        "range": f"{sheet_title}!A1",
        "majorDimension": "ROWS",
        "values": rows,
    }
    endpoint = f"/values/{sheet_title}!A1:append?valueInputOption=RAW&insertDataOption=OVERWRITE"
    res = execute_sheets_api("POST", endpoint, access_token, body=body)

    updates = res.get("updates", {})
    logger.info(
        "✓ Successfully updated %s: %d rows updated across range %s",
        sheet_title,
        updates.get("updatedRows", len(rows)),
        updates.get("updatedRange", f"{sheet_title}!A1"),
    )


def apply_header_formatting(access_token: str, sheet_id: int):
    logger.info("Applying professional styling (freeze header, bold text, auto-resize)...")
    format_req = {
        "requests": [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.08, "green": 0.28, "blue": 0.45},
                            "textFormat": {
                                "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                                "bold": True,
                                "fontSize": 10,
                            },
                            "horizontalAlignment": "CENTER",
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                }
            },
            {
                "autoResizeDimensions": {
                    "dimensions": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": 0,
                        "endIndex": 7,
                    }
                }
            },
        ]
    }
    execute_sheets_api("POST", ":batchUpdate", access_token, body=format_req)
    logger.info("✓ Header styling and auto-resize applied.")


def main():
    logger.info("Starting export of GT_BARU to Google Spreadsheet...")
    access_token = get_fresh_access_token()
    sheet_id = ensure_sheet_exists(access_token, SHEET_TITLE)
    upload_dataset_to_sheet(access_token, SHEET_TITLE)
    apply_header_formatting(access_token, sheet_id)
    logger.info("=" * 60)
    logger.info("✓ EXPORT COMPLETE: Tab '%s' on Spreadsheet %s is live!", SHEET_TITLE, SPREADSHEET_ID)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
