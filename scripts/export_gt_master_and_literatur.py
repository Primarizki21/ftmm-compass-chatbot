#!/usr/bin/env python3
"""
Export GT_MASTER, Katalog_Mata_Kuliah_369, and Literatur to Google Spreadsheet.
Spreadsheet ID: 1jR0Hj3NBBvUv3NDfbU_-pPDIWB7EaDfJk5E7-7SoKSM

Operations:
1. Create/Update tab 'GT_MASTER' (450 rows: 1 header + 449 Q&A).
2. Rename tab 'Sheet12' (gid: 1415192961) -> 'Katalog_Mata_Kuliah_369' and format header.
3. Rename tab 'Sheet11' (gid: 1603934730) -> 'Literatur' and upload 34 structured literature entries.
"""

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_master_and_lit")

SPREADSHEET_ID = "1jR0Hj3NBBvUv3NDfbU_-pPDIWB7EaDfJk5E7-7SoKSM"
GCP_KEYS_PATH = Path("/home/primarizki/.config/google-drive-mcp/gcp-oauth.keys.json")
TOKENS_PATH = Path("/home/primarizki/.config/google-drive-mcp/tokens.json")

GT_MASTER_PATH = Path("data/ground_truth/gt_master.json")
LITERATUR_PATH = Path("data/ground_truth/literatur.json")


def get_access_token() -> str:
    """Retrieve or refresh Google OAuth access token."""
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

    # First check if existing accessToken works
    current_token = account.get("accessToken")
    if current_token:
        test_url = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}?fields=properties.title"
        req = urllib.request.Request(test_url)
        req.add_header("Authorization", f"Bearer {current_token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("✓ Stored access token is still valid.")
                    return current_token
        except Exception:
            logger.info("Stored access token expired, attempting refresh via OAuth endpoint...")

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
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to refresh OAuth token: HTTP {resp.status}")
            res = json.loads(resp.read().decode())
            access_token = res["access_token"]
            logger.info("✓ Fresh OAuth Access Token obtained successfully.")
            return access_token
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode()
        logger.error("OAuth token refresh failed (HTTP %d): %s", e.code, err_msg)
        raise RuntimeError(
            "Google OAuth Refresh Token has expired or was revoked.\n"
            "Please run: npx -y @piotr-agier/google-drive-mcp auth\n"
            "to authenticate in browser and update ~/.config/google-drive-mcp/tokens.json."
        ) from e


def execute_sheets_api(method: str, endpoint: str, access_token: str, body: dict | None = None) -> dict:
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}{endpoint}"
    req_data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=req_data, method=method)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read().decode()
        return json.loads(content) if content else {}


def get_all_sheets_meta(access_token: str) -> dict[str, dict]:
    meta = execute_sheets_api("GET", "?fields=sheets.properties", access_token)
    sheets_map = {}
    for s in meta.get("sheets", []):
        prop = s.get("properties", {})
        sheets_map[prop.get("title")] = prop
    return sheets_map


def rename_sheet_if_exists(access_token: str, old_title: str, new_title: str) -> int | None:
    meta_map = get_all_sheets_meta(access_token)
    if new_title in meta_map:
        logger.info("Sheet '%s' already exists (sheetId=%s).", new_title, meta_map[new_title]["sheetId"])
        return meta_map[new_title]["sheetId"]

    if old_title in meta_map:
        sheet_id = meta_map[old_title]["sheetId"]
        logger.info("Renaming '%s' (sheetId=%s) -> '%s'...", old_title, sheet_id, new_title)
        rename_req = {
            "requests": [
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": sheet_id,
                            "title": new_title,
                        },
                        "fields": "title",
                    }
                }
            ]
        }
        execute_sheets_api("POST", ":batchUpdate", access_token, body=rename_req)
        logger.info("✓ Successfully renamed '%s' -> '%s'", old_title, new_title)
        return sheet_id

    logger.warning("Neither '%s' nor '%s' found in spreadsheet.", old_title, new_title)
    return None


def export_gt_master(access_token: str):
    logger.info("--- Exporting GT_MASTER ---")
    with open(GT_MASTER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta_map = get_all_sheets_meta(access_token)
    sheet_title = "GT_MASTER"

    if sheet_title not in meta_map:
        logger.info("Creating sheet '%s'...", sheet_title)
        add_req = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheet_title,
                            "gridProperties": {
                                "rowCount": 550,
                                "columnCount": 10,
                                "frozenRowCount": 1,
                            },
                        }
                    }
                }
            ]
        }
        res = execute_sheets_api("POST", ":batchUpdate", access_token, body=add_req)
        sheet_id = res["replies"][0]["addSheet"]["properties"]["sheetId"]
    else:
        sheet_id = meta_map[sheet_title]["sheetId"]
        logger.info("Clearing existing sheet '%s'...", sheet_title)
        execute_sheets_api("POST", f"/values/{sheet_title}:clear", access_token, body={})

    header = ["id", "question", "answer", "source_id", "source_title", "source_section", "source_type"]
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

    logger.info("Uploading %d rows to '%s'...", len(rows), sheet_title)
    body = {
        "range": f"{sheet_title}!A1",
        "majorDimension": "ROWS",
        "values": rows,
    }
    endpoint = f"/values/{sheet_title}!A1:append?valueInputOption=RAW&insertDataOption=OVERWRITE"
    execute_sheets_api("POST", endpoint, access_token, body=body)

    # Format header
    format_req = {
        "requests": [
            {
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.08, "green": 0.28, "blue": 0.45},
                            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True},
                            "horizontalAlignment": "CENTER",
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                }
            },
            {
                "autoResizeDimensions": {
                    "dimensions": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 7}
                }
            },
        ]
    }
    execute_sheets_api("POST", ":batchUpdate", access_token, body=format_req)
    logger.info("✓ GT_MASTER export completed (%d rows).", len(rows))


def export_literatur(access_token: str):
    logger.info("--- Exporting Literatur ---")
    with open(LITERATUR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Rename Sheet11 -> Literatur if Sheet11 exists
    sheet_id = rename_sheet_if_exists(access_token, "Sheet11", "Literatur")

    sheet_title = "Literatur"
    if sheet_id is None:
        meta_map = get_all_sheets_meta(access_token)
        if sheet_title not in meta_map:
            logger.info("Creating sheet '%s'...", sheet_title)
            add_req = {
                "requests": [
                    {
                        "addSheet": {
                            "properties": {
                                "title": sheet_title,
                                "gridProperties": {
                                    "rowCount": 100,
                                    "columnCount": 6,
                                    "frozenRowCount": 1,
                                },
                            }
                        }
                    }
                ]
            }
            res = execute_sheets_api("POST", ":batchUpdate", access_token, body=add_req)
            sheet_id = res["replies"][0]["addSheet"]["properties"]["sheetId"]
        else:
            sheet_id = meta_map[sheet_title]["sheetId"]

    logger.info("Clearing sheet '%s'...", sheet_title)
    execute_sheets_api("POST", f"/values/{sheet_title}:clear", access_token, body={})

    header = ["judul", "penulis", "tahun", "abstrak", "artinya untuk project ini"]
    rows = [header]
    for item in data:
        row = [
            str(item["judul"]),
            str(item["penulis"]),
            str(item["tahun"]),
            str(item["abstrak"]),
            str(item["artinya untuk project ini"]),
        ]
        rows.append(row)

    logger.info("Uploading %d rows to '%s'...", len(rows), sheet_title)
    body = {
        "range": f"{sheet_title}!A1",
        "majorDimension": "ROWS",
        "values": rows,
    }
    endpoint = f"/values/{sheet_title}!A1:append?valueInputOption=RAW&insertDataOption=OVERWRITE"
    execute_sheets_api("POST", endpoint, access_token, body=body)

    # Format header
    format_req = {
        "requests": [
            {
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.12, "green": 0.38, "blue": 0.28},
                            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True},
                            "horizontalAlignment": "CENTER",
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                }
            },
            {
                "autoResizeDimensions": {
                    "dimensions": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 5}
                }
            },
        ]
    }
    execute_sheets_api("POST", ":batchUpdate", access_token, body=format_req)
    logger.info("✓ Literatur export completed (%d rows).", len(rows))


def rename_sheet12_to_katalog(access_token: str):
    logger.info("--- Renaming Sheet12 -> Katalog_Mata_Kuliah_369 ---")
    sheet_id = rename_sheet_if_exists(access_token, "Sheet12", "Katalog_Mata_Kuliah_369")
    if sheet_id:
        # Apply freeze row 1
        freeze_req = {
            "requests": [
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": sheet_id,
                            "gridProperties": {
                                "frozenRowCount": 1,
                            },
                        },
                        "fields": "gridProperties.frozenRowCount",
                    }
                }
            ]
        }
        execute_sheets_api("POST", ":batchUpdate", access_token, body=freeze_req)
        logger.info("✓ Frozen header row 1 on Katalog_Mata_Kuliah_369.")


def main():
    logger.info("Starting sync of GT_MASTER, Katalog_Mata_Kuliah_369, and Literatur to Google Spreadsheet...")
    access_token = get_access_token()
    rename_sheet12_to_katalog(access_token)
    export_gt_master(access_token)
    export_literatur(access_token)
    logger.info("=" * 60)
    logger.info("✓ ALL TASKS COMPLETED SUCCESSFULLY ON SPREADSHEET %s", SPREADSHEET_ID)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
