import json
import os
from datetime import datetime, timedelta, timezone
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import List, Dict, Any

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")
TICKET_LOCK = Lock()
VIETNAM_TIME = timezone(timedelta(hours=7))
TICKET_PRIORITIES = ("low", "medium", "high")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    try:
        with open(catalog_file, encoding="utf-8") as file:
            products = json.load(file)
    except FileNotFoundError:
        return [{"error": "Product catalog file not found."}]

    category = category.strip().casefold()
    return [
        product for product in products
        if product["category"].casefold() == category and product["price_vnd"] <= max_price
    ]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    customer_name = customer_name.strip()
    issue_description = issue_description.strip()
    priority = priority.strip().casefold()
    if not customer_name or not issue_description:
        raise ValueError("Ticket cần có tên khách hàng và mô tả vấn đề.")
    if priority not in TICKET_PRIORITIES:
        raise ValueError("Mức ưu tiên phải là low, medium hoặc high.")

    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    with TICKET_LOCK:
        try:
            with open(tickets_file, encoding="utf-8") as file:
                tickets = json.load(file)
        except FileNotFoundError:
            tickets = []

        now = datetime.now(VIETNAM_TIME)
        existing_ids = {ticket["ticket_id"] for ticket in tickets}
        sequence = len(tickets) + 1
        ticket_id = f"TK-{now:%Y%m%d}-{sequence:03d}"
        while ticket_id in existing_ids:
            sequence += 1
            ticket_id = f"TK-{now:%Y%m%d}-{sequence:03d}"

        ticket = {
            "ticket_id": ticket_id,
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority,
            "status": "open",
            "created_at": now.isoformat(),
            "category": "general",
        }
        tickets.append(ticket)

        # Chỉ thay file gốc sau khi đã ghi xong toàn bộ dữ liệu.
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=RAW_DATA_DIR, delete=False) as file:
                temporary_path = file.name
                json.dump(tickets, file, indent=2, ensure_ascii=False)
            os.replace(temporary_path, tickets_file)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority,
        "status": ticket["status"],
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm và dịch vụ Vingroup theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Danh mục xe điện VinFast hoặc du lịch Vinpearl.",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa, tính bằng VNĐ.",
                    "minimum": 0,
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Ghi nhận yêu cầu hỗ trợ hoặc phản hồi của khách hàng Vingroup.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "description": "Tên khách hàng.", "minLength": 1},
                "issue_description": {"type": "string", "description": "Vấn đề cần hỗ trợ.", "minLength": 1},
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên của yêu cầu.",
                    "enum": list(TICKET_PRIORITIES),
                    "default": "medium",
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
