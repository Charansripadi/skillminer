"""Fake e-commerce backend for the demo support agent. All data is made up."""

ORDERS = {
    "A1001": {"customer": "priya.sharma@example.com", "item": "Wireless earbuds", "amount": 59.99,
              "status": "delivered", "days_since_delivery": 5, "address": "12 Park Lane, Leeds"},
    "A1002": {"customer": "tom.baker@example.com", "item": "Standing desk", "amount": 349.00,
              "status": "shipped", "days_since_delivery": None, "address": "4 Mill Road, Bristol"},
    "A1003": {"customer": "li.wei@example.com", "item": "Coffee grinder", "amount": 89.50,
              "status": "delivered", "days_since_delivery": 41, "address": "88 High St, Manchester"},
    "A1004": {"customer": "sara.jones@example.com", "item": "Running shoes", "amount": 120.00,
              "status": "processing", "days_since_delivery": None, "address": "7 Elm Close, York"},
}
REFUND_WINDOW_DAYS = 30


def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID and return its details."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"status": "error", "message": f"No order found with ID {order_id}"}
    return {"status": "ok", "order_id": order_id.upper(), **order}


def check_refund_eligibility(order_id: str) -> dict:
    """Check whether an order can be refunded under the 30-day refund policy."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"status": "error", "message": f"No order found with ID {order_id}"}
    if order["status"] != "delivered":
        return {"status": "ok", "eligible": False, "reason": "Order has not been delivered yet"}
    if order["days_since_delivery"] > REFUND_WINDOW_DAYS:
        return {"status": "ok", "eligible": False, "reason": "Outside the 30-day refund window"}
    return {"status": "ok", "eligible": True, "max_refund": order["amount"]}


def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Issue a refund for an order. Only call after checking eligibility."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"status": "error", "message": f"No order found with ID {order_id}"}
    if amount > order["amount"]:
        return {"status": "error", "message": "Refund amount exceeds order total"}
    return {"status": "ok", "refund_id": f"R-{order_id.upper()}", "amount": amount, "reason": reason}


def track_shipment(order_id: str) -> dict:
    """Get the shipping status and estimated delivery for an order."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"status": "error", "message": f"No order found with ID {order_id}"}
    eta = {"shipped": "2 days", "processing": "5 days"}.get(order["status"], "already delivered")
    return {"status": "ok", "shipping_status": order["status"], "eta": eta}


def update_address(order_id: str, new_address: str) -> dict:
    """Change the delivery address. Only possible before the order has shipped."""
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"status": "error", "message": f"No order found with ID {order_id}"}
    if order["status"] != "processing":
        return {"status": "error", "message": "Address can only be changed before shipping"}
    order["address"] = new_address
    return {"status": "ok", "order_id": order_id.upper(), "new_address": new_address}


def send_email(to: str, subject: str, body: str) -> dict:
    """Send a confirmation email to the customer."""
    return {"status": "ok", "sent_to": to, "subject": subject}
