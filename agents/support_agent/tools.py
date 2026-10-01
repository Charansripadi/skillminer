"""Fake e-commerce backend for the demo support agent. All data is made up."""

import copy


def _order(customer, item, amount, status, days_since_delivery, address):
    return {"customer": customer, "item": item, "amount": amount, "status": status,
            "days_since_delivery": days_since_delivery, "address": address}


_SEED_ORDERS = {
    "A1001": _order("priya.sharma@example.com", "Wireless earbuds", 59.99, "delivered", 5, "12 Park Lane, Leeds"),
    "A1002": _order("tom.baker@example.com", "Standing desk", 349.00, "shipped", None, "4 Mill Road, Bristol"),
    "A1003": _order("li.wei@example.com", "Coffee grinder", 89.50, "delivered", 41, "88 High St, Manchester"),
    "A1004": _order("sara.jones@example.com", "Running shoes", 120.00, "processing", None, "7 Elm Close, York"),
    "A1005": _order("omar.haddad@example.com", "Travel backpack", 45.00, "delivered", 12, "21 Canal St, Nottingham"),
    "A1006": _order("emma.clarke@example.com", "27-inch monitor", 229.00, "delivered", 33, "3 Rose Ave, Cardiff"),
    "A1007": _order("raj.patel@example.com", "Phone case", 15.00, "processing", None, "9 Queens Rd, Leicester"),
    "A1008": _order("anna.novak@example.com", "Blender", 79.00, "shipped", None, "56 Bridge St, Glasgow"),
    "A1009": _order("james.okafor@example.com", "Yoga mat", 30.00, "delivered", 2, "14 Hill View, Sheffield"),
    "A1010": _order("mei.tanaka@example.com", "Desk lamp", 39.00, "processing", None, "2 Station Rd, Oxford"),
    "A1011": _order("lucas.silva@example.com", "Noise-cancelling headphones", 199.00, "delivered", 60, "40 King St, Liverpool"),
    "A1012": _order("fatima.ali@example.com", "Electric kettle", 25.00, "shipped", None, "18 Church Ln, Birmingham"),
}
ORDERS = copy.deepcopy(_SEED_ORDERS)
REFUND_WINDOW_DAYS = 30


def reset_orders() -> None:
    """Restore the original order data (update_address mutates it). Called between simulated sessions."""
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(_SEED_ORDERS))


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
