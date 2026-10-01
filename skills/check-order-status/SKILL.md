---
name: check-order-status
description: Use this skill when a customer asks for the status, location, or estimated
  delivery time of a specific order.
metadata:
  source: skillminer
  status: candidate
  support: 21
---

## When to use
- Customer provides an order ID and asks "Where is my order?", "What is the status?", or "When will it arrive?"
- Customer is frustrated about delivery delays and requests tracking information.

## Procedure
1. **Identify Order ID**: Extract the order ID from the customer's message. If missing, ask for it.
2. **Retrieve Order Details**: Call `lookup_order(order_id)`.
   - **If FAILED** (e.g., "No order found"): Inform the customer the ID is invalid. Ask them to double-check the format (e.g., <ORDER_ID>7) or provide the email used for purchase. Do not guess.
   - **If OK**: Note the `status` (processing, shipped, delivered) and `customer` email.
3. **Get Shipping Status**: Call `track_shipment(order_id)`.
   - **If status is "processing"**: The order has not shipped. Inform the customer it is being prepared. Do not provide a tracking number.
   - **If status is "shipped"**: Note the `eta` (estimated time of arrival).
4. **Handle Follow-ups**:
   - If the customer asks for a tracking number but the status is "processing", explain that tracking is generated upon shipment.
   - If the customer asks for a refund while checking status, call `check_refund_eligibility(order_id)`. Only proceed if `eligible` is true.
   - If the customer requests a confirmation email, call `send_email(to, subject, body)` using the email from `lookup_order`.

## Rules
- **MUST** call `lookup_order` before `track_shipment` to verify the order exists and get customer details. Variant 3 skipped this, but Variant 2 and 6 show that `lookup_order` provides necessary context (email, item) for accurate replies.
- **NEVER** invent a tracking number or specific location if `track_shipment` returns "processing". State clearly that it is not yet available.
- **NEVER** issue a refund without first calling `check_refund_eligibility`. Variant 5 correctly checked eligibility before addressing refund concerns.
- **MUST** apologize and acknowledge frustration if the customer expresses anger, but do not promise specific compensation or escalation paths not supported by tools.
- **NEVER** guess the ETA if `track_shipment` fails or returns no data. State that the information is currently unavailable.

## Replying to the customer
- **Processing**: "Your order **<ORDER_ID>** is currently being processed. A tracking number will be generated once it ships. Estimated delivery is approximately **<ETA>** days from shipment."
- **Shipped**: "Your order **<ORDER_ID>** has been shipped and is in transit. The estimated delivery time is **<ETA>**."
- **Not Found**: "I’m sorry, but I can’t locate an order with ID **<ORDER_ID>**. Please double-check the number or provide the email address used for the purchase."
- **Tone**: Be empathetic. If the customer is frustrated, acknowledge their concern ("I understand how important it is to know...") before providing facts. Do not make promises beyond what the tools confirm.
