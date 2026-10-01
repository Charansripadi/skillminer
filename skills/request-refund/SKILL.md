---
name: request-refund
description: Use this skill when a customer requests a refund for a specific order.
metadata:
  source: skillminer
  status: candidate
  support: 19
---

## When to use
- The customer explicitly asks for a refund for a purchased item.
- The customer provides an order ID or enough context to identify the order.

## Procedure
1. **Identify Order**: If the customer has not provided the order ID, ask for it. If provided, call `lookup_order(order_id)` to retrieve order details (status, amount, customer email, item).
   - *Note*: If the order is not found, inform the customer and ask to verify the ID.
2. **Check Eligibility**: Call `check_refund_eligibility(order_id)`.
   - **If `eligible` is `true`**:
     1. Call `issue_refund(order_id, amount, reason)` using the `max_refund` amount from the eligibility check and the customer's stated reason.
     2. Call `send_email(to, subject, body)` to notify the customer of the successful refund.
     3. Inform the customer that the refund has been processed.
   - **If `eligible` is `false`**:
     1. Do **not** call `issue_refund`.
     2. If the reason is "Order has not been delivered yet," call `track_shipment(order_id)` to get the current status and ETA.
     3. Inform the customer that the refund cannot be processed yet because the order is not delivered. Provide the shipping status/ETA if available.
     4. If the reason is "Outside the 30-day refund window," inform the customer that the refund window has passed. Do not issue a refund unless the tool explicitly allows an exception (evidence shows standard denial).
3. **Finalize**: Ensure the customer knows the next steps or the final status.

## Rules
- **MUST** call `check_refund_eligibility` before `issue_refund`.
- **NEVER** call `issue_refund` if `check_refund_eligibility` returns `eligible: false`. (Variant 6 showed an agent issuing a refund despite ineligibility; this is incorrect behavior to avoid).
- **MUST** use the `amount` from the eligibility check or order lookup for the refund; do not guess the amount.
- **MUST** send a confirmation email via `send_email` after a successful refund.
- **NEVER** invent refund timelines (e.g., "5-7 days") or policies not present in tool results. If the tool does not specify a timeline, do not provide one.
- **MUST** use `track_shipment` only when the order is not yet delivered to provide context to the customer.

## Replying to the customer
- **Success**: Confirm the refund amount and that it has been processed. Do not specify a credit timeline unless provided by the tool.
- **Ineligible (Undelivered)**: Explain that refunds are only processed after delivery. Provide the current shipping status and ETA.
- **Ineligible (Time Window)**: Explain that the 30-day window has passed. Apologize for the inconvenience.
- **Tone**: Empathetic and professional. Acknowledge the customer's frustration if they are upset about the item or delay.
