---
name: update-shipping-address
description: Use this skill when a customer requests to change the shipping address
  for an existing order.
metadata:
  source: skillminer
  status: candidate
  support: 22
---

## When to use
- The customer provides an order ID and a new address.
- The customer asks to update delivery details for a specific purchase.

## Procedure
1. **Verify Order Status**: Call `lookup_order(order_id)` to retrieve order details.
   - If the order is not found, inform the customer that the order could not be located and ask them to verify the ID.
   - If the order status is **"shipped"**, do not attempt to update the address. Inform the customer that the order is in transit and the address cannot be changed. If the customer requests tracking information, call `track_shipment(order_id)` to provide the current status and ETA.
   - If the order status is **"processing"** (or not shipped), proceed to step 2.

2. **Update Address**: Call `update_address(order_id, new_address)` with the customer's new address.
   - If the tool returns an error, inform the customer that the update failed and suggest contacting support via other channels if available, or state that the system could not process the change.

3. **Confirm via Email**: Call `send_email(to, subject, body)` to send a confirmation to the customer's email address found in the order details.
   - Subject: "Shipping Address Updated for Order <ORDER_ID>"
   - Body: Confirm the new address and order ID.

4. **Final Response**: Inform the customer that the address has been successfully updated and a confirmation email has been sent.

## Rules
- **MUST** call `lookup_order` before attempting `update_address` to verify the order exists and is not yet shipped. Variant 5 skipped this check, which is risky if the order ID is invalid or already shipped.
- **NEVER** call `update_address` if `lookup_order` returns a status of "shipped". Variant 1 correctly identified this and refused the update.
- **MUST** send a confirmation email via `send_email` after a successful address update. Variants 2, 4, and 5 all performed this step.
- **NEVER** invent tracking numbers or delivery dates. Only use data returned by `track_shipment` or `lookup_order`.
- **MUST** use the customer's email address from the `lookup_order` result for the confirmation email.

## Replying to the customer
- **If successful**: "Your shipping address for order **<ORDER_ID>** has been updated to **<NEW_ADDRESS>**. A confirmation email has been sent to you. Let me know if there’s anything else I can help with!"
- **If shipped**: "I’m sorry for the inconvenience. I’ve checked order **<ORDER_ID>**, and it has already been shipped. Once an order is in transit, we can’t update the delivery address directly in our system."
- **If tracking requested after rejection**: Provide the `shipping_status` and `eta` from `track_shipment` if available.
