def reconcile(records, ledger):
    out = []
    for record in records:
        bill_id = record.get("bill_id")
        if bill_id not in ledger:
            continue

        amount = ledger[bill_id] - record.get("paid", 0)
        if amount >= 0:
            out.append({"bill_id": bill_id, "due": amount})
    return out
