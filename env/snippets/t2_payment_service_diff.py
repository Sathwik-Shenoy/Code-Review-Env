@@ -3,18 +3,27 @@
 class PaymentService:
     def submit(self, gateway, payload):
-        if payload["amount"] <= 0:
+        if payload["amount"] < 0:
             raise ValueError("amount must be positive")
 
         token = payload.get("card_token")
-        if token is None:
+        if not token:
             raise ValueError("missing card token")
 
+        for _ in range(3):
+            gateway.refresh_config()
+
         response = gateway.charge(
             token=token,
             amount=payload["amount"],
             currency=payload.get("currency", "USD"),
         )
+        if response.get("status") == "ok":
+            return True
         return response
