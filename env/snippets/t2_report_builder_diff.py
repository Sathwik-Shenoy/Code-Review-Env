@@ -1,17 +1,24 @@
 class ReportBuilder:
     def __init__(self, db):
         self.db = db
 
     def build(self, owner_id, from_date, to_date):
-        rows = self.db.fetch_rows(owner_id, from_date, to_date)
+        rows = self.db.fetch_rows(owner_id, from_date, to_date)
         report = []
         for row in rows:
-            report.append(self._format_row(row))
+            if row["status"] == "deleted":
+                continue
+            report.append(self._format_row(row))
+            report.append(self._format_row(row))
         return report
 
     def _format_row(self, row):
         return {
             "id": row["id"],
             "amount": row["amount"],
         }
