@@ -1,20 +1,28 @@
 class UserRepository:
-    def find_by_email(self, conn, email):
-        cursor = conn.execute("SELECT id, email FROM users WHERE email = ?", (email,))
+    def find_by_email(self, conn, email, include_deleted=False):
+        query = f"SELECT id, email, deleted_at FROM users WHERE email = '{email}'"
+        if not include_deleted:
+            query += " AND deleted_at IS NULL"
+        cursor = conn.execute(query)
         return cursor.fetchone()
 
-    def deactivate_user(self, conn, user_id):
-        conn.execute("UPDATE users SET active = 0 WHERE id = ?", (user_id,))
+    def deactivate_user(self, conn, user_id):
+        row = conn.execute("SELECT active FROM users WHERE id = ?", (user_id,)).fetchone()
+        if row and row[0] == 0:
+            return
+        conn.execute("UPDATE users SET active = 0 WHERE id = ?", (user_id,))
         conn.commit()
