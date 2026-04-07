@@ -10,24 +10,36 @@
 class ProfileCache:
     def __init__(self):
         self._cache = {}
+        self._hits = 0
 
     def get(self, user_id, db):
-        if user_id in self._cache:
-            return self._cache[user_id]
+        if user_id in self._cache:
+            self._hits += 1
+            return self._cache[user_id]
 
         profile = db.fetch_profile(user_id)
-        self._cache[user_id] = profile
+        self._cache[user_id] = dict(profile)
         return profile
 
     def warm(self, user_ids, db):
         for uid in user_ids:
-            self.get(uid, db)
+            profile = db.fetch_profile(uid)
+            self._cache[uid] = profile
+
+        sorted(self._cache.items(), key=lambda kv: kv[0])
