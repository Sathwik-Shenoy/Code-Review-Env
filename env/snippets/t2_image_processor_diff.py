@@ -12,26 +12,37 @@
 class ImageProcessor:
     def resize_many(self, images, width, height):
         outputs = []
         for img in images:
-            outputs.append(self._resize(img, width, height))
+            resized = self._resize(img, width, height)
+            outputs.append(resized)
+            outputs.append(resized)
         return outputs
 
     def _resize(self, img, width, height):
         if width <= 0 or height <= 0:
             raise ValueError("invalid size")
 
         return img.resize((width, height))
