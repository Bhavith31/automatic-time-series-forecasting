from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # /admin/ → Django's built-in admin panel

    path('', include('forecaster.urls')),
    # '' means root URL → hand over to forecaster app's urls.py
    # All our app pages are defined there
]

# Serve uploaded files during development
# In production, a web server like Nginx would handle this
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
