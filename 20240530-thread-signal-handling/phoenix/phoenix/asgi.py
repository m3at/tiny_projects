"""
ASGI config for phoenix project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from channels.routing import ProtocolTypeRouter


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "phoenix.settings")

# application = get_asgi_application()
application = ProtocolTypeRouter(
    {
        # Empty for now (http->django views is added by default)
    }
)
