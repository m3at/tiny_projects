import contextlib
import logging
import os
import signal
import threading
import time
from threading import Event
from time import sleep

import redis
import uvicorn
from django.core.management.base import BaseCommand

# from prometheus_client import make_asgi_app
from prometheus_client import Counter, start_wsgi_server

from tasks.models import Birds

logger = logging.getLogger("phoenix." + __file__)

C_COMMAND = Counter("jjzz_counter_command", "Counter in command")


class ShouldRestartException(Exception):
    pass


def receive_usr2(signum, _):
    logger.warning(f"[SIGUSR2: {signum=}]")
    raise ShouldRestartException()


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


class Command(BaseCommand):
    help = "I heard phoenix will keep respawning, let's see"

    def handle(self, *args, **options):
        try:
            latest_bird = Birds.objects.latest("date_added")
        except Birds.DoesNotExist:
            self.stdout.write("Birds aren't real. At least I've not seen one yet!")
            latest_bird = None

        pid = os.getpid()

        # port = 8001
        # app_prometheus = make_asgi_app()
        # config = uvicorn.Config(
        #     app_prometheus, host="127.0.0.1", port=port, log_level="info"
        # )
        # prom_server = Server(config=config)
        start_wsgi_server(8001)

        # with prom_server.run_in_thread():
        # Higher loop
        while True:
            try:
                C_COMMAND.inc()

                listener_thread = threading.Thread(
                    target=self.run_listener,
                    kwargs=dict(pid=pid),
                    # Set as daemon to die on parent process failure
                    daemon=True,
                )
                listener_thread.start()

                # Run the main loop
                signal.signal(signal.SIGUSR2, receive_usr2)

                # If we have at least one bird
                if latest_bird is not None:
                    self.main_loop()
                else:
                    # Block until interrupted
                    Event().wait()

            except ShouldRestartException:
                latest_bird = Birds.objects.latest("date_added")
                self.stdout.write(f"The new creature we spotted: {latest_bird}")

                # Should be instant
                listener_thread.join(timeout=4)

            except KeyboardInterrupt:
                listener_thread.join(timeout=0.2)
                return

    def main_loop(self):
        sleep(0.2)

        # That unicode sequence? It's a phoenix I swear
        self.stdout.write("A new phoenix rises, how majestic")
        self.stdout.write("🔥", ending="")
        self.stdout.flush()
        while True:
            self.stdout.write("\b\b🔥🐦‍🔥", ending="")
            self.stdout.flush()
            sleep(1)

    def run_listener(self, pid: int):
        redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
        pubsub = redis_client.pubsub()
        pubsub.subscribe("command_channel")

        self.stdout.write(self.style.SUCCESS("Watching the skies 🔭"))

        for message in pubsub.listen():
            if message["type"] == "message":
                # m = message["data"].decode("utf-8")
                pubsub.close()
                self.stdout.write(
                    "\nA new bird! Let's celebrate by restarting a cycle: 🪣💦  🐦‍🔥→💀"
                )
                os.kill(pid, signal.SIGUSR2)
                return
