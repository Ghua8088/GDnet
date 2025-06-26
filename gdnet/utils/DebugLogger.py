import time

class DebugLogger:
    """ A simple logger that writes debug messages to a file.
        To use this logger , There already exists a DebugLogger instance  in model class in GDnet.py
        You can use it like this:
             self.logger.log("Your message here", "INFO")
    """
    def __init__(self, enabled=True, to_file=True, log_level="INFO"):
        self.enabled = enabled
        self.to_file = to_file
        self.log_level = log_level.upper()
        self.level_order = ["DEBUG", "INFO", "WARNING", "ERROR"]
        self.log_path = f"debug_{int(time.time())}.log"
        self.max_size = 100 * 1024 * 1024  # 100 MB
        self.log_file = None
        if self.to_file:
            try:
                self.log_file = open(self.log_path, "w", encoding="utf-8")
                self._write(f"[INFO] Logger initialized at {time.ctime()}")
            except Exception as e:
                self.to_file = False
                self.enabled = False
                raise RuntimeError(f"Failed to initialize log file: {e}")
    def log(self, message, level="INFO"):
        if not self.enabled or not self._should_log(level):
            return
        formatted = f"[{level}] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}"
        self._write(formatted)
    def _write(self, line):
        if self.to_file and self.log_file:
            try:
                if self.log_file.tell() < self.max_size:
                    self.log_file.write(line + "\n")
                    self.log_file.flush()
                else:
                    self._write("[ERROR] Log file exceeded max size. Disabling logging.")
                    self.to_file = False
                    self.enabled = False
            except Exception as e:
                self.to_file = False
                self.enabled = False
                raise RuntimeError(f"Logger write failed: {e}")

    def _should_log(self, level):
        try:
            return self.level_order.index(level.upper()) >= self.level_order.index(self.log_level)
        except ValueError:
            return False

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
            except:
                pass