"""Qt worker object to manage subprocess execution."""

from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal


class ProcessWorker(QObject):
    started = pyqtSignal(list)
    output = pyqtSignal(str)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, repo_root, env: Optional[Dict[str, str]] = None):
        super().__init__()
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(repo_root))
        penv = QProcessEnvironment.systemEnvironment()
        if env:
            for k, v in env.items():
                penv.insert(k, v)
        self.proc.setProcessEnvironment(penv)

        self.proc.readyReadStandardOutput.connect(self._emit_stdout)
        self.proc.readyReadStandardError.connect(self._emit_stderr)
        self.proc.finished.connect(lambda code, _status: self.finished.emit(int(code)))
        self.proc.errorOccurred.connect(lambda _e: self.error.emit("Process error"))

    def run(self, args: List[str]):
        if not args:
            self.error.emit("Empty command")
            return
        self.started.emit(args)
        self.proc.start(args[0], args[1:])

    def kill(self):
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()

    def _emit_stdout(self):
        data = bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")
        if data:
            self.output.emit(data)

    def _emit_stderr(self):
        data = bytes(self.proc.readAllStandardError()).decode(errors="ignore")
        if data:
            self.output.emit(data)
