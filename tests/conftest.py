import time
import pytest
from _pytest.config import ExitCode
from _pytest.terminal import TerminalReporter
from _pytest import timing


class SingleLineReporter(TerminalReporter):
    def __init__(self, config):
        super().__init__(config)
        self._session_start = timing.Instant()
        self._perf_start = time.perf_counter()

    def pytest_sessionstart(self, session):
        self._session = session
        self._session_start = timing.Instant()
        self._perf_start = time.perf_counter()

    def pytest_report_header(self, config, start_path):
        return []

    def pytest_collection_finish(self, session):
        pass

    def pytest_runtest_logstart(self, nodeid, location):
        pass

    def pytest_runtest_logfinish(self, nodeid, location):
        pass

    def pytest_runtest_logreport(self, report):
        self._tests_ran = True
        res = self.config.hook.pytest_report_teststatus(report=report, config=self.config)
        category = res[0]
        self._add_stats(category, [report])

    def pytest_sessionfinish(self, session, exitstatus):
        elapsed = time.perf_counter() - self._perf_start
        passed = len(self.stats.get("passed", []))
        failed = len(self.stats.get("failed", []))
        errors = len(self.stats.get("error", []))
        skipped = len(self.stats.get("skipped", []))

        if exitstatus == ExitCode.OK:
            details = [f"{passed} passed"]
            if skipped > 0:
                details.append(f"{skipped} skipped")
            self._tw.line(f"DONE: {', '.join(details)} in {elapsed:.2f}s", green=True)
        else:
            self.summary_failures()
            self.summary_errors()
            self.summary_stats()


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if config.option.verbose > 0:
        return
    standard = config.pluginmanager.get_plugin("terminalreporter")
    if standard is not None:
        config.pluginmanager.unregister(standard)
        reporter = SingleLineReporter(config)
        config.pluginmanager.register(reporter, "terminalreporter")
