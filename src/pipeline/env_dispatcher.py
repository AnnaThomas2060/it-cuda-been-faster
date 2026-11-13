import multiprocessing
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from engine.worker import evaluate_request, worker_loop
from src.contracts import KernelEvalRequest, KernelEvalResponse, MockVerifier


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _timeout_response(request: KernelEvalRequest, message: str) -> KernelEvalResponse:
    return KernelEvalResponse(
        task_id=request.task_id,
        compiled=False,
        correct=False,
        status="timeout",
        stable=False,
        error_message=message,
    )


@dataclass
class _WorkerHandle:
    process: Any
    conn: Any
    lock: threading.Lock


class SubprocessDispatcher:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.context = multiprocessing.get_context("spawn")
        self.worker_count = max(1, int(config.verifier_workers))
        self.handles: List[Optional[_WorkerHandle]] = [None] * self.worker_count
        self._cursor = 0
        self._cursor_lock = threading.Lock()

    def _start_handle(self) -> _WorkerHandle:
        parent_conn, child_conn = self.context.Pipe(duplex=True)
        process = self.context.Process(
            target=worker_loop,
            args=(child_conn, self.config.verifier_device, self.config.to_dict()),
            daemon=True,
        )
        process.start()
        child_conn.close()
        return _WorkerHandle(process=process, conn=parent_conn, lock=threading.Lock())

    def _get_handle(self) -> _WorkerHandle:
        with self._cursor_lock:
            index = self._cursor % self.worker_count
            self._cursor += 1
        handle = self.handles[index]
        if handle is None or not handle.process.is_alive():
            if handle is not None:
                self._close_handle(handle)
            handle = self._start_handle()
            self.handles[index] = handle
        return handle

    def _close_handle(self, handle: Optional[_WorkerHandle]) -> None:
        if handle is None:
            return
        try:
            handle.conn.send(None)
        except Exception:
            pass
        try:
            handle.process.join(timeout=2.0)
        except Exception:
            pass
        if handle.process.is_alive():
            handle.process.terminate()
            handle.process.join(timeout=2.0)
        try:
            handle.conn.close()
        except Exception:
            pass

    def _restart_handle(self, index: int) -> None:
        handle = self.handles[index]
        if handle is not None:
            self._close_handle(handle)
        self.handles[index] = self._start_handle()

    def evaluate(self, request: KernelEvalRequest) -> KernelEvalResponse:
        handle = self._get_handle()
        timeout = float(request.timeout_seconds) + float(self.config.verifier_startup_timeout_sec)
        with handle.lock:
            try:
                handle.conn.send(_dump_model(request))
            except Exception as exc:
                self._restart_handle(self.handles.index(handle))
                return _timeout_response(
                    request, f"Failed to send request to verifier worker: {exc}"
                )

            if not handle.conn.poll(timeout):
                index = self.handles.index(handle)
                self._restart_handle(index)
                return _timeout_response(
                    request, f"Verifier worker timed out after {timeout:.0f}s."
                )

            try:
                payload = handle.conn.recv()
            except Exception as exc:
                index = self.handles.index(handle)
                self._restart_handle(index)
                return _timeout_response(request, f"Verifier worker connection failed: {exc}")

        if payload is None:
            return _timeout_response(request, "Verifier worker returned no payload.")
        return KernelEvalResponse(**payload)

    def close(self) -> None:
        for index, handle in enumerate(self.handles):
            self._close_handle(handle)
            self.handles[index] = None

    def __enter__(self) -> "SubprocessDispatcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class InProcessDispatcher:
    def __init__(self, config: Any) -> None:
        self.config = config

    def evaluate(self, request: KernelEvalRequest) -> KernelEvalResponse:
        payload = _dump_model(request)
        try:
            response = evaluate_request(payload, self.config.verifier_device, self.config)
        except Exception as exc:
            return _timeout_response(request, f"In-process evaluation failed: {exc}")
        return KernelEvalResponse(**response)

    def close(self) -> None:
        return None

    def __enter__(self) -> "InProcessDispatcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MockDispatcher:
    def __init__(self, simulated_speedup: float = 1.05) -> None:
        self.verifier = MockVerifier(simulated_speedup=simulated_speedup)

    def evaluate(self, request: KernelEvalRequest) -> KernelEvalResponse:
        return self.verifier.evaluate(request)

    def close(self) -> None:
        return None

    def __enter__(self) -> "MockDispatcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def build_dispatcher(config: Any, kind: str = "auto") -> Any:
    if kind == "mock":
        return MockDispatcher()
    if kind == "in_process":
        return InProcessDispatcher(config)
    if kind == "subprocess" or (kind == "auto" and config.use_verifier_worker):
        return SubprocessDispatcher(config)
    return InProcessDispatcher(config)
