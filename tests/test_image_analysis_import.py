import importlib
import builtins
import sys


def test_import_image_analysis_without_optional_dependencies(monkeypatch):
    missing = {"cv2", "numpy", "torch", "PIL", "facenet_pytorch"}
    for name in list(sys.modules):
        if name.split('.')[0] in missing:
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split('.')[0] in missing:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "facefind.embedding_utils", raising=False)
    monkeypatch.delitem(sys.modules, "facefind.image_analysis", raising=False)

    module = importlib.import_module("facefind.image_analysis")
    assert module is not None
