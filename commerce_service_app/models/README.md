# Local Models

This directory is intentionally kept out of the public repository except for this note.

The demo can use a local embedding model such as `bge-base-zh-v1.5`. Download model files locally before running GraphRAG and set:

```text
EMBEDDING_MODEL=commerce_service_app/models/bge-base-zh-v1.5
```

Do not commit model weights such as `pytorch_model.bin`, `*.safetensors`, `*.pt`, or `*.onnx`.
