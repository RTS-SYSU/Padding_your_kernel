# 参与开发

使用 Python 3.10 或更新版本，安装开发副本后运行标准库测试。

```text
python -m pip install -e .
python -B -m unittest discover -s tests -v
```

新增解析规则时，请给出对应 CUDA 访存阶段、地址公式、适用条件和唯一补量。为关键边界添加测试，并在 [模型说明](docs/theory.md) 记录如何读取结果。提交独立 GPU 测量时请注明代码版本、设备、输入和计时边界。
