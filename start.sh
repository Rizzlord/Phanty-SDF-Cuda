#!/usr/bin/env bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate comfyui-py312
python webapp/server.py
