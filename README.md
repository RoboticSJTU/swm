# # 关键帧提取示例
```bash
conda create -n swm python=3.11 -y
conda activate swm
pip install -e .

# # 如果没有安装ffmpeg，请先安装ffmpeg
# sudo apt update
# sudo apt install ffmpeg

# 示例运行
python scripts/run_keyframe.py
```

# # 闭环生成pddl示例
```bash
# 安装fastdownward
git clone https://github.com/aibasel/downward.git
cd downward
./build.py

# 写入sii_api_key，VLM默认用便宜的gemini-3-flash-preview
cd ..
echo "SII_API_KEY=你的_api_key_here" > .env

# 运行示例
python scripts/run_domain_generation.py
```