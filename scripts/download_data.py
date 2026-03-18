"""
Dataset: Road Damage Detection (RDD2022) v10
Source: https://universe.roboflow.com/new-workspace-kj87b/road-damage-detection-iicdh

To download programmatically:
    1. Get free API key from roboflow.com
    2. Run: uv add roboflow
    3. Uncomment and run the code below

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("new-workspace-kj87b").project("road-damage-detection-iicdh")
dataset = project.version(10).download("yolov8", location="dataset/")
"""
