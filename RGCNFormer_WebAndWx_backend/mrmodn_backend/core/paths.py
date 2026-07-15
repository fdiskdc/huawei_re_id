"""Resolve backend resource paths independently of the process working directory."""

import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_DIR = os.path.join(PROJECT_ROOT, "json")
HUMAN_JSON_PATH = os.path.join(JSON_DIR, "human.json")
MODEL_GRAPH_JSON_PATH = os.path.join(JSON_DIR, "model_graph.json")
MODEL_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "epoch_040.pt")
REID_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "best.pt")
LINEARFOLD_PATH = os.path.join(PROJECT_ROOT, "LinearFold", "linearfold")


def get_project_root() -> str:
    return PROJECT_ROOT


def get_json_dir() -> str:
    return JSON_DIR


def get_human_json_path() -> str:
    return HUMAN_JSON_PATH


def get_model_graph_json_path() -> str:
    return MODEL_GRAPH_JSON_PATH


def get_model_checkpoint_path() -> str:
    return MODEL_CHECKPOINT_PATH


def get_reid_checkpoint_path() -> str:
    return REID_CHECKPOINT_PATH


def get_linearfold_path() -> str:
    return LINEARFOLD_PATH
