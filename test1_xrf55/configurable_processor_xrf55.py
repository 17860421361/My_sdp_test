import os
import sys
from pathlib import Path

# 当前脚本位于 test_xrf55 目录下，项目根目录是它的上一级。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 优先使用仓库中的 WSDP/SDP 源码，避免误导入环境里已安装的 wsdp 包。
LOCAL_WSDP_SRC = os.path.join(
    PROJECT_ROOT,
    "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main",
    "src",
)
if not os.path.isdir(os.path.join(LOCAL_WSDP_SRC, "wsdp")):
    raise FileNotFoundError(f"Local WSDP source not found: {LOCAL_WSDP_SRC}")
if LOCAL_WSDP_SRC not in sys.path:
    sys.path.insert(0, LOCAL_WSDP_SRC)

# sys.path 配置完成后再导入 wsdp，确保这里使用的是本地源码。
from wsdp import readers
from wsdp.processors.base_processor import _parse_file_info_from_filename


def load_xrf55_first_users(input_path, max_users=3):
    """使用源码解析和 reader，只读取按 user id 排序后的前 max_users 个用户。"""

    input_dir = Path(input_path)
    candidate_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and "truth" not in path.name
        and path.suffix.lower() in {".npy", ".dat"}
    )
    files_by_user = {}
    for file_path in candidate_files:
        parsed = _parse_file_info_from_filename(str(file_path), "xrf55")
        if parsed is None:
            continue
        user_id = int(parsed[0])
        files_by_user.setdefault(user_id, []).append(file_path)

    selected_users = sorted(files_by_user)[:max_users]
    selected_files = [
        file_path
        for user_id in selected_users
        for file_path in sorted(files_by_user[user_id])
    ]
    reader = readers.get_reader_class("xrf55")()
    if not selected_files:
        raise RuntimeError(f"没有找到 xrf55 可读文件: {input_path}")

    csi_data_list = []
    for file_path in selected_files:
        csi_data_list.extend(reader.read_file(str(file_path)))

    return csi_data_list, selected_users, len(selected_files), len(candidate_files)
