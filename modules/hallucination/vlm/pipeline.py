import json
import pathlib
import os

current_dir = pathlib.Path(__file__).parent
project_root = os.path.join(os.path.dirname(current_dir), "../../../")

class vlm_pipeline:
    def __init__(self, domain, data_path_dir=None):
        if domain == "hallusionbench":
            if data_path_dir:
                self.input_file_name = os.path.join(data_path_dir, "HallusionBench.json")
            else:
                self.input_file_name = os.path.join(project_root, "data/dataset/hallusion_bench/HallusionBench.json")
            self.save_response_path = os.path.join(project_root, "modules/hallucination/vlm/vlm_qa/exp/hallusion_output.json")
        elif domain.startswith("vh-test"):
            if domain == "vh-test-oeq":
                if data_path_dir:
                    self.input_file_name = os.path.join(data_path_dir, "OEQ_Benchmark.json")
                else:
                    self.input_file_name = os.path.join(project_root, "data/dataset/vh_test/OEQ_Benchmark.json")
            if domain == "vh-test-ynq":
                if data_path_dir:
                    self.input_file_name = os.path.join(data_path_dir, "YNQ_Benchmark.json")
                else:
                    self.input_file_name = os.path.join(project_root, "data/dataset/vh_test/YNQ_Benchmark.json")
            self.save_response_path = os.path.join(project_root, "modules/hallucination/vlm/vlm_qa/exp/hallusion_output.json")
        elif domain == "auto-detect":
            if data_path_dir:
                self.input_file_name = os.path.join(data_path_dir, "query_all.json")
            else:
                self.input_file_name = os.path.join(project_root, "modules/hallucination/vlm/vlm_autodetect/query/query_all.json")
            self.save_response_path = os.path.join(project_root, "modules/hallucination/vlm/vlm_autodetect/exp/response_all.json")
        elif domain == "vl-uctt":
            # vl-uctt 使用 benchmark 加载数据，不需要文件路径
            self.input_file_name = None
            self.save_response_path = None
        
        self.domain = domain

        if self.input_file_name and os.path.exists(self.input_file_name):
            with open(self.input_file_name, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        else:
            self.data = []
        self.data_result_list = self.data

