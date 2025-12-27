from datasets import load_from_disk

class LLaVABench:

    def __init__(self, data_path_dir):
        self.ds = load_from_disk(data_path_dir)
        
    def obtain_size(self):
        return len(self.ds['train'])

    def retrieve(self, idx):
        row = self.ds['train'][idx]
        result = {
            'idx': idx,
            'img': row['image'],
            'question': row['question'],
            'gt_ans': row['gpt_answer'],
        }
        return result

if __name__ == "__main__":
    benchmark = LLaVABench()
    print(benchmark.retrieve(0))

