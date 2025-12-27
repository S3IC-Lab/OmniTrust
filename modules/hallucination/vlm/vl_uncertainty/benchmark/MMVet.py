from datasets import load_from_disk

class MMVet:

    def __init__(self, data_path_dir):
        self.ds = load_from_disk(data_path_dir)

    def obtain_size(self):
        return len(self.ds['train'])

    def retrieve(self, idx):
        row = self.ds['train'][idx]
        question = f"{row['question']}\nNOTE: Provide only the final answer. Do not provide unrelated details."
        result = {
            'idx': idx,
            'img': row['image'],
            'question': question,
            'gt_ans': row['answer'],
        }
        return result

if __name__ == "__main__":
    benchmark = MMVet()
    print(benchmark.retrieve(0))

