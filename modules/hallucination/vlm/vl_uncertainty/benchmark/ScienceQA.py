from datasets import load_from_disk
import io
from PIL import Image




class ScienceQA:

    def __init__(self, data_path_dir):
        self.ds = load_from_disk(data_path_dir)

    def obtain_size(self):
        return len(self.ds['test'])

    def retrieve(self, idx):
        row = self.ds['test'][idx]
        question = row['question']
        question += '\n'
        choices = ""
        choice_numbers = ""
        for i, c in enumerate(row['choices']):
            choices += f'({i}): {c}\n'
            choice_numbers += f'{i}, '
        choice_numbers = choice_numbers[:-2]
        question += choices
        question += '\n'
        question += f'This is a single choice question, answer only with choice number in {choice_numbers}.'
        
        if row['image'] is None:
            image = None
        else:
            image_stream = io.BytesIO(row['image']['bytes'])
            image = Image.open(image_stream)
        result = {
            'idx': idx,
            'img': image,
            'question': question,
            'gt_ans': row['answer'],
            'num_c': len(row['choices']),
        }
        return result

if __name__ == "__main__":
    benchmark = ScienceQA()
    print(benchmark.retrieve(0))

